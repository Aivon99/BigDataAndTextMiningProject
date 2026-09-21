import Levenshtein
import chess
import numpy as np
import pandas as pd
import re
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from huggingface_hub import HfApi, snapshot_download
from peft import get_peft_model
from functools import partial
from torch.utils.data import DataLoader


def find_resumable_checkpoint(repo_id: str) -> str | None:
    """
    If `repo_id` already exists on the Hub and has a `last-checkpoint/`
    folder (written by a previous `Trainer` run with
    `hub_strategy="checkpoint"`), download it and return its local path —
    pass that straight to `Trainer.train(resume_from_checkpoint=...)` to
    pick training back up after an interruption (e.g. a Colab disconnect
    wiping the local runtime) instead of starting over from epoch 0.
    Returns None if there's nothing to resume from, so training starts fresh.
    """
    try:
        api = HfApi()
        if not api.repo_exists(repo_id=repo_id, repo_type="model"):
            return None

        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns="last-checkpoint/*",
        )
        candidate = Path(local_dir) / "last-checkpoint"
        if candidate.exists() and any(candidate.iterdir()):
            # Checkpoints saved by an earlier fp16 run include a gradient-scaler
            # state; training now uses bf16 (no scaler), so Trainer would crash
            # trying to load it. Drop the local copy (the Hub file is untouched).
            (candidate / "scaler.pt").unlink(missing_ok=True)
            print(f"Found a resumable checkpoint on the Hub for '{repo_id}': {candidate}")
            return str(candidate)
    except Exception as e:
        print(f"No resumable checkpoint found for '{repo_id}' ({e}); starting fresh.")

    return None

class Qwen35VisionDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id

    def __call__(self, examples):
        # Convert lists saved by .map() back to PyTorch tensors
        input_ids = [torch.tensor(ex["input_ids"]) if not isinstance(ex["input_ids"], torch.Tensor) else ex["input_ids"] for ex in examples]
        labels = [torch.tensor(ex["labels"]) if not isinstance(ex["labels"], torch.Tensor) else ex["labels"] for ex in examples]
        attention_mask = [torch.tensor(ex["attention_mask"]) if not isinstance(ex["attention_mask"], torch.Tensor) else ex["attention_mask"] for ex in examples]
        
        # Pad text sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        
        # Handle mm_token_type_ids (pad with 0)
        if "mm_token_type_ids" in examples[0] and examples[0]["mm_token_type_ids"] is not None:
            mm_token_type_ids = [torch.tensor(ex["mm_token_type_ids"]) if not isinstance(ex["mm_token_type_ids"], torch.Tensor) else ex["mm_token_type_ids"] for ex in examples]
            batch["mm_token_type_ids"] = torch.nn.utils.rnn.pad_sequence(mm_token_type_ids, batch_first=True, padding_value=0)
        
        # Handle pixel_values
        if "pixel_values" in examples[0] and examples[0]["pixel_values"] is not None:
            pixel_values_list = [torch.tensor(ex["pixel_values"]) if not isinstance(ex["pixel_values"], torch.Tensor) else ex["pixel_values"] for ex in examples]
            batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)
            
        # Handle image_grid_thw (concatenation along dimension 0)
        if "image_grid_thw" in examples[0] and examples[0]["image_grid_thw"] is not None:
            grid_list = [torch.tensor(ex["image_grid_thw"]) if not isinstance(ex["image_grid_thw"], torch.Tensor) else ex["image_grid_thw"] for ex in examples]
            batch["image_grid_thw"] = torch.cat(grid_list, dim=0)

        return batch


class Qwen35OnTheFlyCollator(Qwen35VisionDataCollator):
    """
    Takes RAW dataset rows (PIL `image`/`image_t1`, `prompt`, `target`, `task`)
    and runs the image processor per batch, optionally reordering patches first.

    Avoids `dataset.map(preprocess_function)`, which materialises every image's
    `pixel_values` (~6 MB each at 512px, ~20 GB for a 3200-sample train split)
    into an Arrow cache and forces the collator to rebuild tensors from nested
    Python lists on every step. Use with `remove_unused_columns=False` and
    `dataloader_num_workers > 0` so preprocessing overlaps with GPU compute.
    """

    def __init__(self, processor, strategy=None, grid_size=8):
        super().__init__(processor)
        self.strategy = strategy
        self.grid_size = grid_size

    def __call__(self, examples):
        processed = []
        for ex in examples:
            ex = dict(ex)
            if self.strategy is not None:
                for key in ("image", "image_t1"):
                    if ex.get(key) is not None:
                        ex[key] = reorder_chessboard_image(
                            ex[key], strategy=self.strategy, grid_size=self.grid_size
                        )
            processed.append(preprocess_function(ex, self.processor))
        return super().__call__(processed)


_FEN_RE = re.compile(
    r"(?:[pnbrqkPNBRQK1-8]+/){7}[pnbrqkPNBRQK1-8]+"
    r"(?: +[wb] +(?:-|[KQkq]+) +(?:-|[a-h][36]) +[0-9]+ +[0-9]+)?"
)


def extract_fen(text: str) -> str:
    """
    Pulls the first FEN-shaped string out of a model reply (board part plus, if
    present, the side/castling/en-passant/clock fields). A chatty answer like
    "The FEN is r3k2r/... w KQkq - 1 13." is scored on the FEN it contains
    instead of on all the surrounding words; if nothing FEN-shaped is found the
    stripped reply is returned unchanged, so garbage is still scored as garbage.
    """
    match = _FEN_RE.search(text)
    return match.group(0).strip() if match else text.strip()


def calculate_fen_exact_match(predicted_fen: str, ground_truth_fen: str) -> float:
    """
    Calculates FEN Exact Match: returns 1.0 if the predicted FEN string 
    matches the ground truth exactly, otherwise 0.0.
    """
    return 1.0 if predicted_fen.strip() == ground_truth_fen.strip() else 0.0


def calculate_levenshtein_metrics(predicted_fen: str, ground_truth_fen: str) -> dict:
    """
    Calculates the Levenshtein distance and Character Error Rate (CER) 
    between the predicted FEN and the ground truth FEN string.
    """
    pred = predicted_fen.strip()
    gt = ground_truth_fen.strip()
    
    dist = Levenshtein.distance(pred, gt)
    cer = dist / max(len(gt), 1)
    
    return {
        "levenshtein_distance": dist,
        "character_error_rate": cer
    }


def calculate_square_by_square_accuracy(predicted_fen: str, ground_truth_fen: str) -> float:
    """
    Calculates Square-by-Square Accuracy across the 64 squares of the chessboard.
    It parses both FEN strings into python-chess Board objects and compares 
    each of the 64 squares individually.
    """
    try:
        # We only consider the piece placement part of the FEN string (before the first space)
        pred_board = chess.Board(predicted_fen.strip().split()[0])
        gt_board = chess.Board(ground_truth_fen.strip().split()[0])
    except ValueError:
        # If the predicted FEN is malformed and cannot be parsed, accuracy is 0.0
        return 0.0
    
    correct_squares = 0
    total_squares = 64
    
    for square in chess.SQUARES:
        pred_piece = pred_board.piece_at(square)
        gt_piece = gt_board.piece_at(square)
        if pred_piece == gt_piece:
            correct_squares += 1
            
    return correct_squares / total_squares

def calculate_san_exact_match(predicted_move: str, ground_truth_move: str) -> int:
    """
    Computes Exact Match (EM) for chess moves in SAN notation.
    Returns 1 if predictions match the ground truth exactly, 0 otherwise.
    """
    return 1 if predicted_move.strip() == ground_truth_move.strip() else 0

def evaluate_chessboard_model_task_1(model, processor, dataset_split, model_name: str) -> pd.DataFrame:
    """
    Evaluates a given VLM model (vanilla or fine-tuned) on the chessboard Task 1 dataset 
    and returns a DataFrame containing predictions, metrics, and aggregate results.
    """
    model.eval()
    results_list = []

    # Iterate over the test dataset
    for test_sample in tqdm(dataset_split, desc=f"Evaluating {model_name}"):
        # 1. Extract fields from the sample based on the dataset structure
        task_prompt = test_sample["prompt"]
        ground_truth_fen = test_sample["target"]
        sample_id = test_sample["sample_id"]
        board_image = test_sample["image"]

        # 2. Prepare the multimodal input format for the model chat
        chat_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": board_image},
                    {"type": "text", "text": task_prompt},
                ]
            }
        ]

        # 3. Apply the processor's chat template
        formatted_text = processor.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)

        # 4. Tokenize inputs and move them to the model's device
        model_inputs = processor(
            text=[formatted_text],
            images=board_image,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # 5. Generate the prediction. A FEN is at most ~90 characters and a token
        # is never shorter than one character, so 100 new tokens can't cut off a
        # correct answer; fine-tuned models stop earlier on their own (EOS).
        with torch.no_grad():
            output_token_ids = model.generate(**model_inputs, max_new_tokens=100)

        # 6. Trim prompt tokens from the generated output
        trimmed_output_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_token_ids)
        ]
        raw_output = processor.batch_decode(
            trimmed_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        # Score the FEN inside a chatty reply rather than the whole reply
        predicted_fen_string = extract_fen(raw_output)

        # 7. Compute metrics for the current sample using predefined functions
        exact_match = calculate_fen_exact_match(predicted_fen_string, ground_truth_fen)
        levenshtein_res = calculate_levenshtein_metrics(predicted_fen_string, ground_truth_fen)
        square_accuracy = calculate_square_by_square_accuracy(predicted_fen_string, ground_truth_fen)

        # 8. Append row data including predictions and metrics
        results_list.append({
            "sample_id": sample_id,
            "ground_truth": ground_truth_fen,
            "predicted": predicted_fen_string,
            "raw_output": raw_output,
            "fen_exact_match": exact_match,
            "levenshtein_distance": levenshtein_res["levenshtein_distance"],
            "character_error_rate": levenshtein_res["character_error_rate"],
            "square_by_square_accuracy": square_accuracy
        })

    # Convert results into a Pandas DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Save sample-level results to CSV
    csv_filename = f"task1_{model_name.lower().replace(' ', '_')}_results.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"\nEvaluation completed for {model_name}! Results saved to {csv_filename}.")

    # Compute global aggregate metrics across the dataset
    mean_fen_em = results_df["fen_exact_match"].mean()
    mean_cer = results_df["character_error_rate"].mean()
    mean_square_acc = results_df["square_by_square_accuracy"].mean()
    mean_levenshtein = results_df["levenshtein_distance"].mean()

    # Create the model summary row for the global comparison table
    model_summary_df = pd.DataFrame([
        {
            "model_name": model_name,
            "fen_exact_match": mean_fen_em,
            "character_error_rate": mean_cer,
            "square_by_square_accuracy": mean_square_acc,
            "levenshtein_distance": mean_levenshtein
        }
    ])

    return results_df, model_summary_df

def evaluate_chessboard_model_task_2(model, processor, dataset_split, model_name: str) -> pd.DataFrame:
    """
    Evaluates a given VLM model (vanilla or fine-tuned) on the chessboard Task 2 dataset 
    (Move Prediction) and returns a DataFrame containing predictions, metrics, and aggregate results.
    """
    model.eval()
    results_list = []

    # Iterate over the test dataset
    for test_sample in tqdm(dataset_split, desc=f"Evaluating {model_name} (Task 2)"):
        # 1. Extract fields from the sample based on the Task 2 structure
        task_prompt = test_sample["prompt"]
        ground_truth_move = test_sample["target"]
        sample_id = test_sample["sample_id"]
        board_image = test_sample["image"]

        # 2. Prepare the multimodal input format for the model chat
        chat_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": board_image},
                    {"type": "text", "text": task_prompt},
                ]
            }
        ]

        # 3. Apply the processor's chat template
        formatted_text = processor.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)

        # 4. Tokenize inputs and move them to the model's device
        model_inputs = processor(
            text=[formatted_text],
            images=board_image,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # 5. Generate the prediction (max_new_tokens is smaller for SAN notation)
        with torch.no_grad():
            output_token_ids = model.generate(**model_inputs, max_new_tokens=16)

        # 6. Trim prompt tokens from the generated output
        trimmed_output_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_token_ids)
        ]
        predicted_move = processor.batch_decode(
            trimmed_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        # 7. Compute metrics for the current sample
        exact_match = calculate_san_exact_match(predicted_move, ground_truth_move)

        # 8. Append row data including predictions and metrics
        results_list.append({
            "sample_id": sample_id,
            "ground_truth": ground_truth_move,
            "predicted": predicted_move,
            "exact_match": exact_match
        })

    # Convert results into a Pandas DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Save sample-level results to CSV
    csv_filename = f"task2_{model_name.lower().replace(' ', '_')}_results.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"\nEvaluation completed for {model_name}! Results saved to {csv_filename}.")

    # Compute global aggregate metrics across the dataset
    mean_em = results_df["exact_match"].mean()

    # Create the model summary row for the global comparison table
    model_summary_df = pd.DataFrame([
        {
            "model_name": model_name,
            "exact_match": mean_em
        }
    ])

    return results_df, model_summary_df


def evaluate_chessboard_model_task_3(model, processor, dataset_split, model_name: str) -> pd.DataFrame:
    """
    Evaluates a given VLM model (vanilla or fine-tuned) on the chessboard Task 3 dataset 
    (Dual-Image Delta Move) and returns a DataFrame containing predictions, metrics, and aggregate results.
    """
    model.eval()
    results_list = []

    # Iterate over the dataset split
    for test_sample in tqdm(dataset_split, desc=f"Evaluating {model_name} on Task 3"):
        # 1. Extract fields from the sample based on the dataset structure
        task_prompt = test_sample["prompt"]
        ground_truth_move = test_sample["target"]
        sample_id = test_sample["sample_id"]
        
        # Frame 1 (State t) and Frame 2 (State t+1) images
        frame_t_image = test_sample["image"]
        frame_t_plus_1_image = test_sample["image_t1"]

        # 2. Prepare the multimodal input format for the model chat with two images
        chat_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame_t_image},
                    {"type": "image", "image": frame_t_plus_1_image},
                    {"type": "text", "text": task_prompt},
                ]
            }
        ]

        # 3. Apply the processor's chat template
        formatted_text = processor.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)

        # 4. Tokenize inputs and pass both images as a list, then move them to the model's device
        model_inputs = processor(
            text=[formatted_text],
            images=[frame_t_image, frame_t_plus_1_image],
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # 5. Generate the prediction
        with torch.no_grad():
            output_token_ids = model.generate(**model_inputs, max_new_tokens=128)

        # 6. Trim prompt tokens from the generated output
        trimmed_output_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_token_ids)
        ]
        predicted_move_string = processor.batch_decode(
            trimmed_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        # 7. Compute Exact Match (EM) metric using the predefined function
        exact_match = calculate_san_exact_match(predicted_move_string, ground_truth_move)

        # 8. Append row data including predictions and metrics
        results_list.append({
            "sample_id": sample_id,
            "ground_truth": ground_truth_move,
            "predicted": predicted_move_string,
            "exact_match": exact_match
        })

    # Convert results into a Pandas DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Save sample-level results to CSV
    csv_filename = f"task3_{model_name.lower().replace(' ', '_')}_results.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"\nEvaluation completed for {model_name} on Task 3! Results saved to {csv_filename}.")

    # Compute global aggregate metrics across the dataset split
    mean_em = results_df["exact_match"].mean()

    # Create the model summary row for the global comparison table (omitting the task number)
    model_summary_df = pd.DataFrame([
        {
            "model_name": model_name,
            "exact_match": mean_em
        }
    ])

    return results_df, model_summary_df

def preprocess_function(sample, processor):
    task = sample.get("task", "task1")
    prompt_text = sample["prompt"]
    target_text = sample["target"]

    if task in ["task1", "task2"]:
        board_image = sample["image"]
        chat_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": board_image},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target_text},
                ],
            }
        ]
        images_input = [board_image]

    elif task == "task3":
        img_t = sample.get("image") or sample.get("image_t")
        img_t1 = sample.get("image_t1")

        chat_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_t},
                    {"type": "image", "image": img_t1},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target_text},
                ],
            }
        ]
        images_input = [img_t, img_t1]

    else:
        raise ValueError(f"Unsupported task type: '{task}'")

    # Genera la stringa tramite il template nativo
    text = processor.apply_chat_template(
        chat_messages, 
        tokenize=False, 
        add_generation_prompt=False
    )

    # Processa testo e immagini insieme
    batch = processor(
        text=[text],
        images=images_input,
        padding=False,
        return_tensors="pt"
    )

    # Estrazione sicura dei tensori rimuovendo la dimensione del batch iniziale
    result = {
        "input_ids": batch["input_ids"][0],
        "attention_mask": batch["attention_mask"][0],
        "pixel_values": batch["pixel_values"],
    }

    if "mm_token_type_ids" in batch:
        result["mm_token_type_ids"] = batch["mm_token_type_ids"][0]

    if "image_grid_thw" in batch:
        result["image_grid_thw"] = batch["image_grid_thw"]

    # Configura i labels per il Causal LM
    labels = result["input_ids"].clone()
    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100
    result["labels"] = labels

    return result


def get_patch_reordering_indices(strategy="raster", grid_size=8):
    """
    Generates patch reordering index maps for an 8x8 chessboard grid.
    Strategies supported: 'raster', 'zigzag', 'spiral', 'file_wise', 'rank_wise'
    """
    total_patches = grid_size * grid_size
    indices = np.arange(total_patches).reshape(grid_size, grid_size)

    if strategy == "raster":
        return [int(x) for x in indices.flatten()]

    elif strategy == "zigzag":
        reordered = []
        for r in range(grid_size):
            row = indices[r, :]
            if r % 2 == 1:
                row = row[::-1]
            reordered.extend(row)
        return [int(x) for x in reordered]

    elif strategy == "spiral":
        reordered = []
        top, bottom, left, right = 0, grid_size - 1, 0, grid_size - 1
        while top <= bottom and left <= right:
            for c in range(left, right + 1):
                reordered.append(indices[top, c])
            top += 1
            for r in range(top, bottom + 1):
                reordered.append(indices[r, right])
            right -= 1
            if top <= bottom:
                for c in range(right, left - 1, -1):
                    reordered.append(indices[bottom, c])
                bottom -= 1
            if left <= right:
                for r in range(bottom, top - 1, -1):
                    reordered.append(indices[r, left])
                left += 1
        return [int(x) for x in reordered]

    elif strategy == "file_wise": # Column-wise
        return [int(x) for x in indices.T.flatten()]

    elif strategy == "rank_wise": # Row-wise (same as raster)
        return [int(x) for x in indices.flatten()]

    else:
        raise ValueError(f"Unknown reordering strategy: {strategy}")

def apply_patch_permutation(image, permutation, grid_size=8, img_size=512):
    """
    Slices `image` into a grid_size x grid_size grid of tiles and rearranges
    them according to `permutation` (a length grid_size**2 sequence of tile
    indices in raster order, e.g. from `get_patch_reordering_indices()` or
    sampled from a learned policy such as `PlackettLucePatchPolicy`).

    This is the low-level primitive both `reorder_chessboard_image` (fixed,
    named strategies) and the learned-reordering training loop
    (`src/training/learned_reordering.py`, a different permutation per step)
    build on.
    """
    image = image.resize((img_size, img_size))
    tile_size = img_size // grid_size

    # 1. Split image into individual square tiles
    tiles = []
    for r in range(grid_size):
        for c in range(grid_size):
            box = (c * tile_size, r * tile_size, (c + 1) * tile_size, (r + 1) * tile_size)
            tiles.append(image.crop(box))

    # 2. Rearrange tiles based on the given permutation
    reordered_tiles = [tiles[int(i)] for i in permutation]

    # 3. Stitch tiles back together into a new image
    new_image = Image.new("RGB", (img_size, img_size))
    for idx, tile in enumerate(reordered_tiles):
        r = idx // grid_size
        c = idx % grid_size
        new_image.paste(tile, (c * tile_size, r * tile_size))

    return new_image


def reorder_chessboard_image(image, strategy="raster", grid_size=8, img_size=512):
    """
    Slices a chessboard PIL Image into an 8x8 grid of tiles and rearranges
    them according to one of the fixed, named strategies from
    `get_patch_reordering_indices()`.
    """
    reorder_indices = get_patch_reordering_indices(strategy=strategy, grid_size=grid_size)
    return apply_patch_permutation(image, reorder_indices, grid_size=grid_size, img_size=img_size)



def finetune_and_push_chessboard_model(
    strategy_name,
    dataset,
    processor,
    model,
    peft_config,
    task,
    hf_org_prefix="bdatm-project",
    repo_root=None,
    num_train_epochs=10,
    early_stopping_patience=2,
):
    print(f"\n==============================================")
    print(
        f"Starting pipeline for TASK: {task.upper()} | STRATEGY:"
        f" {strategy_name.upper()}"
    )
    print(f"==============================================")

    if task not in ("task1", "task2", "task3"):
        raise ValueError(f"Unknown task: {task}")

    # 1-2. Reordering and preprocessing happen on the fly inside the collator
    # (both frames for task3), so no reordered/tokenized copy of the dataset
    # is ever materialised on disk.

    # 3. Apply PEFT/LoRA to the provided model instance
    print("Applying LoRA to the provided model...")
    lora_model_instance = get_peft_model(model, peft_config)

    # 4. Resolve the target Hub repo now (needed before training starts) and
    # check whether an earlier, interrupted run already left a resumable
    # checkpoint there (e.g. after a Colab disconnect wiped the local runtime).
    repo_id_target = f"{hf_org_prefix}/qwen-{task}-{strategy_name}-lora"
    resume_checkpoint = find_resumable_checkpoint(repo_id_target)

    # 5. Configure Training Arguments for this specific run. Trains for up to
    # `num_train_epochs`, but relies on the validation set (via
    # EarlyStoppingCallback below) to stop once eval_loss stops improving,
    # and to keep the best-performing checkpoint rather than just the last one.
    # push_to_hub + hub_strategy="checkpoint" uploads a fully resumable
    # checkpoint (optimizer/scheduler/RNG state included) after every epoch,
    # so an interruption at epoch i loses at most that epoch's progress.
    training_args_instance = TrainingArguments(
        output_dir=f"./temp_{task}_{strategy_name}_output",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=num_train_epochs,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        push_to_hub=True,
        hub_model_id=repo_id_target,
        hub_strategy="checkpoint",
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    )

    # 6. Initialize Data Collator and Trainer
    data_collator = Qwen35OnTheFlyCollator(processor=processor, strategy=strategy_name)

    trainer_instance = Trainer(
        model=lora_model_instance,
        args=training_args_instance,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    # 7. Train (or resume) the model
    if resume_checkpoint:
        print(f"Resuming training for {task} ({strategy_name} reordering) from {resume_checkpoint}...")
    else:
        print(f"Training model for {task} with {strategy_name} reordering...")
    trainer_instance.train(resume_from_checkpoint=resume_checkpoint)

    # 8. Push final weights and processor directly to Hugging Face Hub
    print(
        f"Pushing model and processor to Hugging Face Hub: {repo_id_target}..."
    )

    trainer_instance.model.push_to_hub(
        repo_id_target,
        commit_message=(
            f"Training complete for {task} using {strategy_name} reordering"
            " strategy"
        ),
    )
    processor.push_to_hub(repo_id_target)

    print(f"Finished! Successfully uploaded to Hub: {repo_id_target}")

    return trainer_instance.model
