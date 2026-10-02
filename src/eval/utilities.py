import json
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
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
try:
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
except ImportError:  # older huggingface_hub
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
from transformers import TrainerCallback
from peft import get_peft_model, set_peft_model_state_dict
from peft.utils import load_peft_weights
from functools import partial
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset

from eval.diagnostics import (
    FEN_DIAGNOSTIC_COLUMNS,
    SAN_DIAGNOSTIC_COLUMNS,
    analyze_fen_predictions,
    analyze_san_predictions,
)


# Identifies the training objective the current code implements. Bump it
# whenever preprocess_function / the collators / the loss change in a way
# that makes an older checkpoint's weights (or its eval_loss history)
# incomparable with a fresh run. find_resumable_checkpoint refuses to resume
# a run stamped with a different recipe (or none at all).
#
# Why this exists: every Task 3 run was started before the prompt-masking fix
# (commit 9df1c7d, loss over the whole sequence) and resumed after it (loss
# over the answer only). eval_loss jumped 2.76 -> 0.75 at the resume point,
# early stopping / best-checkpoint selection compared incomparable numbers,
# and only the last 2-6 epochs trained on the real objective -- Task 3 then
# scored ~0% for every model. Task 1 was trained entirely before the fix.
TRAINING_RECIPE = "answer-only-loss-v1"
_RECIPE_FILE = "training_recipe.json"


def read_training_recipe(repo_id: str) -> str | None:
    """
    The recipe a Hub model repo was trained with, or None if the repo or its
    recipe file doesn't exist. Any other error (network, auth, rate limit)
    is raised: a None here makes find_resumable_checkpoint delete the
    checkpoint and retrain from scratch, which must never happen just
    because the Hub was briefly unreachable.
    """
    try:
        path = hf_hub_download(repo_id, _RECIPE_FILE, repo_type="model")
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    return json.loads(Path(path).read_text()).get("recipe")


def stamp_training_recipe(repo_id: str, recipe: str = TRAINING_RECIPE) -> None:
    """Records `recipe` on the Hub repo (creating it if needed) so a later resume can check it."""
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=json.dumps({"recipe": recipe}).encode("utf-8"),
        path_in_repo=_RECIPE_FILE,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Training recipe: {recipe}",
    )


# Hub folders holding training state for a run in progress: the resumable
# Trainer checkpoint, and the best epoch's adapter (HubBestAdapterCallback).
_RUN_STATE_FOLDERS = ("last-checkpoint", "best-checkpoint")


def find_resumable_checkpoint(repo_id: str) -> str | None:
    """
    If `repo_id` already exists on the Hub and has a `last-checkpoint/`
    folder (written by a previous `Trainer` run with
    `hub_strategy="checkpoint"`), download it and return its local path —
    pass that straight to `Trainer.train(resume_from_checkpoint=...)` (or
    `train_or_load_finished`) to pick training back up after an
    interruption (e.g. a Colab disconnect wiping the local runtime) instead
    of starting over from epoch 0. Returns None if there's nothing to
    resume from, so training starts fresh.

    Only resumes a checkpoint stamped with the current TRAINING_RECIPE; an
    unstamped or differently-stamped run's state is deleted from the Hub
    and training restarts from scratch. Whenever training is about to start
    fresh, the repo is stamped with the current recipe.

    Hub errors are raised rather than treated as "nothing to resume": a
    silent fresh start would retrain for hours and overwrite the checkpoint.
    """
    api = HfApi()
    if not api.repo_exists(repo_id=repo_id, repo_type="model"):
        stamp_training_recipe(repo_id)
        return None

    recipe = read_training_recipe(repo_id)
    if recipe != TRAINING_RECIPE:
        print(
            f"Ignoring the checkpoint in '{repo_id}': it was trained with recipe {recipe!r}, "
            f"the current code uses {TRAINING_RECIPE!r}. Starting fresh."
        )
        # Drop the stale run state before stamping: otherwise a disconnect
        # before the first new epoch is saved would leave a correctly-stamped
        # repo still holding the old checkpoint, and the next run would resume it.
        files = api.list_repo_files(repo_id, repo_type="model")
        for folder in _RUN_STATE_FOLDERS:
            if any(f.startswith(folder + "/") for f in files):
                api.delete_folder(
                    folder, repo_id=repo_id, repo_type="model",
                    commit_message=f"Remove {folder} trained with recipe {recipe!r}",
                )
        stamp_training_recipe(repo_id)
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
    return None


def training_is_complete(checkpoint_dir: str) -> bool:
    """
    True if the Trainer checkpoint in `checkpoint_dir` belongs to a run that
    already finished -- stopped early (EarlyStoppingCallback) or reached
    max_steps -- as recorded in its trainer_state.json.
    """
    state = json.loads((Path(checkpoint_dir) / "trainer_state.json").read_text())
    control = state.get("stateful_callbacks", {}).get("TrainerControl", {}).get("args", {})
    return bool(control.get("should_training_stop")) or state["global_step"] >= state["max_steps"]


def train_or_load_finished(trainer, repo_id: str, resume_checkpoint: str | None) -> bool:
    """
    `trainer.train(resume_from_checkpoint=...)`, except when the Hub run in
    `repo_id` already finished: then the final adapter pushed to the repo
    root is loaded into `trainer.model` and training is skipped. Returns
    True if training ran, False if it was skipped (nothing new to push).

    Needed because resuming a finished run is not a no-op: Trainer resets
    `should_training_stop` in on_train_begin, so a run that early-stopped at
    epoch 9/10 trains epoch 10 again, and the re-push would replace the
    published weights (invalidating every cached evaluation of them).
    """
    if resume_checkpoint and training_is_complete(resume_checkpoint):
        print(f"Training for '{repo_id}' already finished on the Hub -- skipping training and loading its final adapter.")
        set_peft_model_state_dict(trainer.model, load_peft_weights(repo_id))
        return False
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    return True


class HubBestAdapterCallback(TrainerCallback):
    """
    Keeps the best epoch's LoRA adapter on the Hub (`best-checkpoint/`), so
    `load_best_model_at_end` still works when a run is resumed in a new
    Colab session.

    Trainer only knows the best checkpoint by its local path
    (`./<output_dir>/checkpoint-N`); hub_strategy="checkpoint" pushes only the
    latest checkpoint. After a resume in a fresh runtime that path is gone,
    Trainer logs "The best checkpoint ... does not exist anymore. Ignoring
    it" and ends with the LAST epoch's weights -- which is what happened to
    the Task 1 raster run (published epoch 9, eval_loss 0.119, instead of
    the best epoch, 0.110).

    on_save: when the checkpoint just written is the new best, upload its
    adapter + a best.json (step, metric, recipe) to `best-checkpoint/`.
    on_train_end: if Trainer could not restore the best model locally,
    load it from `best-checkpoint/` -- only if best.json matches this run's
    best_metric and recipe, so a stale upload is never used.
    """

    _FOLDER = "best-checkpoint"

    def __init__(self, repo_id: str):
        self.repo_id = repo_id

    def on_save(self, args, state, control, **kwargs):
        best = state.best_model_checkpoint
        if not best or Path(best).name != f"checkpoint-{state.global_step}" or not Path(best).is_dir():
            return
        from huggingface_hub import CommitOperationAdd

        ops = [
            CommitOperationAdd(f"{self._FOLDER}/{name}", str(Path(best) / name))
            for name in ("adapter_model.safetensors", "adapter_config.json")
            if (Path(best) / name).exists()
        ]
        if not ops:
            return
        meta = {"global_step": state.global_step, "best_metric": state.best_metric, "recipe": TRAINING_RECIPE}
        ops.append(CommitOperationAdd(f"{self._FOLDER}/best.json", json.dumps(meta).encode("utf-8")))
        HfApi().create_commit(
            repo_id=self.repo_id, repo_type="model", operations=ops,
            commit_message=f"Best adapter so far: step {state.global_step} ({args.metric_for_best_model}={state.best_metric:.4f})",
        )

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not args.load_best_model_at_end or state.best_metric is None or model is None:
            return
        local_best = state.best_model_checkpoint
        if local_best and Path(local_best).is_dir():
            return  # Trainer already restored it from disk
        try:
            meta_path = hf_hub_download(self.repo_id, f"{self._FOLDER}/best.json", repo_type="model")
        except EntryNotFoundError:
            print(f"[best-checkpoint] No best adapter on the Hub for '{self.repo_id}'; keeping the last epoch's weights.")
            return
        meta = json.loads(Path(meta_path).read_text())
        if meta.get("recipe") != TRAINING_RECIPE or abs(meta["best_metric"] - state.best_metric) > 1e-6:
            print(f"[best-checkpoint] Hub best adapter {meta} doesn't match this run's best_metric "
                  f"{state.best_metric}; keeping the last epoch's weights.")
            return
        set_peft_model_state_dict(model, load_peft_weights(self.repo_id, subfolder=self._FOLDER))
        print(f"[best-checkpoint] Restored the best adapter (step {meta['global_step']}, "
              f"{args.metric_for_best_model}={meta['best_metric']:.4f}) from the Hub.")


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

    # Convert results into a Pandas DataFrame, with per-field FEN columns
    # (board vs side-to-move/castling/en-passant/clocks, see eval/diagnostics.py)
    results_df = analyze_fen_predictions(pd.DataFrame(results_list))
    
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
            "levenshtein_distance": mean_levenshtein,
            **{col: results_df[col].mean() for col in FEN_DIAGNOSTIC_COLUMNS},
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
        raw_output = processor.batch_decode(
            trimmed_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        # No SAN-move extraction yet (unlike Task 1's extract_fen): a bare
        # square coordinate like "c1" is itself valid SAN, so a naive
        # first-match regex risks pulling an unrelated square mentioned in
        # the model's reasoning rather than its actual answer. predicted ==
        # raw_output for now; raw_output is still kept as its own column so
        # a chatty vs. clean-but-wrong failure mode can be told apart later
        # (needed for the spec's qualitative error analysis) without
        # re-running inference.
        predicted_move = raw_output

        # 7. Compute metrics for the current sample
        exact_match = calculate_san_exact_match(predicted_move, ground_truth_move)

        # 8. Append row data including predictions and metrics
        results_list.append({
            "sample_id": sample_id,
            "ground_truth": ground_truth_move,
            "predicted": predicted_move,
            "raw_output": raw_output,
            "exact_match": exact_match
        })

    # Convert results into a Pandas DataFrame, with move-level diagnostics
    # (legal / same move / from-to squares, see eval/diagnostics.py)
    results_df = analyze_san_predictions(pd.DataFrame(results_list), dataset_split["fen"])

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
            "exact_match": mean_em,
            **{col: results_df[col].mean() for col in SAN_DIAGNOSTIC_COLUMNS},
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
            # SAN answers are a few tokens; same budget as Task 2
            output_token_ids = model.generate(**model_inputs, max_new_tokens=16)

        # 6. Trim prompt tokens from the generated output
        trimmed_output_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_token_ids)
        ]
        raw_output = processor.batch_decode(
            trimmed_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        # See the matching comment in evaluate_chessboard_model_task_2 --
        # no SAN-move extraction here yet, same false-positive risk applies.
        predicted_move_string = raw_output

        # 7. Compute Exact Match (EM) metric using the predefined function
        exact_match = calculate_san_exact_match(predicted_move_string, ground_truth_move)

        # 8. Append row data including predictions and metrics
        results_list.append({
            "sample_id": sample_id,
            "ground_truth": ground_truth_move,
            "predicted": predicted_move_string,
            "raw_output": raw_output,
            "exact_match": exact_match
        })

    # Convert results into a Pandas DataFrame, with move-level diagnostics
    # (legal / same move / from-to squares, see eval/diagnostics.py)
    results_df = analyze_san_predictions(pd.DataFrame(results_list), dataset_split["fen"])

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
            "exact_match": mean_em,
            **{col: results_df[col].mean() for col in SAN_DIAGNOSTIC_COLUMNS},
        }
    ])

    return results_df, model_summary_df


_EVAL_FUNCS = {
    "task1": evaluate_chessboard_model_task_1,
    "task2": evaluate_chessboard_model_task_2,
    "task3": evaluate_chessboard_model_task_3,
}


def _eval_results_repo_id(task: str, hf_org_prefix: str) -> str:
    return f"{hf_org_prefix}/evaluation-results-{task}"


_RESULTS_PARQUET = "data/train-00000-of-00001.parquet"

# Dataset card for the results tables: declares the data files but NOT the
# column schema, so readers infer it from the parquet itself. A card written
# by Dataset.push_to_hub pins the columns, and push_to_hub does not update
# them when a column is added -- after `adapter_sha` was introduced, the
# Task 1 card still listed the old 5 columns and every
# load_dataset("evaluation-results-task1") raised a CastError.
_RESULTS_CARD = """---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
Model-comparison table for this task: one row per evaluated model, written by
`push_results_table` in `src/eval/utilities.py`. Per-sample predictions are in
`per_sample/`.
"""


def load_results_table(repo_id: str) -> pd.DataFrame | None:
    """
    Reads a results table straight from its parquet file(s) on the Hub,
    or returns None if the repo / table doesn't exist yet. Bypasses
    `load_dataset`, so a stale schema in the dataset card can't break it.
    Network/auth errors are raised, not mistaken for "nothing cached".
    """
    api = HfApi()
    if not api.repo_exists(repo_id=repo_id, repo_type="dataset"):
        return None
    files = sorted(
        f for f in api.list_repo_files(repo_id, repo_type="dataset")
        if f.startswith("data/") and f.endswith(".parquet")
    )
    if not files:
        return None
    return pd.concat(
        [pd.read_parquet(hf_hub_download(repo_id, f, repo_type="dataset")) for f in files],
        ignore_index=True,
    )


def push_results_table(repo_id: str, table: pd.DataFrame, commit_message: str = "Update results table") -> None:
    """
    Replaces the results table on the Hub with `table` in a single commit:
    one parquet file plus a schema-free dataset card (see _RESULTS_CARD),
    removing any other parquet shard so readers never see a mix of old and new.
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    stale = [
        f for f in api.list_repo_files(repo_id, repo_type="dataset")
        if f.startswith("data/") and f.endswith(".parquet") and f != _RESULTS_PARQUET
    ]
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        operations=[
            CommitOperationAdd(_RESULTS_PARQUET, table.reset_index(drop=True).to_parquet(index=False)),
            CommitOperationAdd("README.md", _RESULTS_CARD.encode("utf-8")),
            *[CommitOperationDelete(f) for f in stale],
        ],
    )


def load_or_init_results_table(task: str, hf_org_prefix: str = "bdatm-project") -> pd.DataFrame:
    """
    Loads the running model-comparison table for `task` from the Hub (the
    same `{hf_org_prefix}/evaluation-results-{task}` dataset the notebooks
    already push to at the end), or returns an empty DataFrame if nothing's
    been pushed yet.

    Call this once, early, to initialise `all_models_results` -- instead of
    `all_models_results = <first model's summary_df>`, which silently
    discards everything already computed in a previous session the moment
    it runs. With this, running any evaluation cell (in any order, in a
    fresh session or not) always appends to what's already known rather
    than starting over or crashing with a NameError.
    """
    repo_id = _eval_results_repo_id(task, hf_org_prefix)
    table = load_results_table(repo_id)
    if table is None:
        print(f"No existing comparison table found for {task} on the Hub (or repo doesn't exist yet) -- starting fresh.")
        return pd.DataFrame()
    print(f"Loaded existing comparison table for {task} from '{repo_id}' ({len(table)} model(s) already evaluated).")
    return table


def append_or_replace(all_results: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends `summary_df` (one model's comparison row) to `all_results`,
    replacing any existing row for the same `model_name` instead of
    duplicating it. Needed because `all_results` may already contain a row
    for this model (loaded from the Hub via `load_or_init_results_table` or
    `evaluate_with_cache`'s own cache hit) -- a plain `pd.concat` would
    otherwise leave two rows for the same model in the comparison table.
    """
    if "model_name" in all_results.columns and len(summary_df) > 0:
        model_name = summary_df["model_name"].iloc[0]
        all_results = all_results[all_results["model_name"] != model_name]
    return pd.concat([all_results, summary_df], ignore_index=True)


_ADAPTER_WEIGHTS = "adapter_model.safetensors"


def adapter_fingerprint(repo_id: str, revision: str | None = None) -> str:
    """
    sha256 of the LoRA weights file in `repo_id` (at `revision`, default the
    latest commit), read from the Hub's file metadata -- nothing is downloaded.

    Used as the evaluation-cache key instead of the repo's commit sha: the
    commit changes on every push to the repo (model-card or processor
    re-uploads, which also differ whenever Colab installs newer library
    versions; checkpoint pushes; the recipe stamp), none of which change
    the weights being evaluated.
    """
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    for sibling in info.siblings:
        if sibling.rfilename == _ADAPTER_WEIGHTS and sibling.lfs is not None:
            lfs = sibling.lfs
            return lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256
    raise FileNotFoundError(f"No {_ADAPTER_WEIGHTS} in '{repo_id}' at revision {revision or 'main'}")


def _cached_adapter_matches(cached, adapter_repo_id: str, current_fingerprint: str) -> bool:
    """
    Whether a cached row's `adapter_sha` refers to the weights currently in
    `adapter_repo_id`. Rows written since the fingerprint change store the
    weights' sha256 directly; older rows stored the repo commit sha, which is
    resolved to the weights fingerprint at that commit -- so those cached
    evaluations stay valid as long as the weights haven't changed.
    """
    if not isinstance(cached, str) or not cached:
        return False
    if cached == current_fingerprint:
        return True
    try:
        return adapter_fingerprint(adapter_repo_id, revision=cached) == current_fingerprint
    except (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError, FileNotFoundError):
        return False


def evaluate_with_cache(
    task: str,
    model,
    processor,
    dataset_split,
    model_name: str,
    hf_org_prefix: str = "bdatm-project",
    force: bool = False,
    adapter_repo_id: str | None = None,
):
    """
    Drop-in replacement for `evaluate_chessboard_model_task_{1,2,3}` (same
    return shape: `(results_df, summary_df)`) that checks
    `{hf_org_prefix}/evaluation-results-{task}` on the Hub first. If a row
    for this exact `model_name` is already there, returns it directly --
    no model.generate() calls, no GPU time -- instead of re-running
    inference that can take tens of minutes. Otherwise runs the real
    evaluation and pushes both the summary row (upserted into the shared
    comparison table) and the per-sample results (as their own CSV, so
    qualitative error analysis survives a session restart too) to the Hub,
    so the *next* run/session hits the cache instead.

    Pass `force=True` to bypass the cache and re-evaluate anyway (e.g.
    after a metric or prompt change makes the cached numbers stale).

    Pass `adapter_repo_id` (the Hub repo the evaluated LoRA weights come
    from) to tie the cache entry to those weights: the row stores
    `adapter_sha` (see `adapter_fingerprint`), and a cached row is only
    reused while the repo still holds the same weights. Without it,
    retraining a model under the same `model_name` would keep serving the
    old model's cached numbers.
    """
    if task not in _EVAL_FUNCS:
        raise ValueError(f"evaluate_with_cache: unknown task '{task}'")

    repo_id = _eval_results_repo_id(task, hf_org_prefix)
    slug = model_name.lower().replace(" ", "_")
    adapter_sha = adapter_fingerprint(adapter_repo_id) if adapter_repo_id else None

    if not force:
        # Read errors propagate on purpose: swallowing them used to turn a
        # broken table into a silent cache miss, re-running every evaluation.
        table = load_results_table(repo_id)
        if table is not None and "model_name" in table.columns:
            match = table[table["model_name"] == model_name]
            if adapter_sha is not None and len(match) > 0:
                cached_sha = match["adapter_sha"].iloc[0] if "adapter_sha" in match.columns else None
                if not _cached_adapter_matches(cached_sha, adapter_repo_id, adapter_sha):
                    print(
                        f"[{task}] Cached evaluation for '{model_name}' was for different weights than "
                        f"those now in '{adapter_repo_id}' -- re-evaluating."
                    )
                    match = match.iloc[0:0]
            if len(match) > 0:
                print(f"[{task}] Using cached evaluation for '{model_name}' -- skipping inference.")
                summary_df = match.reset_index(drop=True)
                try:
                    sample_path = hf_hub_download(repo_id, f"per_sample/{slug}.csv", repo_type="dataset")
                    results_df = pd.read_csv(sample_path)
                except Exception:
                    results_df = None  # summary is cached but the per-sample file isn't (e.g. an older cache entry)
                return results_df, summary_df

    results_df, summary_df = _EVAL_FUNCS[task](
        model=model, processor=processor, dataset_split=dataset_split, model_name=model_name
    )
    if adapter_sha is not None:
        summary_df["adapter_sha"] = adapter_sha

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=results_df.to_csv(index=False).encode("utf-8"),
        path_in_repo=f"per_sample/{slug}.csv",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Cache per-sample results for '{model_name}' ({task})",
    )

    # Upsert into the existing table. A read failure must raise here: the old
    # fallback (push just this one row) wiped every other model's results.
    existing = load_results_table(repo_id)
    updated = append_or_replace(existing if existing is not None else pd.DataFrame(), summary_df)
    push_results_table(repo_id, updated, commit_message=f"Cache evaluation for '{model_name}' ({task})")
    print(f"[{task}] Cached evaluation for '{model_name}' -- pushed to '{repo_id}'.")

    return results_df, summary_df


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

    # Labels for the Causal LM: loss must cover ONLY the assistant's reply,
    # not the prompt (instructions + images + question). Without this mask
    # -- the previous behaviour -- the model also gets gradient on copying
    # back its own prompt, which it already predicts almost perfectly under
    # teacher forcing; for short targets like SAN moves (Task 2/3, a handful
    # of tokens) this dilutes the useful signal in the reported average loss
    # to near-invisibility (prompt+images vastly outnumber the target in
    # token count), making the training loss hard to read and likely
    # slowing learning on the part that actually matters. The prompt/reply
    # boundary is found via the literal "<|im_start|>assistant\n" marker in
    # the same string already tokenized above -- not a fresh
    # apply_chat_template(..., add_generation_prompt=True) call, which for
    # Qwen3.5 in non-thinking mode would also insert a
    # "<think>\n\n</think>\n\n" block that isn't present in the real
    # sequence, throwing off the token count.
    assistant_marker = "<|im_start|>assistant\n"
    marker_pos = text.index(assistant_marker)
    prompt_only_text = text[: marker_pos + len(assistant_marker)]
    prompt_batch = processor(
        text=[prompt_only_text],
        images=images_input,
        padding=False,
        return_tensors="pt",
    )
    prompt_len = prompt_batch["input_ids"].shape[1]

    labels = result["input_ids"].clone()
    labels[:prompt_len] = -100
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
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
            HubBestAdapterCallback(repo_id_target),
        ],
    )

    # 7. Train (or resume) the model
    if resume_checkpoint:
        print(f"Resuming training for {task} ({strategy_name} reordering) from {resume_checkpoint}...")
    else:
        print(f"Training model for {task} with {strategy_name} reordering...")
    if not train_or_load_finished(trainer_instance, repo_id_target, resume_checkpoint):
        print(f"Nothing new to push for {repo_id_target}.")
        return trainer_instance.model

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
