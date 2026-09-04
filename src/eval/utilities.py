import Levenshtein
import chess
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, Trainer, TrainingArguments
from peft import get_peft_model
from functools import partial


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

        # 5. Generate the prediction
        with torch.no_grad():
            output_token_ids = model.generate(**model_inputs, max_new_tokens=128)

        # 6. Trim prompt tokens from the generated output
        trimmed_output_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_token_ids)
        ]
        predicted_fen_string = processor.batch_decode(
            trimmed_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        # 7. Compute metrics for the current sample using predefined functions
        exact_match = calculate_fen_exact_match(predicted_fen_string, ground_truth_fen)
        levenshtein_res = calculate_levenshtein_metrics(predicted_fen_string, ground_truth_fen)
        square_accuracy = calculate_square_by_square_accuracy(predicted_fen_string, ground_truth_fen)

        # 8. Append row data including predictions and metrics
        results_list.append({
            "sample_id": sample_id,
            "ground_truth": ground_truth_fen,
            "predicted": predicted_fen_string,
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

def reorder_chessboard_image(image, strategy="raster", grid_size=8):
    """
    Slices a chessboard PIL Image into an 8x8 grid of tiles and
    rearranges them according to the specified reordering strategy.
    """
    # Ensure image is square and resize to a multiple of grid_size (e.g., 512x512)
    img_size = 512
    image = image.resize((img_size, img_size))
    tile_size = img_size // grid_size

    # 1. Split image into 64 individual square tiles
    tiles = []
    for r in range(grid_size):
        for c in range(grid_size):
            box = (c * tile_size, r * tile_size, (c + 1) * tile_size, (r + 1) * tile_size)
            tile = image.crop(box)
            tiles.append(tile)

    # 2. Get reordering indices for the chosen strategy
    reorder_indices = get_patch_reordering_indices(strategy=strategy, grid_size=grid_size)

    # 3. Rearrange tiles based on the indices
    reordered_tiles = [tiles[i] for i in reorder_indices]

    # 4. Stitch tiles back together into a new reordered image
    new_image = Image.new("RGB", (img_size, img_size))
    for idx, tile in enumerate(reordered_tiles):
        r = idx // grid_size
        c = idx % grid_size
        new_image.paste(tile, (c * tile_size, r * tile_size))

    return new_image




def finetune_and_push_chessboard_model(
    strategy_name,
    dataset,
    processor,
    peft_config,
    task,
    base_model_id="Qwen/Qwen3.5-0.8B",
    hf_org_prefix="bdatm-project",
    repo_root=None,
):
  print(f"\n==============================================")
  print(
      f"Starting pipeline for TASK: {task.upper()} | STRATEGY:"
      f" {strategy_name.upper()}"
  )
  print(f"==============================================")

  # 1. Apply image-level reordering based on the task type
  print(
      f"Applying {strategy_name} reordering to datasets for {task}..."
  )

  def reorder_split(split_ds):
    def transform(sample):
      if task in ["task1", "task2"]:
        # Single-image tasks
        img = sample["image"]
        reordered_img = reorder_chessboard_image(
            img, strategy=strategy_name, grid_size=8
        )
        return {"image": reordered_img}

      elif task == "task3":
        # Dual-image task (reorder both frame t and frame t+1)
        img_t = sample["image"]
        reordered_img_t = reorder_chessboard_image(
            img_t, strategy=strategy_name, grid_size=8
        )

        # Handle second frame
        t1_path = sample.get("file_name_t1")
        if isinstance(t1_path, str) and t1_path.strip() != "":
          img_t1_path = (
              Path(repo_root) / t1_path if repo_root else Path(t1_path)
          )
          img_t1 = Image.open(img_t1_path).convert("RGB")
        else:
          img_t1 = sample.get("image_t1") or t1_path

        reordered_img_t1 = reorder_chessboard_image(
            img_t1, strategy=strategy_name, grid_size=8
        )

        # Return both reordered frames
        return {"image": reordered_img_t, "image_t1": reordered_img_t1}
      else:
        raise ValueError(f"Unknown task: {task}")

    return split_ds.map(transform)

  reordered_train = reorder_split(dataset["train"])
  reordered_val = reorder_split(dataset["validation"])

  # 2. Tokenize and preprocess the reordered datasets using the unified preprocessing function
  print("Preprocessing datasets...")
  tokenized_train = reordered_train.map(
      partial(
          preprocess_function, processor=processor, repo_root=repo_root
      ),
      remove_columns=reordered_train.column_names,
  )
  tokenized_val = reordered_val.map(
      partial(
          preprocess_function, processor=processor, repo_root=repo_root
      ),
      remove_columns=reordered_val.column_names,
  )

  # 3. Load a fresh base model instance and apply PEFT/LoRA
  print("Loading base model and applying LoRA...")
  model_instance = AutoModelForImageTextToText.from_pretrained(
      base_model_id,
      torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
      device_map="auto",
  )
  lora_model_instance = get_peft_model(model_instance, peft_config)

  # 4. Configure Training Arguments for this specific run
  training_args_instance = TrainingArguments(
      output_dir=f"./temp_{task}_{strategy_name}_output",
      per_device_train_batch_size=1,
      per_device_eval_batch_size=1,
      gradient_accumulation_steps=8,
      learning_rate=2e-4,
      logging_steps=10,
      num_train_epochs=2,
      save_strategy="epoch",
      eval_strategy="epoch",
      fp16=True,
      remove_unused_columns=False,
      report_to="none",
  )

  # 5. Initialize Trainer
  trainer_instance = Trainer(
      model=lora_model_instance,
      args=training_args_instance,
      train_dataset=tokenized_train,
      eval_dataset=tokenized_val,
  )

  # 6. Train the model
  print(f"Training model for {task} with {strategy_name} reordering...")
  trainer_instance.train()

  # 7. Push final weights and processor directly to Hugging Face Hub
  repo_id_target = f"{hf_org_prefix}/qwen-{task}-{strategy_name}-lora"
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
