import pandas as pd
import torch
from tqdm import tqdm

from eval.diagnostics import (
    FEN_DIAGNOSTIC_COLUMNS,
    SAN_DIAGNOSTIC_COLUMNS,
    analyze_fen_predictions,
    analyze_san_predictions,
)
from eval.metrics import (
    calculate_fen_exact_match,
    calculate_levenshtein_metrics,
    calculate_san_exact_match,
    calculate_square_by_square_accuracy,
    extract_fen,
)


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
