import Levenshtein
import chess

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

def evaluate_chessboard_model(model, processor, dataset_split, model_name: str) -> pd.DataFrame:
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
