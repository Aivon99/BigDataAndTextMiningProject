"""Runs a model on the test set of each task and computes the metrics."""

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


def stop_token_kwargs(processor) -> dict:
    """generate() arguments that stop at the end of the answer.

    The checkpoint has no generation_config.json, so generate() would only stop
    at <|endoftext|>, while the chat template ends every answer with <|im_end|>.
    """
    tok = processor.tokenizer
    stop_ids = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    return {"eos_token_id": stop_ids, "pad_token_id": tok.pad_token_id}


def _sample_images(sample, task: str) -> list:
    if task == "task3":
        return [sample["image"], sample["image_t1"]]
    return [sample["image"]]


def build_chat(sample, task: str, few_shot_examples=None):
    """Chat messages and their images for one sample. Each few-shot example is
    added before it as a solved question/answer pair.
    """
    messages, images = [], []
    for shot in list(few_shot_examples or []) + [sample]:
        shot_images = _sample_images(shot, task)
        messages.append({
            "role": "user",
            "content": [{"type": "image", "image": im} for im in shot_images]
                       + [{"type": "text", "text": shot["prompt"]}],
        })
        images.extend(shot_images)
        if shot is not sample:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": shot["target"]}]})
    return messages, images


def evaluate_chessboard_model_task_1(model, processor, dataset_split, model_name: str, few_shot_examples=None) -> pd.DataFrame:
    """Evaluates a model on Task 1 (image to FEN). Returns the per-sample
    results and a one-row summary.
    """
    model.eval()
    results_list = []

    for test_sample in tqdm(dataset_split, desc=f"Evaluating {model_name}"):
        ground_truth_fen = test_sample["target"]
        sample_id = test_sample["sample_id"]

        chat_messages, chat_images = build_chat(test_sample, "task1", few_shot_examples)

        formatted_text = processor.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)

        model_inputs = processor(
            text=[formatted_text],
            images=chat_images,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_token_ids = model.generate(**model_inputs, max_new_tokens=100, **stop_token_kwargs(processor))

        trimmed_output_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_token_ids)
        ]
        raw_output = processor.batch_decode(
            trimmed_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        predicted_fen_string = extract_fen(raw_output)

        exact_match = calculate_fen_exact_match(predicted_fen_string, ground_truth_fen)
        levenshtein_res = calculate_levenshtein_metrics(predicted_fen_string, ground_truth_fen)
        square_accuracy = calculate_square_by_square_accuracy(predicted_fen_string, ground_truth_fen)

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

    results_df = analyze_fen_predictions(pd.DataFrame(results_list))

    csv_filename = f"task1_{model_name.lower().replace(' ', '_')}_results.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"\nEvaluation completed for {model_name}! Results saved to {csv_filename}.")

    mean_fen_em = results_df["fen_exact_match"].mean()
    mean_cer = results_df["character_error_rate"].mean()
    mean_square_acc = results_df["square_by_square_accuracy"].mean()
    mean_levenshtein = results_df["levenshtein_distance"].mean()

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

def evaluate_chessboard_model_task_2(model, processor, dataset_split, model_name: str, few_shot_examples=None) -> pd.DataFrame:
    """Evaluates a model on Task 2 (last move from one image). Returns the
    per-sample results and a one-row summary.
    """
    model.eval()
    results_list = []

    for test_sample in tqdm(dataset_split, desc=f"Evaluating {model_name} (Task 2)"):
        ground_truth_move = test_sample["target"]
        sample_id = test_sample["sample_id"]

        chat_messages, chat_images = build_chat(test_sample, "task2", few_shot_examples)

        formatted_text = processor.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)

        model_inputs = processor(
            text=[formatted_text],
            images=chat_images,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_token_ids = model.generate(**model_inputs, max_new_tokens=16, **stop_token_kwargs(processor))

        trimmed_output_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_token_ids)
        ]
        raw_output = processor.batch_decode(
            trimmed_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        predicted_move = raw_output

        exact_match = calculate_san_exact_match(predicted_move, ground_truth_move)

        results_list.append({
            "sample_id": sample_id,
            "ground_truth": ground_truth_move,
            "predicted": predicted_move,
            "raw_output": raw_output,
            "exact_match": exact_match
        })

    results_df = analyze_san_predictions(pd.DataFrame(results_list), dataset_split["fen"])

    csv_filename = f"task2_{model_name.lower().replace(' ', '_')}_results.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"\nEvaluation completed for {model_name}! Results saved to {csv_filename}.")

    mean_em = results_df["exact_match"].mean()

    model_summary_df = pd.DataFrame([
        {
            "model_name": model_name,
            "exact_match": mean_em,
            **{col: results_df[col].mean() for col in SAN_DIAGNOSTIC_COLUMNS},
        }
    ])

    return results_df, model_summary_df


def evaluate_chessboard_model_task_3(model, processor, dataset_split, model_name: str, few_shot_examples=None) -> pd.DataFrame:
    """Evaluates a model on Task 3 (move between two images). Returns the
    per-sample results and a one-row summary.
    """
    model.eval()
    results_list = []

    for test_sample in tqdm(dataset_split, desc=f"Evaluating {model_name} on Task 3"):
        ground_truth_move = test_sample["target"]
        sample_id = test_sample["sample_id"]

        chat_messages, chat_images = build_chat(test_sample, "task3", few_shot_examples)

        formatted_text = processor.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)

        model_inputs = processor(
            text=[formatted_text],
            images=chat_images,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_token_ids = model.generate(**model_inputs, max_new_tokens=16, **stop_token_kwargs(processor))

        trimmed_output_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_token_ids)
        ]
        raw_output = processor.batch_decode(
            trimmed_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        predicted_move_string = raw_output

        exact_match = calculate_san_exact_match(predicted_move_string, ground_truth_move)

        results_list.append({
            "sample_id": sample_id,
            "ground_truth": ground_truth_move,
            "predicted": predicted_move_string,
            "raw_output": raw_output,
            "exact_match": exact_match
        })

    results_df = analyze_san_predictions(pd.DataFrame(results_list), dataset_split["fen"])

    csv_filename = f"task3_{model_name.lower().replace(' ', '_')}_results.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"\nEvaluation completed for {model_name} on Task 3! Results saved to {csv_filename}.")

    mean_em = results_df["exact_match"].mean()

    model_summary_df = pd.DataFrame([
        {
            "model_name": model_name,
            "exact_match": mean_em,
            **{col: results_df[col].mean() for col in SAN_DIAGNOSTIC_COLUMNS},
        }
    ])

    return results_df, model_summary_df
