"""Evaluates the published models again and compares them with our saved
predictions. Used by notebooks/Reproduce_results.ipynb.

Each task must run with the torch version that produced its saved results
(TORCH_VERSIONS): with another version the bfloat16 kernels change the logits
slightly, and greedy decoding can then pick a different token. A task can also be
run from the command line:

    python -m eval.reproduce --task task1 --level main --output task1.json
"""

import argparse
import gc
import io
import json
import os

import pandas as pd
import torch
import transformers
import peft
from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from eval.evaluation import (
    evaluate_chessboard_model_task_1,
    evaluate_chessboard_model_task_2,
    evaluate_chessboard_model_task_3,
)
from eval.diagnostics import analyze_fen_predictions
from eval.results import load_results_table
from training.finetune import experiment_model_name, experiment_repo_id
from training.reordering import resolve_square_order
from training.resolution import configure_image_resolution
from training.token_reordering import install_token_reordering, verify_token_reordering


MODEL_ID = "Qwen/Qwen3.5-0.8B"
HF_ORG = "bdatm-project"
IMAGE_SIZE = 256            # 256px -> one visual token per chess square
ABLATION_IMAGE_SIZE = 512   # resolution ablation: 2x2 visual tokens per square
TAG = f"tok{IMAGE_SIZE}"
STRATEGIES = ["zigzag", "spiral_center_out", "file_wise", "hilbert"]
N_SHOT = 3                  # solved train examples in the prompt of the 3-shot vanilla model
LEVELS = ("main", "tables", "full")

# torch version that produced the saved results of each task: the one preinstalled on
# Colab during our runs (Task 1 was evaluated again with it)
TORCH_VERSIONS = {"task1": "2.11.0", "task2": "2.11.0", "task3": "2.11.0"}

# Main score of each task, as in the README: Task 1 compares the board part of the FEN,
# Tasks 2 and 3 the SAN move taken from the first line of the answer
MAIN_METRIC = {"task1": "board_exact_match", "task2": "em_extracted", "task3": "em_extracted"}
EVAL_FUNCS = {
    "task1": evaluate_chessboard_model_task_1,
    "task2": evaluate_chessboard_model_task_2,
    "task3": evaluate_chessboard_model_task_3,
}


def check_torch_version(task):
    """Warns if the installed torch is not the one that produced the task's saved results."""
    installed = torch.__version__.split("+")[0]
    expected = TORCH_VERSIONS[task]
    if installed != expected:
        print(f"WARNING: our {task} results were produced with torch {expected}, this is torch "
              f"{installed}; more predictions than usual can differ from ours.")


def load_task_dataset(task, num_workers):
    """Loads a task's dataset from the Hub the same way the task notebooks do."""
    name = f"{HF_ORG}/dataset_{task}"
    if task != "task3":
        return load_dataset(name, num_proc=num_workers)
    # load_dataset(name) would download every image one by one and hit the Hub rate limit
    local_dir = snapshot_download(repo_id=name, repo_type="dataset", local_dir=f"./{task}_files",
                                  max_workers=num_workers)
    dataset = load_dataset("imagefolder", data_dir=local_dir, num_proc=num_workers)
    for split in dataset.keys():
        def add_second_frame(example, split=split):
            path = os.path.join(local_dir, split, example["file_name_t1"])
            example["image_t1"] = Image.open(path).convert("RGB")
            return example
        dataset[split] = dataset[split].map(add_second_frame, num_proc=num_workers)
    return dataset


def load_base_model():
    return AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def make_processors():
    """One processor at 256px (the main experiments) and one at 512px (the resolution ablation)."""
    processors = {}
    for size in (IMAGE_SIZE, ABLATION_IMAGE_SIZE):
        processors[size] = AutoProcessor.from_pretrained(MODEL_ID)
        configure_image_resolution(processors[size], size)
    return processors


def check_token_reordering(processor):
    """Checks that the token reordering patch works with the installed transformers."""
    check_dataset = load_dataset(f"{HF_ORG}/dataset_task1", split="test[:1]")
    check_model = load_base_model()
    verify_token_reordering(check_model, processor, check_dataset[0]["image"], resolve_square_order("zigzag"))
    del check_model
    free_gpu()


def learned_order(adapter_repo):
    """Scan order used to evaluate a learned-reordering adapter."""
    with open(hf_hub_download(adapter_repo, "learned_order.json")) as f:
        return json.load(f)["order"]


def experiment_rows(task):
    """The rows of the README tables for a task."""
    raster = experiment_repo_id(task, "raster", TAG, HF_ORG)
    rows = [
        {"name": "Vanilla Qwen3.5-0.8B (Zero-Shot)", "adapter": None, "order": None},
        {"name": f"Vanilla Qwen3.5-0.8B ({N_SHOT}-Shot)", "adapter": None, "order": None, "few_shot": True},
        {"name": "Qwen + LoRA (Raster - SFT)", "adapter": raster, "order": None},
    ]
    # Training-free reordering: the raster model reads the tokens in another order
    rows += [{"name": f"Qwen + LoRA Raster, read {s} (TF)", "adapter": raster, "order": s} for s in STRATEGIES]
    # Fine-tuned reordering: one adapter per order, tested with that order
    rows += [{"name": f"Qwen + LoRA ({s.capitalize()} - SFT)", "adapter": experiment_repo_id(task, s, TAG, HF_ORG),
              "order": s} for s in STRATEGIES]
    for strategy, label in [("learned", "Learned Reordering"), ("learned-frozen", "Learned Reordering, frozen policy")]:
        repo = experiment_repo_id(task, strategy, TAG, HF_ORG)
        rows.append({"name": f"Qwen + LoRA ({label})", "adapter": repo, "order": learned_order(repo)})
    for row in rows:
        row["name"] = experiment_model_name(row["name"], TAG)
        row["label"] = row["name"]
        row["image_size"] = IMAGE_SIZE
        row["is_main"] = row["adapter"] == raster and row["order"] is None
    # Resolution ablation: the raster LoRA model trained and tested at 512px
    rows.append({
        "name": "Qwen + LoRA Fine-Tuning (from HF)",   # its name in the results table
        "label": "Qwen + LoRA (Raster - SFT) [512px]",
        "adapter": f"{HF_ORG}/qwen-{task}-standard-lora",
        "order": None,
        "image_size": ABLATION_IMAGE_SIZE,
        "is_main": False,
    })
    return rows


def selected_rows(task, level, task1_subset):
    """The rows to run for this level, each with its number of test samples."""
    rows = [r for r in experiment_rows(task) if level != "main" or r["is_main"]]
    for row in rows:
        subset = level == "tables" and task == "task1" and not row["is_main"]
        row["n_samples"] = task1_subset if subset else None   # None = all test samples
    return rows


def evaluate_row(task, row, dataset, processors):
    """Loads the base model and the adapter, sets the scan order and runs the task's evaluation."""
    model = load_base_model()
    if row["adapter"] is not None:
        model = PeftModel.from_pretrained(model, row["adapter"])
    test = dataset["test"]
    if row["n_samples"] is not None:
        test = test.select(range(row["n_samples"]))
    few_shot = [dataset["train"][i] for i in range(N_SHOT)] if row.get("few_shot") else None
    reorderer = install_token_reordering(model)
    with reorderer.order(resolve_square_order(row["order"])):
        results_df, _ = EVAL_FUNCS[task](
            model=model, processor=processors[row["image_size"]], dataset_split=test, model_name=row["name"],
            few_shot_examples=few_shot,
        )
    del model, reorderer
    free_gpu()
    return results_df


def compare_with_published(task, row, results_df, published_table):
    """Compares the new predictions with our saved ones for the same model, sample by sample."""
    repo = f"{HF_ORG}/evaluation-results-{task}"
    slug = row["name"].lower().replace(" ", "_")
    published = pd.read_csv(hf_hub_download(repo, f"per_sample/{slug}.csv", repo_type="dataset"))
    published = published.set_index("sample_id").loc[results_df["sample_id"]].reset_index()
    if task == "task1" and MAIN_METRIC[task] not in published.columns:
        published = analyze_fen_predictions(published)   # older saved results lack the board columns
    # write the new results to CSV and back, so both sides are parsed the same way
    new = pd.read_csv(io.StringIO(results_df.to_csv(index=False)))
    same = new["predicted"].fillna("").astype(str).values == published["predicted"].fillna("").astype(str).values
    metric = MAIN_METRIC[task]
    readme_value = published_table.loc[published_table["model_name"] == row["name"], metric].iloc[0]
    return {
        "task": task,
        "model": row["label"],
        "test samples": len(new),
        "ours, same samples (%)": round(100 * float(published[metric].mean()), 2),
        "reproduced (%)": round(100 * float(new[metric].mean()), 2),
        "identical predictions": f"{same.sum()}/{len(same)}",
        "ours, 400 samples (README, %)": round(100 * float(readme_value), 2),
    }


def reproduce_task(task, level, processors, num_workers, task1_subset):
    """Evaluates every selected row of a task and returns one comparison dict per row."""
    assert level in LEVELS, f"Unknown level: {level}"
    check_torch_version(task)
    dataset = load_task_dataset(task, num_workers)
    published_table = load_results_table(f"{HF_ORG}/evaluation-results-{task}")
    comparison = []
    for row in selected_rows(task, level, task1_subset):
        print(f"\n=== {row['label']} ({row['n_samples'] or len(dataset['test'])} test samples) ===", flush=True)
        results_df = evaluate_row(task, row, dataset, processors)
        comparison.append(compare_with_published(task, row, results_df, published_table))
        print(comparison[-1], flush=True)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=list(EVAL_FUNCS))
    parser.add_argument("--level", required=True, choices=LEVELS)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--task1-subset", type=int, default=100)
    parser.add_argument("--output", required=True, help="JSON file where the comparison rows are written")
    args = parser.parse_args()

    print(f"torch {torch.__version__}, transformers {transformers.__version__}, peft {peft.__version__}")
    assert torch.cuda.is_available(), "This evaluation needs a GPU (an L4 or A100)."
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    processors = make_processors()
    check_token_reordering(processors[IMAGE_SIZE])
    comparison = reproduce_task(args.task, args.level, processors, args.num_workers, args.task1_subset)
    with open(args.output, "w") as f:
        json.dump(comparison, f, indent=2)


if __name__ == "__main__":
    main()
