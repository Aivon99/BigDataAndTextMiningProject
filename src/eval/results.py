"""Saves and loads the evaluation results on the Hugging Face Hub, so a model
is not evaluated twice.
"""

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
try:
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
except ImportError:
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError

from eval.evaluation import (
    evaluate_chessboard_model_task_1,
    evaluate_chessboard_model_task_2,
    evaluate_chessboard_model_task_3,
)


_EVAL_FUNCS = {
    "task1": evaluate_chessboard_model_task_1,
    "task2": evaluate_chessboard_model_task_2,
    "task3": evaluate_chessboard_model_task_3,
}


def _eval_results_repo_id(task, hf_org_prefix):
    return f"{hf_org_prefix}/evaluation-results-{task}"


_RESULTS_PARQUET = "data/train-00000-of-00001.parquet"

_RESULTS_CARD = """---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
Model-comparison table for this task: one row per evaluated model, written by
`push_results_table` in `src/eval/results.py`. Per-sample predictions are in
`per_sample/`.
"""


def load_results_table(repo_id):
    """Reads the results table from the Hub. Returns None if it doesn't exist yet."""
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


def push_results_table(repo_id, table, commit_message="Update results table"):
    """Replaces the results table on the Hub with the given one."""
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


def load_or_init_results_table(task, hf_org_prefix="bdatm-project"):
    """Loads the results table of a task, or returns an empty one."""
    repo_id = _eval_results_repo_id(task, hf_org_prefix)
    table = load_results_table(repo_id)
    if table is None:
        print(f"No existing comparison table found for {task} on the Hub (or the repo doesn't exist yet), starting fresh.")
        return pd.DataFrame()
    print(f"Loaded existing comparison table for {task} from '{repo_id}' ({len(table)} model(s) already evaluated).")
    return table


def append_or_replace(all_results, summary_df):
    """Adds the summary row of a model, replacing the old row with the same
    model_name.
    """
    if "model_name" in all_results.columns and len(summary_df) > 0:
        model_name = summary_df["model_name"].iloc[0]
        all_results = all_results[all_results["model_name"] != model_name]
    return pd.concat([all_results, summary_df], ignore_index=True)


_ADAPTER_WEIGHTS = "adapter_model.safetensors"


def adapter_fingerprint(repo_id, revision=None):
    """Returns the sha256 of the LoRA weights file on the Hub, without
    downloading it.
    """
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    for sibling in info.siblings:
        if sibling.rfilename == _ADAPTER_WEIGHTS and sibling.lfs is not None:
            lfs = sibling.lfs
            return lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256
    raise FileNotFoundError(f"No {_ADAPTER_WEIGHTS} in '{repo_id}' at revision {revision or 'main'}")


def _cached_adapter_matches(cached, adapter_repo_id, current_fingerprint):
    if not isinstance(cached, str) or not cached:
        return False
    if cached == current_fingerprint:
        return True
    try:
        return adapter_fingerprint(adapter_repo_id, revision=cached) == current_fingerprint
    except (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError, FileNotFoundError):
        return False


def evaluate_with_cache(
    task,
    model,
    processor,
    dataset_split,
    model_name,
    hf_org_prefix="bdatm-project",
    force=False,
    adapter_repo_id=None,
    **eval_kwargs,
):
    """Same as evaluate_chessboard_model_task_N, but first looks for a cached
    result on the Hub.

    If adapter_repo_id is given, the cached result is used only if the weights
    have not changed. Use force=True to evaluate again anyway. eval_kwargs go
    to the evaluation function (e.g. few_shot_examples).
    """
    if task not in _EVAL_FUNCS:
        raise ValueError(f"evaluate_with_cache: unknown task '{task}'")

    repo_id = _eval_results_repo_id(task, hf_org_prefix)
    slug = model_name.lower().replace(" ", "_")
    adapter_sha = adapter_fingerprint(adapter_repo_id) if adapter_repo_id else None

    if not force:
        table = load_results_table(repo_id)
        if table is not None and "model_name" in table.columns:
            match = table[table["model_name"] == model_name]
            if adapter_sha is not None and len(match) > 0:
                cached_sha = match["adapter_sha"].iloc[0] if "adapter_sha" in match.columns else None
                if not _cached_adapter_matches(cached_sha, adapter_repo_id, adapter_sha):
                    print(
                        f"[{task}] Cached evaluation for '{model_name}' was for different weights than "
                        f"those now in '{adapter_repo_id}', re-evaluating."
                    )
                    match = match.iloc[0:0]
            if len(match) > 0:
                print(f"[{task}] Using cached evaluation for '{model_name}', skipping inference.")
                summary_df = match.reset_index(drop=True)
                try:
                    sample_path = hf_hub_download(repo_id, f"per_sample/{slug}.csv", repo_type="dataset")
                    results_df = pd.read_csv(sample_path)
                except Exception:
                    results_df = None
                return results_df, summary_df

    results_df, summary_df = _EVAL_FUNCS[task](
        model=model, processor=processor, dataset_split=dataset_split, model_name=model_name, **eval_kwargs
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

    existing = load_results_table(repo_id)
    updated = append_or_replace(existing if existing is not None else pd.DataFrame(), summary_df)
    push_results_table(repo_id, updated, commit_message=f"Cache evaluation for '{model_name}' ({task})")
    print(f"[{task}] Cached evaluation for '{model_name}', pushed to '{repo_id}'.")

    return results_df, summary_df


def evaluate_with_order(
    task,
    model,
    processor,
    dataset_split,
    model_name,
    order=None,
    hf_org_prefix="bdatm-project",
    adapter_repo_id=None,
    force=False,
    **eval_kwargs,
):
    """evaluate_with_cache with the visual tokens read in the given order
    (strategy name, square permutation, or None for raster). The model's
    previous order is restored afterwards.
    """
    from training.reordering import resolve_square_order
    from training.token_reordering import install_token_reordering

    reorderer = install_token_reordering(model)
    with reorderer.order(resolve_square_order(order)):
        return evaluate_with_cache(
            task=task, model=model, processor=processor, dataset_split=dataset_split,
            model_name=model_name, hf_org_prefix=hf_org_prefix, force=force,
            adapter_repo_id=adapter_repo_id, **eval_kwargs,
        )
