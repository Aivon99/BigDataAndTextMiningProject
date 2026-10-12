import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
try:
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
except ImportError:  # older huggingface_hub
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
`push_results_table` in `src/eval/results.py`. Per-sample predictions are in
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
