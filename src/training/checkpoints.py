"""Helpers to save, resume and check the training runs on the Hugging Face Hub."""

import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
try:
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
except ImportError:
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from transformers import TrainerCallback
from peft import set_peft_model_state_dict
from peft.utils import load_peft_weights


TRAINING_RECIPE = "answer-only-loss-v1"
_RECIPE_FILE = "training_recipe.json"


def read_training_recipe(repo_id: str) -> str | None:
    """Returns the training recipe saved in a Hub repo, or None if there is none."""
    try:
        path = hf_hub_download(repo_id, _RECIPE_FILE, repo_type="model")
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    return json.loads(Path(path).read_text()).get("recipe")


def stamp_training_recipe(repo_id: str, recipe: str = TRAINING_RECIPE) -> None:
    """Saves the training recipe in the Hub repo."""
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=json.dumps({"recipe": recipe}).encode("utf-8"),
        path_in_repo=_RECIPE_FILE,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Training recipe: {recipe}",
    )


_RUN_STATE_FOLDERS = ("last-checkpoint", "best-checkpoint")


def find_resumable_checkpoint(repo_id: str) -> str | None:
    """Downloads the last checkpoint of a run from the Hub, if there is one, so
    training can resume.

    If the checkpoint was made with a different recipe it is deleted and
    None is returned.
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
        (candidate / "scaler.pt").unlink(missing_ok=True)
        print(f"Found a resumable checkpoint on the Hub for '{repo_id}': {candidate}")
        return str(candidate)
    return None


def training_is_complete(checkpoint_dir: str) -> bool:
    """True if the run in the checkpoint already finished."""
    state = json.loads((Path(checkpoint_dir) / "trainer_state.json").read_text())
    control = state.get("stateful_callbacks", {}).get("TrainerControl", {}).get("args", {})
    return bool(control.get("should_training_stop")) or state["global_step"] >= state["max_steps"]


def train_or_load_finished(trainer, repo_id: str, resume_checkpoint: str | None) -> bool:
    """Trains the model, or loads the final adapter if the run already finished.
    Returns True if training ran.
    """
    if resume_checkpoint and training_is_complete(resume_checkpoint):
        print(f"Training for '{repo_id}' already finished on the Hub -- skipping training and loading its final adapter.")
        set_peft_model_state_dict(trainer.model, load_peft_weights(repo_id))
        return False
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    return True


class HubBestAdapterCallback(TrainerCallback):
    """Keeps the adapter of the best epoch on the Hub, so the best model can be
    restored after a resume in a new session.
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
            return
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
