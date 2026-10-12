import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
try:
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
except ImportError:  # older huggingface_hub
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from transformers import TrainerCallback
from peft import set_peft_model_state_dict
from peft.utils import load_peft_weights


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
