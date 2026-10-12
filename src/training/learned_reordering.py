"""Learned patch reordering: a small policy learns which patch order works
best while LoRA trains the model. Based on the REOrder paper.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from huggingface_hub import HfApi, snapshot_download
from peft import get_peft_model, set_peft_model_state_dict
from peft.utils import load_peft_weights
from transformers import get_linear_schedule_with_warmup

from training.checkpoints import TRAINING_RECIPE
from training.preprocessing import preprocess_function
from training.reordering import apply_patch_permutation

_CHECKPOINT_SUBDIR = "last-learned-checkpoint"

DEFAULT_INIT_SPACING = 2.0


def _learned_recipe(init_spacing: float) -> str:
    return f"{TRAINING_RECIPE}+pl-init-spacing-{init_spacing:g}"


def _sample_gumbel(shape, device, eps=1e-8):
    u = torch.empty(shape, device=device).uniform_(0, 1)
    return -torch.log(-torch.log(u + eps) + eps)


class PlackettLucePatchPolicy(nn.Module):
    """Learns a ranking of the board patches and samples patch orders from it.
    init_spacing sets how close the samples stay to raster order at the start.
    """

    def __init__(self, grid_size: int = 8, baseline_momentum: float = 0.9, init_spacing: float = DEFAULT_INIT_SPACING):
        super().__init__()
        num_patches = grid_size * grid_size
        self.grid_size = grid_size
        self.num_patches = num_patches
        self.logits = nn.Parameter(-init_spacing * torch.arange(num_patches, dtype=torch.float32))
        self.register_buffer("running_baseline", torch.tensor(0.0))
        self.baseline_momentum = baseline_momentum
        self.temperature = 1.0

    def sample(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a batch of permutations and their log probabilities."""
        logits = self.logits.unsqueeze(0)
        g = _sample_gumbel((batch_size, self.num_patches), device=logits.device)
        z = logits + self.temperature * g

        sorted_indices = torch.argsort(z, dim=-1, descending=True)
        logits_sorted = torch.gather(logits.expand(batch_size, -1), dim=1, index=sorted_indices)

        log_cumsums_rev = torch.logcumsumexp(torch.flip(logits_sorted, dims=[-1]), dim=-1)
        log_denominators = torch.flip(log_cumsums_rev, dims=[-1])
        log_prob = (logits_sorted - log_denominators).sum(dim=-1)

        return sorted_indices, log_prob

    def greedy_permutation(self) -> list:
        """Best patch order according to the policy, used at evaluation time."""
        return torch.argsort(self.logits, descending=True).tolist()

    def update_baseline(self, reward: torch.Tensor) -> None:
        with torch.no_grad():
            self.running_baseline.copy_(
                self.running_baseline * self.baseline_momentum
                + (1.0 - self.baseline_momentum) * reward.mean()
            )

    def reinforce_loss(self, reward: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
        advantage = reward.detach() - self.running_baseline
        return -(advantage * log_prob).mean()


def _save_learned_reordering_checkpoint(
    local_dir: Path,
    repo_id: str,
    lora_model,
    policy: "PlackettLucePatchPolicy",
    model_optimizer,
    policy_optimizer,
    scheduler,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    recipe: str,
) -> None:
    ckpt_dir = local_dir / _CHECKPOINT_SUBDIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    lora_model.save_pretrained(str(ckpt_dir))
    torch.save(policy.state_dict(), ckpt_dir / "policy.pt")
    torch.save(model_optimizer.state_dict(), ckpt_dir / "model_optimizer.pt")
    torch.save(policy_optimizer.state_dict(), ckpt_dir / "policy_optimizer.pt")
    torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
    metadata = {"epoch": epoch, "step_in_epoch": step_in_epoch, "global_step": global_step, "recipe": recipe}
    (ckpt_dir / "metadata.json").write_text(json.dumps(metadata))

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(ckpt_dir),
        path_in_repo=_CHECKPOINT_SUBDIR,
        commit_message=f"Learned-reordering checkpoint: epoch {epoch}, step {step_in_epoch} (global step {global_step})",
    )


def find_resumable_learned_reordering_checkpoint(repo_id: str, recipe: Optional[str] = None) -> Optional[Tuple[Path, dict]]:
    """Like find_resumable_checkpoint, but for the checkpoints of the learned
    reordering loop.
    """
    try:
        api = HfApi()
        if not api.repo_exists(repo_id=repo_id, repo_type="model"):
            return None

        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=f"{_CHECKPOINT_SUBDIR}/*",
        )
        candidate = Path(local_dir) / _CHECKPOINT_SUBDIR
        metadata_path = candidate / "metadata.json"
        if candidate.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if recipe is not None and metadata.get("recipe") != recipe:
                print(
                    f"Ignoring the learned-reordering checkpoint in '{repo_id}': recipe "
                    f"{metadata.get('recipe')!r} != current {recipe!r}. Starting fresh."
                )
                return None
            print(f"Found a resumable learned-reordering checkpoint on the Hub for '{repo_id}': {candidate}")
            return candidate, metadata
    except Exception as e:
        print(f"No resumable learned-reordering checkpoint found for '{repo_id}' ({e}); starting fresh.")

    return None


def train_with_learned_reordering(
    dataset,
    processor,
    model,
    peft_config,
    task: str,
    hf_org_prefix: str = "bdatm-project",
    grid_size: int = 8,
    img_size: int = 512,
    num_train_epochs: int = 10,
    learning_rate: float = 2e-4,
    policy_learning_rate: float = 1e-2,
    policy_weight: float = 1.0,
    log_every: int = 10,
    output_dir: str = "./learned_reordering_output",
    checkpoint_every_steps: int = 200,
    init_spacing: float = DEFAULT_INIT_SPACING,
):
    """Trains LoRA and the reordering policy together with REINFORCE, using the
    loss as reward.

    A checkpoint is pushed to the Hub during training, so an interrupted run
    resumes where it stopped. Returns (lora_model, policy, repo_id_target).
    """
    if task not in ("task1", "task2", "task3"):
        raise ValueError(f"train_with_learned_reordering: unknown task '{task}'")

    repo_id_target = f"{hf_org_prefix}/qwen-{task}-learned-reordering-lora"
    recipe = _learned_recipe(init_spacing)
    resume = find_resumable_learned_reordering_checkpoint(repo_id_target, recipe=recipe)

    device = model.device
    lora_model = get_peft_model(model, peft_config)
    lora_model.train()

    policy = PlackettLucePatchPolicy(grid_size=grid_size, init_spacing=init_spacing).to(device)

    model_optimizer = torch.optim.AdamW(lora_model.parameters(), lr=learning_rate)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_learning_rate)

    train_split = dataset["train"]
    num_steps = len(train_split) * num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        model_optimizer, num_warmup_steps=0, num_training_steps=num_steps
    )

    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    if resume is not None:
        ckpt_dir, metadata = resume
        set_peft_model_state_dict(lora_model, load_peft_weights(str(ckpt_dir)))
        policy.load_state_dict(torch.load(ckpt_dir / "policy.pt", map_location=device))
        model_optimizer.load_state_dict(torch.load(ckpt_dir / "model_optimizer.pt", map_location=device))
        policy_optimizer.load_state_dict(torch.load(ckpt_dir / "policy_optimizer.pt", map_location=device))
        scheduler.load_state_dict(torch.load(ckpt_dir / "scheduler.pt", map_location=device))
        start_epoch = metadata["epoch"]
        start_step_in_epoch = metadata["step_in_epoch"]
        global_step = metadata["global_step"]
        print(
            f"Resuming learned-reordering training for {task} from epoch {start_epoch}, "
            f"step {start_step_in_epoch} (global step {global_step})..."
        )
    else:
        print(f"No resumable learned-reordering checkpoint found for {task}; starting fresh.")

    local_output_dir = Path(output_dir)

    NO_UNSQUEEZE_KEYS = {"pixel_values", "image_grid_thw"}

    for epoch in range(start_epoch, num_train_epochs):
        epoch_start_offset = start_step_in_epoch if epoch == start_epoch else 0
        rows = (
            train_split.select(range(epoch_start_offset, len(train_split)))
            if epoch_start_offset else train_split
        )
        for row_idx, row in enumerate(rows, start=epoch_start_offset):
            permutation, log_prob = policy.sample(batch_size=1)
            perm_list = permutation[0].tolist()

            reordered_image = apply_patch_permutation(
                row["image"], perm_list, grid_size=grid_size, img_size=img_size
            )
            sample_input = {"task": task, "image": reordered_image, "prompt": row["prompt"], "target": row["target"]}

            if task == "task3":
                sample_input["image_t1"] = apply_patch_permutation(
                    row["image_t1"], perm_list, grid_size=grid_size, img_size=img_size
                )

            sample = preprocess_function(sample_input, processor=processor)

            inputs = {}
            for k, v in sample.items():
                if k == "labels":
                    continue
                t = v if torch.is_tensor(v) else torch.tensor(v)
                if k not in NO_UNSQUEEZE_KEYS:
                    t = t.unsqueeze(0)
                inputs[k] = t.to(device)

            labels = sample["labels"]
            labels = (labels if torch.is_tensor(labels) else torch.tensor(labels)).unsqueeze(0).to(device)

            outputs = lora_model(**inputs, labels=labels)
            lm_loss = outputs.loss

            model_optimizer.zero_grad()
            lm_loss.backward()
            model_optimizer.step()
            scheduler.step()

            reward = -lm_loss.detach().unsqueeze(0)
            policy.update_baseline(reward)
            policy_loss = policy_weight * policy.reinforce_loss(reward, log_prob)

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            global_step += 1
            if global_step % log_every == 0:
                print(
                    f"[{task}] epoch {epoch} step {global_step}: "
                    f"lm_loss={lm_loss.item():.4f} policy_loss={policy_loss.item():.4f} "
                    f"baseline={policy.running_baseline.item():.4f}"
                )

            if global_step % checkpoint_every_steps == 0:
                _save_learned_reordering_checkpoint(
                    local_output_dir, repo_id_target, lora_model, policy,
                    model_optimizer, policy_optimizer, scheduler,
                    epoch=epoch, step_in_epoch=row_idx + 1, global_step=global_step, recipe=recipe,
                )
                print(f"[{task}] checkpoint saved at epoch {epoch}, step {row_idx + 1} (global step {global_step}).")

        _save_learned_reordering_checkpoint(
            local_output_dir, repo_id_target, lora_model, policy,
            model_optimizer, policy_optimizer, scheduler,
            epoch=epoch + 1, step_in_epoch=0, global_step=global_step, recipe=recipe,
        )
        print(f"[{task}] checkpoint saved at end of epoch {epoch} (global step {global_step}).")

    print(f"Pushing learned-reordering model and processor to Hugging Face Hub: {repo_id_target}...")
    lora_model.push_to_hub(
        repo_id_target,
        commit_message=f"Training complete for {task} with learned patch reordering",
    )
    processor.push_to_hub(repo_id_target)

    policy_path = Path(output_dir) / "policy.pt"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), policy_path)
    print(f"Saved learned reordering policy weights to {policy_path}")

    return lora_model, policy, repo_id_target
