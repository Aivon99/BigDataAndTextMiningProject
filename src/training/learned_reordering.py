"""
Learned Reordering (project spec, REOrder methodology section) -- DRAFT.

A lightweight ranking module (`PlackettLucePatchPolicy`) that learns which
8x8 patch-serialization order helps the VLM most, trained jointly with LoRA
fine-tuning. Follows the Plackett-Luce / Gumbel-top-k sampling approach from
the REOrder paper (NeurIPS 2025; see external/REOrder/src/models/layers/plackett_luce.py
for the reference implementation this is adapted from), but reworked for a
*pretrained* VLM (Qwen2.5-VL) instead of REOrder's from-scratch classifiers:

- REOrder can reach directly into its own models' patch-embedding sequence,
  so the permutation can (in principle) sit on a path that's part of the
  model's forward pass. This project has no such hook into Qwen's vision
  tower -- reordering happens *before* the model ever sees the image, as a
  literal pixel-tile rearrangement (`apply_patch_permutation` in
  `src/eval/utilities.py`). That rearrangement is discrete/non-differentiable,
  which is exactly why the policy is trained via REINFORCE (policy gradient)
  rather than backprop, same as REOrder's own approach.
- The reward is the (negative) language-modeling loss on the target FEN/SAN
  tokens, since there's no classification loss/accuracy here -- this is a
  generation task, not classification like REOrder's ImageNet/fMoW setup.

This intentionally does NOT use `transformers.Trainer`: a different
permutation is sampled every step, so images have to be reordered and
preprocessed on the fly per batch rather than precomputed once into a
static dataset (which is how `finetune_and_push_chessboard_model`'s
training-free strategies work). Kept deliberately simple for a first pass:
single sample per forward pass (mirrors this project's existing
`per_device_train_batch_size=1` pattern, since the VLM is already large),
no gradient accumulation, single-GPU, no mixed-precision scaler. Treat this
as a starting point to profile and iterate on, not a finished, tuned
training pipeline.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from peft import get_peft_model
from transformers import get_linear_schedule_with_warmup

from eval.utilities import apply_patch_permutation, preprocess_function


def _sample_gumbel(shape, device, eps=1e-8):
    u = torch.empty(shape, device=device).uniform_(0, 1)
    return -torch.log(-torch.log(u + eps) + eps)


class PlackettLucePatchPolicy(nn.Module):
    """
    Learns a soft ranking over the grid_size x grid_size chessboard patches.

    Holds one learnable logit per patch position (initialized close to
    raster order, so training starts from a sane baseline rather than a
    random permutation). `sample()` draws a permutation via Gumbel-top-k
    sampling from the induced Plackett-Luce distribution, and returns both
    the permutation and its log-probability (needed for the REINFORCE loss).
    """

    def __init__(self, grid_size: int = 8, baseline_momentum: float = 0.9):
        super().__init__()
        num_patches = grid_size * grid_size
        self.grid_size = grid_size
        self.num_patches = num_patches
        self.logits = nn.Parameter(torch.linspace(0, -1, num_patches))
        self.register_buffer("running_baseline", torch.tensor(0.0))
        self.baseline_momentum = baseline_momentum
        self.temperature = 1.0

    def sample(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            permutation: LongTensor [batch_size, num_patches]
            log_prob: FloatTensor [batch_size] -- log pi(permutation)
        """
        logits = self.logits.unsqueeze(0)
        g = _sample_gumbel((batch_size, self.num_patches), device=logits.device)
        z = logits + self.temperature * g

        sorted_indices = torch.argsort(z, dim=-1, descending=True)
        logits_sorted = torch.gather(logits.expand(batch_size, -1), dim=1, index=sorted_indices)

        # Single-pass Plackett-Luce log-probability of the sampled ranking
        log_cumsums_rev = torch.logcumsumexp(torch.flip(logits_sorted, dims=[-1]), dim=-1)
        log_denominators = torch.flip(log_cumsums_rev, dims=[-1])
        log_prob = (logits_sorted - log_denominators).sum(dim=-1)

        return sorted_indices, log_prob

    def greedy_permutation(self) -> list:
        """
        The policy's single best-guess ordering (argsort of the learned
        logits, no sampling noise) -- what to actually use at evaluation
        time, since training samples stochastically but eval needs one fixed
        ordering to reorder the test set with.
        """
        return torch.argsort(self.logits, descending=True).tolist()

    def update_baseline(self, reward: torch.Tensor) -> None:
        """Exponential-moving-average baseline for REINFORCE variance reduction."""
        with torch.no_grad():
            self.running_baseline.copy_(
                self.running_baseline * self.baseline_momentum
                + (1.0 - self.baseline_momentum) * reward.mean()
            )

    def reinforce_loss(self, reward: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
        """-(advantage * log pi(permutation)), averaged over the batch."""
        advantage = reward.detach() - self.running_baseline
        return -(advantage * log_prob).mean()


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
):
    """
    Jointly trains LoRA adapters on `model` and a `PlackettLucePatchPolicy`
    that learns which patch order helps it most, via REINFORCE using the
    per-sample LM loss as the (negative) reward.

    `dataset` is a `datasets.DatasetDict` with the same schema as the other
    task datasets (raw, un-reordered `image`/`image_t1`/`prompt`/`target`
    columns -- reordering happens here, per step, not precomputed). For
    `task="task3"` the same sampled permutation is applied independently to
    both `image` and `image_t1`, mirroring how `finetune_and_push_chessboard_model`
    reorders both frames of a dual-image sample with the same fixed strategy.

    Returns (lora_model, policy, repo_id_target).
    """
    if task not in ("task1", "task2", "task3"):
        raise ValueError(f"train_with_learned_reordering: unknown task '{task}'")

    device = model.device
    lora_model = get_peft_model(model, peft_config)
    lora_model.train()

    policy = PlackettLucePatchPolicy(grid_size=grid_size).to(device)

    model_optimizer = torch.optim.AdamW(lora_model.parameters(), lr=learning_rate)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_learning_rate)

    train_split = dataset["train"]
    num_steps = len(train_split) * num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        model_optimizer, num_warmup_steps=0, num_training_steps=num_steps
    )

    # These fields carry a real leading batch dimension already (as produced by
    # `processor(..., return_tensors="pt")` for a batch of 1) and must NOT be
    # unsqueezed again; only the plain-sequence fields do. Mirrors the
    # distinction `Qwen35VisionDataCollator` makes (cat vs pad_sequence).
    NO_UNSQUEEZE_KEYS = {"pixel_values", "image_grid_thw"}

    global_step = 0
    for epoch in range(num_train_epochs):
        for row in train_split:
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

            # Reward: lower LM loss (better prediction under this ordering) -> higher reward
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

    repo_id_target = f"{hf_org_prefix}/qwen-{task}-learned-reordering-lora"
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
