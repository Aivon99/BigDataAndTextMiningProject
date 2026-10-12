"""Learned reordering (REOrder): a Plackett-Luce policy over the 64 squares,
trained together with LoRA by REINFORCE, using REOrder's own policy.

- The order is applied to the visual tokens (training/token_reordering.py), so
  every order shows the same board and the reward measures only the order.
- REOrder's recipe: the policy starts after a warm-up in raster order, with
  REOrder's row-major init, and the Gumbel temperature goes from 0.2 to 0.
- Reward = -cross-entropy of the answer. With orders_per_sample=2 two orders
  are scored on the same sample and compared with each other, so the
  difficulty of the puzzle cancels out; with 1, a moving-average baseline.
- Effective batch 8, gradient clipping, shuffled data every epoch, and the
  best epoch by validation loss is kept.

Checkpoints are pushed to the Hub during training and resumed automatically.
"""

import json
import math
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from peft import get_peft_model, set_peft_model_state_dict
from peft.utils import load_peft_weights
from transformers import get_linear_schedule_with_warmup

from training.checkpoints import TRAINING_RECIPE
from training.finetune import experiment_repo_id
from training.preprocessing import preprocess_function
from training.reorder_lib import load_reorder
from training.token_reordering import install_token_reordering

LEARNED_RECIPE = f"{TRAINING_RECIPE}+token-reorder-pl-v1"
_CHECKPOINT_SUBDIR = "last-learned-checkpoint"
_BEST_SUBDIR = "best-learned-checkpoint"
_ORDER_FILE = "learned_order.json"
_NO_UNSQUEEZE = {"pixel_values", "image_grid_thw"}


def make_policy(grid_size: int = 8):
    """REOrder's Plackett-Luce policy, one independent order per sample."""
    return load_reorder().PlackettLucePolicy(
        num_patches=grid_size * grid_size, method="gumbel", granularity="image", logit_init="row-major"
    )


def greedy_order(policy) -> list:
    """The policy's single best ordering (argsort of its logits, no noise) -- used at evaluation."""
    return torch.argsort(policy.logits.detach(), descending=True).tolist()


def _to_model_inputs(sample: dict, device) -> tuple:
    inputs = {}
    for k, v in sample.items():
        if k == "labels":
            continue
        t = v if torch.is_tensor(v) else torch.tensor(v)
        inputs[k] = (t if k in _NO_UNSQUEEZE else t.unsqueeze(0)).to(device)
    labels = sample["labels"]
    labels = (labels if torch.is_tensor(labels) else torch.tensor(labels)).unsqueeze(0).to(device)
    return inputs, labels


def _raw_sample(row: dict, task: str) -> dict:
    sample = {"task": task, "image": row["image"], "prompt": row["prompt"], "target": row["target"]}
    if task == "task3":
        sample["image_t1"] = row["image_t1"]
    return sample


def _upload(local_dir: Path, repo_id: str, path_in_repo: str, message: str) -> None:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(local_dir),
                      path_in_repo=path_in_repo, commit_message=message)


def _save_state(local_dir: Path, lora_model, policy, extra: Optional[dict] = None, optimizers=None) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    lora_model.save_pretrained(str(local_dir))
    torch.save(policy.state_dict(), local_dir / "policy.pt")
    if optimizers:
        for name, obj in optimizers.items():
            torch.save(obj.state_dict(), local_dir / f"{name}.pt")
    (local_dir / "metadata.json").write_text(json.dumps({"recipe": LEARNED_RECIPE, **(extra or {})}))


def _find_state(repo_id: str, subdir: str) -> Optional[tuple]:
    """(local_dir, metadata) of a learned-reordering state on the Hub, if its recipe matches."""
    api = HfApi()
    if not api.repo_exists(repo_id=repo_id, repo_type="model"):
        return None
    if f"{subdir}/metadata.json" not in api.list_repo_files(repo_id, repo_type="model"):
        return None
    local = Path(snapshot_download(repo_id=repo_id, repo_type="model", allow_patterns=f"{subdir}/*")) / subdir
    meta = json.loads((local / "metadata.json").read_text())
    if meta.get("recipe") != LEARNED_RECIPE:
        print(f"Ignoring {repo_id}/{subdir}: recipe {meta.get('recipe')!r} != {LEARNED_RECIPE!r}.")
        return None
    return local, meta


@torch.no_grad()
def _validation_loss(lora_model, reorderer, processor, rows, task, order, device) -> float:
    lora_model.eval()
    losses = []
    with reorderer.order(order):
        for row in rows:
            inputs, labels = _to_model_inputs(preprocess_function(_raw_sample(row, task), processor), device)
            losses.append(lora_model(**inputs, labels=labels).loss.item())
    lora_model.train()
    return sum(losses) / max(len(losses), 1)


def train_with_learned_reordering(
    dataset,
    processor,
    model,
    peft_config,
    task: str,
    hf_org_prefix: str = "bdatm-project",
    tag: Optional[str] = None,
    grid_size: int = 8,
    num_train_epochs: int = 4,
    policy_start_after_epochs: int = 1,
    gumbel_temp_start: float = 0.2,
    gumbel_temp_end: float = 0.0,
    orders_per_sample: int = 2,
    baseline_momentum: float = 0.99,
    learning_rate: float = 2e-4,
    policy_learning_rate: float = 1e-3,
    grad_accum_steps: int = 8,
    max_grad_norm: float = 1.0,
    val_samples: int = 200,
    checkpoint_every_steps: int = 100,
    log_every: int = 50,
    output_dir: str = "./learned_reordering_output",
    seed: int = 42,
):
    """
    Jointly trains LoRA and a REOrder Plackett-Luce policy over square
    orders. Returns (lora_model, policy, repo_id, order): `lora_model` has
    its TokenReorderer set to `order`, the order of the best validation
    epoch (the policy's greedy order, or None = raster if a warm-up epoch
    was best).
    """
    if task not in ("task1", "task2", "task3"):
        raise ValueError(f"train_with_learned_reordering: unknown task '{task}'")
    repo_id = experiment_repo_id(task, "learned", tag, hf_org_prefix)
    device = model.device
    out = Path(output_dir) / repo_id.split("/")[-1]

    reorderer = install_token_reordering(model, grid_size=grid_size)
    lora_model = get_peft_model(model, peft_config)
    lora_model.train()
    policy = make_policy(grid_size).to(device)

    trainable = [p for p in lora_model.parameters() if p.requires_grad]
    model_optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=policy_learning_rate, weight_decay=0.0)
    train_split = dataset["train"]
    steps_per_epoch = math.ceil(len(train_split) / grad_accum_steps)
    scheduler = get_linear_schedule_with_warmup(model_optimizer, 0, steps_per_epoch * num_train_epochs)
    policy_steps_total = max(1, steps_per_epoch * (num_train_epochs - policy_start_after_epochs))
    val_rows = dataset["validation"].select(range(min(val_samples, len(dataset["validation"]))))

    state = {"epoch": 0, "next_row": 0, "opt_step": 0, "best_val": None, "best_epoch": None}
    resume = _find_state(repo_id, _CHECKPOINT_SUBDIR)
    if resume is not None:
        ckpt, meta = resume
        set_peft_model_state_dict(lora_model, load_peft_weights(str(ckpt)))
        policy.load_state_dict(torch.load(ckpt / "policy.pt", map_location=device))
        model_optimizer.load_state_dict(torch.load(ckpt / "model_optimizer.pt", map_location=device))
        policy_optimizer.load_state_dict(torch.load(ckpt / "policy_optimizer.pt", map_location=device))
        scheduler.load_state_dict(torch.load(ckpt / "scheduler.pt", map_location=device))
        state.update({k: meta[k] for k in state if k in meta})
        print(f"[{task}] Resuming learned reordering at epoch {state['epoch']}, row {state['next_row']} "
              f"(optimizer step {state['opt_step']}).")
    else:
        print(f"[{task}] Learned reordering: starting fresh -> {repo_id}")

    def checkpoint(message: str) -> None:
        _save_state(out / _CHECKPOINT_SUBDIR, lora_model, policy, extra=state, optimizers={
            "model_optimizer": model_optimizer, "policy_optimizer": policy_optimizer, "scheduler": scheduler})
        _upload(out / _CHECKPOINT_SUBDIR, repo_id, _CHECKPOINT_SUBDIR, message)

    for epoch in range(state["epoch"], num_train_epochs):
        policy_active = epoch >= policy_start_after_epochs
        rows = train_split.shuffle(seed=seed + epoch)
        n_rows = len(rows)
        running = {"lm": 0.0, "adv": 0.0, "n": 0}
        for i in range(state["next_row"], n_rows):
            sample = preprocess_function(_raw_sample(rows[i], task), processor)
            inputs, labels = _to_model_inputs(sample, device)

            if policy_active:
                policy_step = state["opt_step"] - steps_per_epoch * policy_start_after_epochs
                frac = min(1.0, max(0, policy_step) / policy_steps_total)
                policy.temperature = gumbel_temp_start + (gumbel_temp_end - gumbel_temp_start) * frac
                orders, log_probs, _ = policy(None, batch_size=orders_per_sample)
            else:
                orders, log_probs = [None], None

            with reorderer.order(None if orders[0] is None else orders[0].tolist()):
                lm_loss = lora_model(**inputs, labels=labels).loss
            (lm_loss / grad_accum_steps).backward()
            running["lm"] += lm_loss.item()
            running["n"] += 1

            if policy_active:
                losses = [lm_loss.detach()]
                with torch.no_grad():
                    for k in range(1, orders_per_sample):
                        with reorderer.order(orders[k].tolist()):
                            losses.append(lora_model(**inputs, labels=labels).loss)
                rewards = -torch.stack(losses).float()
                if orders_per_sample > 1:
                    # leave-one-out baseline: each order vs the others on the SAME sample
                    advantage = rewards - (rewards.sum() - rewards) / (orders_per_sample - 1)
                else:
                    advantage = rewards - policy.running_baseline
                    policy.running_baseline.copy_(baseline_momentum * policy.running_baseline
                                                  + (1 - baseline_momentum) * rewards.mean())
                policy_loss = -(advantage.detach() * log_probs).mean()
                (policy_loss / grad_accum_steps).backward()
                running["adv"] += advantage.abs().mean().item()

            if (i + 1) % grad_accum_steps == 0 or i + 1 == n_rows:
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                model_optimizer.step()
                scheduler.step()
                model_optimizer.zero_grad()
                if policy_active:
                    policy_optimizer.step()
                policy_optimizer.zero_grad()
                state["opt_step"] += 1
                state["next_row"] = i + 1
                if state["opt_step"] % log_every == 0 and running["n"]:
                    msg = f"[{task}] epoch {epoch} step {state['opt_step']}: lm_loss={running['lm'] / running['n']:.4f}"
                    if policy_active:
                        msg += f" |adv|={running['adv'] / running['n']:.4f} temp={policy.temperature:.3f}"
                    print(msg)
                    running = {"lm": 0.0, "adv": 0.0, "n": 0}
                if state["opt_step"] % checkpoint_every_steps == 0 and state["next_row"] < n_rows:
                    checkpoint(f"Learned reordering: epoch {epoch}, step {state['opt_step']}")

        # end of epoch: validation loss under the order that would be evaluated
        eval_order = greedy_order(policy) if policy_active else None
        val_loss = _validation_loss(lora_model, reorderer, processor, val_rows, task, eval_order, device)
        print(f"[{task}] epoch {epoch} done: val_loss={val_loss:.4f} "
              f"(order: {'policy greedy' if eval_order else 'raster'})")
        if state["best_val"] is None or val_loss < state["best_val"]:
            state.update({"best_val": val_loss, "best_epoch": epoch})
            _save_state(out / _BEST_SUBDIR, lora_model, policy,
                         extra={"epoch": epoch, "val_loss": val_loss, "order": eval_order})
            _upload(out / _BEST_SUBDIR, repo_id, _BEST_SUBDIR, f"Best so far: epoch {epoch}, val_loss {val_loss:.4f}")
        state.update({"epoch": epoch + 1, "next_row": 0})
        checkpoint(f"Learned reordering: end of epoch {epoch}")

    best = _find_state(repo_id, _BEST_SUBDIR)
    order = None
    if best is not None:
        best_dir, meta = best
        set_peft_model_state_dict(lora_model, load_peft_weights(str(best_dir)))
        policy.load_state_dict(torch.load(best_dir / "policy.pt", map_location=device))
        order = meta.get("order")
        print(f"[{task}] Using the best epoch ({meta['epoch']}, val_loss {meta['val_loss']:.4f}).")
    reorderer.set_order(order)

    print(f"Pushing learned-reordering model to {repo_id}...")
    lora_model.push_to_hub(repo_id, commit_message=f"Learned reordering ({task}): best epoch")
    processor.push_to_hub(repo_id)
    final = out / "final"
    final.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), final / "policy.pt")
    (final / _ORDER_FILE).write_text(json.dumps({"order": order, "greedy_order": greedy_order(policy),
                                                 "recipe": LEARNED_RECIPE}))
    for name in ("policy.pt", _ORDER_FILE):
        HfApi().upload_file(path_or_fileobj=str(final / name), path_in_repo=name, repo_id=repo_id,
                            repo_type="model", commit_message=f"Learned order + policy ({task})")
    return lora_model, policy, repo_id, order


def load_learned_order(repo_id: str) -> Optional[list]:
    """The order a finished learned-reordering run selected (None = raster), from the Hub."""
    path = hf_hub_download(repo_id, _ORDER_FILE, repo_type="model")
    return json.loads(Path(path).read_text())["order"]


def describe_order(order: Optional[list], grid_size: int = 8) -> str:
    """The order as a board diagram: each square shows its reading step (0 = read first)."""
    if order is None:
        order = list(range(grid_size * grid_size))
    step = {sq: i for i, sq in enumerate(order)}
    lines = [f"{grid_size - r} " + " ".join(f"{step[r * grid_size + c]:2d}" for c in range(grid_size))
             for r in range(grid_size)]
    lines.append("   " + "  ".join("abcdefgh"[:grid_size]))
    return "\n".join(lines)
