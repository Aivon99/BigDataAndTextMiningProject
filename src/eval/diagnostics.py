"""
Diagnostics for the chessboard tasks -- tools for finding out *why* a model
scores what it scores, not just what it scores.

Written after Task 3 came back at ~0% exact match for every model (vanilla,
LoRA, all reorderings, learned reordering). The root cause turned out to be
the training runs themselves (see `training_health_report`): every Task 3
run started before the prompt-masking fix in `preprocess_function` and was
resumed after it, so the objective changed mid-run and only the last few
epochs actually trained on the answer tokens. None of the existing metrics
could show that -- a single exact-match number can't tell "almost right"
from "garbage", and can't tell "didn't learn" from "learned but the eval is
broken". The functions here can:

- `analyze_san_predictions` / `summarize_san_diagnostics` (Task 2/3): partial
  credit for move predictions -- is the move legal, is it the *same move*
  written differently (missing "x" or "+", over/under-disambiguation), are the
  from/to squares right, is it the right piece. Pure python-chess, no GPU,
  works on the per-sample CSVs already cached on the Hub.
- `training_health_report`: reads `trainer_state.json` of a Trainer run
  from the Hub and flags eval-loss discontinuities (the signature of an
  objective change across a resume) and missing/mismatched training-recipe
  markers.
- `supervised_tokens_preview`: decodes exactly which tokens carry loss for
  one training sample -- a sanity check of the label mask.
- `frame_ablation_losses` (Task 3): teacher-forced answer loss with the two
  frames correct / swapped / duplicated. If the model actually compares the
  frames, the correct pairing must have clearly lower loss; if all
  conditions tie, the model is ignoring (at least) one frame.
- `save_diagnostics`: writes any DataFrame to Google Drive when it's mounted
  (Colab), else to a local folder, so diagnostics survive a runtime reset.

torch/transformers are imported lazily inside the GPU functions, so the
chess-level analysis can run on a laptop without a working ML stack.
"""

import json
import os
import re
from pathlib import Path
from typing import Iterable, Optional

import chess
import pandas as pd

# A SAN move anywhere in a string: castling, or [piece][disambiguation][x]square[=promo][+#].
_SAN_RE = re.compile(
    r"(O-O-O|O-O|0-0-0|0-0|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=?[QRBN])?)[+#]?"
)


def extract_san(text: str) -> str:
    """
    Best-effort SAN extraction from a model reply: looks at the first
    non-empty line only (a fine-tuned model answers there; later lines are
    repetition or reasoning, where a stray square mention would be a false
    positive -- the concern noted in `evaluate_chessboard_model_task_2`),
    and returns the first SAN-shaped token on it, or the stripped line if
    there is none.
    """
    if not isinstance(text, str):
        return ""
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    match = _SAN_RE.search(first_line)
    if not match:
        return first_line
    return match.group(0).replace("0-0-0", "O-O-O").replace("0-0", "O-O")


def _parse(board: chess.Board, san: str) -> Optional[chess.Move]:
    try:
        return board.parse_san(san)
    except Exception:
        return None


def analyze_san_predictions(results_df: pd.DataFrame, fens: Iterable[str]) -> pd.DataFrame:
    """
    Adds move-level diagnostic columns to a Task 2/3 per-sample results frame.

    `fens` must be the dataset's `fen` column in the same row order as
    `results_df` -- it is the position *before* the move (checked: every
    ground-truth target is legal in it), so a prediction can be replayed on
    the real board.

    Added columns:
      pred_san        -- `extract_san(raw_output)`
      em_extracted    -- exact match after extraction (fixes chatty/repeating output)
      multiline       -- the model kept generating after its answer
      legal           -- pred_san is a legal move in the position
      same_move       -- pred_san is the ground-truth move, however it was written
                         (e.g. "Be4" for "Bxe4", "Nf5" for "Nf5+", "Re8" for "Rge8")
      from_sq_match / to_sq_match / piece_match -- partial credit, legal moves only
    """
    df = results_df.copy().reset_index(drop=True)
    fens = list(fens)
    if len(fens) != len(df):
        raise ValueError(f"analyze_san_predictions: {len(fens)} FENs for {len(df)} results")

    raw_col = "raw_output" if "raw_output" in df.columns else "predicted"
    rows = []
    for fen, gt, raw in zip(fens, df["ground_truth"].astype(str), df[raw_col].fillna("").astype(str)):
        board = chess.Board(fen)
        pred = extract_san(raw)
        gt_move = _parse(board, gt)
        pred_move = _parse(board, pred)
        legal = pred_move is not None
        rows.append({
            "pred_san": pred,
            "em_extracted": int(pred == gt.strip()),
            "multiline": int("\n" in raw.strip()),
            "legal": int(legal),
            "same_move": int(legal and gt_move is not None and pred_move == gt_move),
            "from_sq_match": int(legal and gt_move is not None and pred_move.from_square == gt_move.from_square),
            "to_sq_match": int(legal and gt_move is not None and pred_move.to_square == gt_move.to_square),
            "piece_match": int(
                legal and gt_move is not None
                and board.piece_type_at(pred_move.from_square) == board.piece_type_at(gt_move.from_square)
            ),
        })
    return pd.concat([df, pd.DataFrame(rows)], axis=1)


SAN_DIAGNOSTIC_COLUMNS = [
    "em_extracted", "same_move", "legal", "to_sq_match", "from_sq_match", "piece_match", "multiline",
]


def summarize_san_diagnostics(analyzed_df: pd.DataFrame, model_name: Optional[str] = None) -> pd.DataFrame:
    """One-row summary (means) of `analyze_san_predictions` output, plus strict EM if present."""
    summary = {"model_name": model_name} if model_name else {}
    if "exact_match" in analyzed_df.columns:
        summary["exact_match"] = analyzed_df["exact_match"].mean()
    for col in SAN_DIAGNOSTIC_COLUMNS:
        summary[col] = analyzed_df[col].mean()
    return pd.DataFrame([summary])


def san_error_breakdown(analyzed_df: pd.DataFrame) -> pd.Series:
    """
    Buckets every prediction into one failure mode, most-correct first --
    the table the qualitative error analysis needs.
    """
    def bucket(r):
        if r["em_extracted"]:
            return "exact"
        if r["same_move"]:
            return "same move, notation differs"
        if not r["legal"]:
            return "illegal / unparseable"
        if r["to_sq_match"] and r["piece_match"]:
            return "right piece type + destination, wrong origin"
        if r["to_sq_match"]:
            return "right destination, wrong piece"
        if r["from_sq_match"]:
            return "right piece moved, wrong destination"
        return "legal but unrelated move"
    return analyzed_df.apply(bucket, axis=1).value_counts()


def rescore_cached_san_results(task: str, fens: Iterable[str], hf_org_prefix: str = "bdatm-project") -> pd.DataFrame:
    """
    Re-scores every per-sample CSV cached under
    `{hf_org_prefix}/evaluation-results-{task}/per_sample/` with the move-level
    diagnostics -- no GPU, no re-inference. `fens` is the test split's `fen`
    column (same order the CSVs were written in). Returns one row per cached
    model, best `same_move` first. Models evaluated before per-sample caching
    existed have no CSV and are simply absent.
    """
    from huggingface_hub import HfApi, hf_hub_download

    repo_id = f"{hf_org_prefix}/evaluation-results-{task}"
    fens = list(fens)
    files = [f for f in HfApi().list_repo_files(repo_id, repo_type="dataset")
             if f.startswith("per_sample/") and f.endswith(".csv")]
    rows = []
    for f in files:
        df = pd.read_csv(hf_hub_download(repo_id, f, repo_type="dataset"), keep_default_na=False)
        rows.append(summarize_san_diagnostics(analyze_san_predictions(df, fens), model_name=f[len("per_sample/"):-4]))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("same_move", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Training-run health
# ---------------------------------------------------------------------------

def training_health_report(repo_id: str, jump_threshold: float = 0.4, verbose: bool = True) -> dict:
    """
    Inspects a Trainer run pushed with `hub_strategy="checkpoint"`: eval loss
    per epoch, the train loss at start/end, and whether the run carries a
    training-recipe marker matching the current code
    (`eval.utilities.TRAINING_RECIPE`).

    Flags any epoch-to-epoch eval-loss change larger than `jump_threshold`
    (relative) *after the first epoch*. A healthy run moves smoothly; a
    sudden step (Task 3: 2.76 -> 0.75 in one epoch, at a different epoch in
    every run) means the loss itself was redefined partway through -- e.g.
    a code change picked up on resume -- so "best checkpoint" selection and
    early stopping across that point compared incomparable numbers.
    """
    from huggingface_hub import hf_hub_download

    report = {"repo_id": repo_id, "problems": []}
    try:
        path = hf_hub_download(repo_id, "last-checkpoint/trainer_state.json")
        state = json.loads(Path(path).read_text())
    except Exception as e:
        report["problems"].append(f"no trainer_state.json on the Hub ({type(e).__name__})")
        if verbose:
            print(f"[{repo_id}] {report['problems'][-1]}")
        return report

    evals = [(round(h["epoch"], 2), h["eval_loss"]) for h in state["log_history"] if "eval_loss" in h]
    trains = [(round(h["epoch"], 2), h["loss"]) for h in state["log_history"] if "loss" in h]
    report.update({
        "eval_loss": evals,
        "train_loss_first": trains[0][1] if trains else None,
        "train_loss_last": trains[-1][1] if trains else None,
        "best_metric": state.get("best_metric"),
        "global_step": state.get("global_step"),
    })

    for (e0, l0), (e1, l1) in zip(evals, evals[1:]):
        if l0 > 0 and abs(l1 - l0) / l0 > jump_threshold:
            report["problems"].append(
                f"eval_loss jumps {l0:.3f} -> {l1:.3f} between epoch {e0} and {e1}: "
                "the loss definition probably changed mid-run (resumed across a code change)"
            )

    try:
        from eval.utilities import TRAINING_RECIPE, read_training_recipe
        recipe = read_training_recipe(repo_id)
        report["recipe"] = recipe
        if recipe != TRAINING_RECIPE:
            report["problems"].append(
                f"training recipe {recipe!r} != current {TRAINING_RECIPE!r}: "
                "weights were (at least partly) trained with older code"
            )
    except ImportError:
        pass

    if verbose:
        print(f"=== {repo_id} ===")
        print("  eval_loss by epoch: " + ", ".join(f"{e:g}:{l:.3f}" for e, l in evals))
        print(f"  train loss first/last: {report['train_loss_first']} / {report['train_loss_last']}")
        for p in report["problems"] or ["no problems detected"]:
            print(f"  - {p}")
    return report


# ---------------------------------------------------------------------------
# Label mask + frame-ablation (GPU helpers, lazy imports)
# ---------------------------------------------------------------------------

def supervised_tokens_preview(sample: dict, processor) -> str:
    """
    Runs `preprocess_function` on one raw dataset row and decodes only the
    tokens whose label != -100 -- i.e. exactly what the loss is computed on.
    Expected for Task 2/3: "<think>\\n\\n</think>\\n\\n<MOVE><|im_end|>\\n".
    """
    from eval.utilities import preprocess_function

    processed = preprocess_function(sample, processor)
    labels = processed["labels"]
    supervised_ids = labels[labels != -100].tolist()
    text = processor.tokenizer.decode(supervised_ids, skip_special_tokens=False)
    print(f"{len(supervised_ids)} supervised tokens out of {len(labels)}: {text!r}")
    return text


def _answer_nll(model, processor, frames: list, prompt: str, target: str) -> float:
    """Mean NLL of the answer tokens given `frames` + `prompt` (teacher forcing)."""
    import torch
    from eval.utilities import preprocess_function

    sample = {"task": "task3" if len(frames) == 2 else "task2", "prompt": prompt, "target": target,
              "image": frames[0]}
    if len(frames) == 2:
        sample["image_t1"] = frames[1]
    p = preprocess_function(sample, processor)
    inputs = {}
    for k, v in p.items():
        if k == "labels":
            continue
        inputs[k] = (v if k in ("pixel_values", "image_grid_thw") else v.unsqueeze(0)).to(model.device)
    labels = p["labels"].unsqueeze(0).to(model.device)
    with torch.no_grad():
        return model(**inputs, labels=labels).loss.item()


def frame_ablation_losses(model, processor, dataset_split, n_samples: int = 50) -> pd.DataFrame:
    """
    Task 3: teacher-forced answer loss under four frame conditions, per sample:

      correct  -- (t, t+1), what the model is trained on
      swapped  -- (t+1, t): the "move" would run backwards
      both_t   -- (t, t):   no change visible at all
      both_t1  -- (t+1, t+1)

    A model that reads both frames must do clearly better on `correct` than
    on every other condition. Near-equal losses mean it answers from one
    frame (or from the prior over common moves) and never learned to diff.
    Note the loss also covers the trivially-predictable
    "<think></think>" / "<|im_end|>" tokens, so differences are diluted --
    compare conditions, not absolute values.
    """
    model.eval()
    n = min(n_samples, len(dataset_split))
    rows = []
    for i in range(n):
        s = dataset_split[i]
        t, t1 = s["image"], s["image_t1"]
        rows.append({
            "sample_id": s["sample_id"],
            "correct": _answer_nll(model, processor, [t, t1], s["prompt"], s["target"]),
            "swapped": _answer_nll(model, processor, [t1, t], s["prompt"], s["target"]),
            "both_t": _answer_nll(model, processor, [t, t], s["prompt"], s["target"]),
            "both_t1": _answer_nll(model, processor, [t1, t1], s["prompt"], s["target"]),
        })
    df = pd.DataFrame(rows)
    means = df.drop(columns="sample_id").mean()
    print("Mean answer NLL by frame condition (lower = more confident in the true move):")
    print(means.round(4).to_string())
    print(f"'correct' is the lowest-loss condition on {(df[['correct','swapped','both_t','both_t1']].idxmin(axis=1) == 'correct').mean():.0%} of samples")
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

DRIVE_ROOT = "/content/drive/MyDrive/bdatm_diagnostics"


def save_diagnostics(df: pd.DataFrame, name: str, task: str, local_root: str = "./diagnostics") -> Path:
    """
    Saves `df` as CSV under `<root>/<task>/<name>.csv`, where root is Google
    Drive (`DRIVE_ROOT`) if Drive is mounted in this Colab runtime, else
    `local_root`. Drive survives a Colab runtime reset; the local folder does not.
    Mount Drive first with `from google.colab import drive; drive.mount('/content/drive')`.
    """
    root = Path(DRIVE_ROOT) if os.path.isdir("/content/drive/MyDrive") else Path(local_root)
    out = root / task / f"{re.sub(r'[^A-Za-z0-9_.+-]+', '_', name)}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved diagnostics to {out}")
    return out
