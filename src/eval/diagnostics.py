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


_SAN_PARTS_RE = re.compile(r"^([KQRBN])?([a-h])?([1-8])?x?([a-h][1-8])(?:=?([QRBN]))?[+#]?$")
_PIECES = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT}


def _ambiguous_candidates(board: chess.Board, san: str) -> list:
    """
    Legal moves an under-disambiguated SAN could mean ("Rc8" when both rooks
    can reach c8). python-chess refuses to parse such SAN; the model,
    though, has identified piece and destination -- and in Task 2 the
    origin square is highlighted -- so it's a notation slip, not a wrong move.
    """
    m = _SAN_PARTS_RE.match(san)
    if not m:
        return []
    piece, from_file, from_rank, dest, promo = m.groups()
    piece_type = _PIECES[piece] if piece else chess.PAWN
    to_sq = chess.parse_square(dest)
    promotion = _PIECES[promo] if promo else None
    return [
        mv for mv in board.legal_moves
        if mv.to_square == to_sq
        and board.piece_type_at(mv.from_square) == piece_type
        and mv.promotion == promotion
        and (from_file is None or chess.square_file(mv.from_square) == "abcdefgh".index(from_file))
        and (from_rank is None or chess.square_rank(mv.from_square) == int(from_rank) - 1)
    ]


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
      legal           -- pred_san denotes a legal move in the position (an
                         under-disambiguated SAN counts if some legal move fits it)
      ambiguous       -- pred_san is under-disambiguated ("Rc8" with two rooks able to go)
      same_move       -- pred_san is the ground-truth move, however it was written
                         (e.g. "Be4" for "Bxe4", "Nf5" for "Nf5+", "Rc8" for "Rac8"
                         -- ambiguous SAN counts if the true move is one of its readings)
      from_sq_match / to_sq_match / piece_match -- partial credit, legal moves only

    Most strict-EM misses of the fine-tuned Task 2 model are notation, not
    perception: a missing/extra capture "x" (the post-move image doesn't show
    whether a piece was taken), a missing "+", or missing disambiguation.
    """
    df = results_df.copy().reset_index(drop=True)
    # Re-analysing a CSV that already has these columns (any evaluation since
    # they were added) must replace them, not duplicate them.
    df = df.drop(columns=[c for c in ["pred_san", *SAN_DIAGNOSTIC_COLUMNS] if c in df.columns])
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
        ambiguous = False
        if pred_move is None:
            candidates = _ambiguous_candidates(board, pred)
            if len(candidates) > 1:
                ambiguous = True
                # credit the true move if it's one of the readings, else any reading
                pred_move = gt_move if gt_move in candidates else candidates[0]
        legal = pred_move is not None
        rows.append({
            "pred_san": pred,
            "em_extracted": int(pred == gt.strip()),
            "multiline": int("\n" in raw.strip()),
            "legal": int(legal),
            "ambiguous": int(ambiguous),
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
    "em_extracted", "same_move", "legal", "ambiguous", "to_sq_match", "from_sq_match", "piece_match", "multiline",
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
        if r["same_move"] and r["ambiguous"]:
            return "same move, under-disambiguated"
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


# ---------------------------------------------------------------------------
# Task 1: FEN field-level analysis
# ---------------------------------------------------------------------------

FEN_DIAGNOSTIC_COLUMNS = [
    "board_exact_match", "board_character_error_rate",
    "side_to_move_match", "castling_match", "en_passant_match", "clocks_match",
]


def _levenshtein(a: str, b: str) -> int:
    try:
        import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:  # small pure-python fallback, FEN strings are short
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def analyze_fen_predictions(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-field columns to a Task 1 per-sample results frame.

    A FEN has six fields, and only the first is fully visible in the image:
    the board is always drawn from White's side with no turn marker, and the
    halfmove clock / fullmove number can't be seen at all. Full-FEN exact
    match therefore measures guessing as much as perception (Task 1 after
    retraining: 98.8% of boards exactly right, 0.8% full-FEN exact match;
    the model outputs halfmove "0" for 99% of samples). The board fields are
    the perception metric; the others are reported separately.

      board_exact_match           -- piece placement (field 1) identical
      board_character_error_rate  -- CER on field 1 only
      side_to_move_match / castling_match / en_passant_match -- fields 2-4
      clocks_match                -- fields 5-6 (halfmove, fullmove) both right
    """
    df = results_df.copy().reset_index(drop=True)
    df = df.drop(columns=[c for c in FEN_DIAGNOSTIC_COLUMNS if c in df.columns])
    rows = []
    for gt, pred in zip(df["ground_truth"].fillna("").astype(str), df["predicted"].fillna("").astype(str)):
        g, p = gt.split(), pred.split()
        field = lambda f, i: f[i] if len(f) > i else ""
        g_board, p_board = field(g, 0), field(p, 0)
        rows.append({
            "board_exact_match": int(g_board == p_board and g_board != ""),
            "board_character_error_rate": _levenshtein(p_board, g_board) / max(len(g_board), 1),
            "side_to_move_match": int(field(g, 1) == field(p, 1)),
            "castling_match": int(field(g, 2) == field(p, 2)),
            "en_passant_match": int(field(g, 3) == field(p, 3)),
            "clocks_match": int(field(g, 4) == field(p, 4) and field(g, 5) == field(p, 5)),
        })
    return pd.concat([df, pd.DataFrame(rows)], axis=1)


def rescore_cached_fen_results(hf_org_prefix: str = "bdatm-project") -> pd.DataFrame:
    """
    Task 1 counterpart of `rescore_cached_san_results`: per-field FEN metrics
    for every per-sample CSV cached on the Hub, no GPU. One row per model,
    best `board_exact_match` first.
    """
    from huggingface_hub import HfApi, hf_hub_download

    repo_id = f"{hf_org_prefix}/evaluation-results-task1"
    files = [f for f in HfApi().list_repo_files(repo_id, repo_type="dataset")
             if f.startswith("per_sample/") and f.endswith(".csv")]
    rows = []
    for f in files:
        df = analyze_fen_predictions(pd.read_csv(hf_hub_download(repo_id, f, repo_type="dataset"), keep_default_na=False))
        summary = {"model_name": f[len("per_sample/"):-4]}
        for col in ["fen_exact_match", "square_by_square_accuracy", "character_error_rate", *FEN_DIAGNOSTIC_COLUMNS]:
            if col in df.columns:
                summary[col] = df[col].mean()
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("board_exact_match", ascending=False).reset_index(drop=True)


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


def backfill_cached_metrics(
    table: pd.DataFrame,
    task: str,
    fens: Optional[Iterable[str]] = None,
    hf_org_prefix: str = "bdatm-project",
    drop_models: Iterable[str] = (),
) -> pd.DataFrame:
    """
    Recomputes the diagnostic columns (FEN_DIAGNOSTIC_COLUMNS for task1,
    SAN_DIAGNOSTIC_COLUMNS for task2/3) of a model-comparison table from the
    per-sample CSVs cached on the Hub -- no inference.

    Rows evaluated before those columns existed otherwise show NaN next to
    rows evaluated after, and rows evaluated under an older version of a
    metric (e.g. before ambiguous SAN was credited) would mix definitions.
    The per-sample predictions are the source of truth, so every row with a
    CSV gets all its diagnostic columns recomputed with the current code;
    strict metrics (exact_match, CER, ...) are left as they are. A row is
    matched to `per_sample/<slug>.csv` with the same slug
    `evaluate_with_cache` writes; rows without a CSV (evaluated before
    per-sample caching) keep whatever they have. Rows named in `drop_models` are
    removed (e.g. stale results of a broken run). `fens` (the test split's
    `fen` column) is required for task2/task3.
    """
    from huggingface_hub import HfApi, hf_hub_download

    if task not in ("task1", "task2", "task3"):
        raise ValueError(f"backfill_cached_metrics: unknown task '{task}'")
    if task != "task1" and fens is None:
        raise ValueError("backfill_cached_metrics: `fens` is required for task2/task3")
    fens = list(fens) if fens is not None else None
    cols = FEN_DIAGNOSTIC_COLUMNS if task == "task1" else SAN_DIAGNOSTIC_COLUMNS

    repo_id = f"{hf_org_prefix}/evaluation-results-{task}"
    cached = {
        f[len("per_sample/"):-4]: f for f in HfApi().list_repo_files(repo_id, repo_type="dataset")
        if f.startswith("per_sample/") and f.endswith(".csv")
    }
    out = table[~table["model_name"].isin(list(drop_models))].reset_index(drop=True).copy()
    for col in cols:
        if col not in out.columns:
            out[col] = float("nan")
    filled = []
    for i, name in out["model_name"].items():
        slug = str(name).lower().replace(" ", "_")
        if slug not in cached:
            continue
        df = pd.read_csv(hf_hub_download(repo_id, cached[slug], repo_type="dataset"), keep_default_na=False)
        analyzed = analyze_fen_predictions(df) if task == "task1" else analyze_san_predictions(df, fens)
        for col in cols:
            out.loc[i, col] = analyzed[col].mean()
        filled.append(name)
    missing = [n for n in out["model_name"] if str(n).lower().replace(" ", "_") not in cached]
    print(f"[{task}] recomputed diagnostics for {len(filled)} row(s) from per-sample CSVs"
          + (f"; no per-sample CSV (left as NaN): {missing}" if missing else ""))
    return out


# ---------------------------------------------------------------------------
# Training-run health
# ---------------------------------------------------------------------------

def training_health_report(repo_id: str, jump_threshold: float = 0.4, verbose: bool = True) -> dict:
    """
    Inspects a Trainer run pushed with `hub_strategy="checkpoint"`: eval loss
    per epoch, the train loss at start/end, and whether the run carries a
    training-recipe marker matching the current code
    (`eval.utilities.TRAINING_RECIPE`).

    Flags an epoch-to-epoch eval-loss change larger than `jump_threshold`
    (relative) that comes right after a plateau (previous change < 10%).
    Healthy runs drop steeply in the first epochs and then flatten; a step
    *after* flattening (Task 3: 2.76 -> 0.75 in one epoch, at a different
    epoch in every run) means the loss itself was redefined partway through
    -- e.g. a code change picked up on resume -- so best-checkpoint
    selection and early stopping across that point compared incomparable
    numbers.

    Also reports the best vs the last evaluated epoch, and whether the best
    epoch's adapter is on the Hub (`best-checkpoint/`, written by
    HubBestAdapterCallback): a run resumed in a new session before that
    callback existed published its last epoch, not its best.

    Learned-reordering repos (custom loop, no Trainer state) report their
    checkpoint metadata and recipe instead.
    """
    from huggingface_hub import HfApi, hf_hub_download

    report = {"repo_id": repo_id, "problems": []}
    files = HfApi().list_repo_files(repo_id, repo_type="model")
    if "last-checkpoint/trainer_state.json" not in files:
        if "last-learned-checkpoint/metadata.json" in files:
            meta = json.loads(Path(hf_hub_download(repo_id, "last-learned-checkpoint/metadata.json")).read_text())
            report.update({"learned_checkpoint": meta, "recipe": meta.get("recipe")})
            if verbose:
                print(f"=== {repo_id} === (learned-reordering loop, no Trainer state)\n  checkpoint metadata: {meta}")
            return report
        report["problems"].append("no trainer_state.json on the Hub")
        if verbose:
            print(f"=== {repo_id} ===\n  - {report['problems'][-1]}")
        return report
    state = json.loads(Path(hf_hub_download(repo_id, "last-checkpoint/trainer_state.json")).read_text())

    evals = [(round(h["epoch"], 2), h["eval_loss"]) for h in state["log_history"] if "eval_loss" in h]
    trains = [(round(h["epoch"], 2), h["loss"]) for h in state["log_history"] if "loss" in h]
    report.update({
        "eval_loss": evals,
        "train_loss_first": trains[0][1] if trains else None,
        "train_loss_last": trains[-1][1] if trains else None,
        "best_metric": state.get("best_metric"),
        "global_step": state.get("global_step"),
    })

    rel = [abs(l1 - l0) / l0 if l0 > 0 else 0.0 for (_, l0), (_, l1) in zip(evals, evals[1:])]
    for i in range(1, len(rel)):
        if rel[i] > jump_threshold and rel[i - 1] < 0.10:
            (e0, l0), (e1, l1) = evals[i], evals[i + 1]
            report["problems"].append(
                f"eval_loss jumps {l0:.3f} -> {l1:.3f} between epoch {e0} and {e1} after a plateau: "
                "the loss definition probably changed mid-run (resumed across a code change)"
            )

    if evals:
        best_epoch, best_loss = min(evals, key=lambda x: x[1])
        last_epoch, last_loss = evals[-1]
        report.update({"best_epoch": best_epoch, "last_epoch": last_epoch})
        if best_epoch != last_epoch and not any(f.startswith("best-checkpoint/") for f in files):
            report["problems"].append(
                f"best epoch {best_epoch:g} (eval_loss {best_loss:.4f}) != last epoch {last_epoch:g} ({last_loss:.4f}) "
                "and no best-checkpoint/ on the Hub: if this run was resumed in a new session, the published "
                "adapter is the last epoch's, not the best"
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


def _split_frame_patches(pixel_values, image_grid_thw):
    """
    Splits the processor's flattened `pixel_values` of a two-image sample into
    (frame 1 patches, frame 2 patches), using `image_grid_thw` (rows per image
    = t*h*w). Returns None unless both frames have the same patch grid -- the
    only case where frames can be swapped without changing the token sequence.
    """
    sizes = [int(n) for n in image_grid_thw.prod(dim=-1).tolist()]
    if len(sizes) != 2 or sizes[0] != sizes[1] or pixel_values.shape[0] != sum(sizes):
        return None
    return pixel_values[: sizes[0]], pixel_values[sizes[0]:]


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

    Each sample is preprocessed once, with `preprocess_function` (the
    training path, so tokens and labels are exactly what the model was
    trained on). Both frames are 512px, so all four conditions share the same
    token sequence and differ only in the order of the two frames' image
    patches -- the conditions are built by reordering those, not by
    re-running the (CPU-bound) image processor four times.
    """
    import torch
    from tqdm.auto import tqdm
    from eval.utilities import preprocess_function

    model.eval()
    n = min(n_samples, len(dataset_split))
    rows = []
    for i in tqdm(range(n), desc="Frame ablation"):
        s = dataset_split[i]
        p = preprocess_function(
            {"task": "task3", "prompt": s["prompt"], "target": s["target"],
             "image": s["image"], "image_t1": s["image_t1"]},
            processor,
        )
        frames = _split_frame_patches(p["pixel_values"], p["image_grid_thw"])
        if frames is None:
            raise ValueError(
                f"{s['sample_id']}: the two frames don't have identical patch grids "
                f"({p['image_grid_thw'].tolist()}), so they can't be swapped in place"
            )
        f_t, f_t1 = frames
        shared = {
            k: (v if k == "image_grid_thw" else v.unsqueeze(0)).to(model.device)
            for k, v in p.items() if k not in ("labels", "pixel_values")
        }
        labels = p["labels"].unsqueeze(0).to(model.device)
        row = {"sample_id": s["sample_id"]}
        conditions = {"correct": (f_t, f_t1), "swapped": (f_t1, f_t), "both_t": (f_t, f_t), "both_t1": (f_t1, f_t1)}
        for name, (first, second) in conditions.items():
            with torch.no_grad():
                out = model(**shared, pixel_values=torch.cat([first, second]).to(model.device), labels=labels)
            row[name] = out.loss.item()
        rows.append(row)
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
