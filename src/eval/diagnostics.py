"""Extra analysis of the model outputs, used to understand the errors and to
check the training runs.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

import chess
import pandas as pd

_SAN_RE = re.compile(
    r"(O-O-O|O-O|0-0-0|0-0|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=?[QRBN])?)[+#]?"
)


def extract_san(text):
    """Takes the SAN move from the first line of the model output."""
    if not isinstance(text, str):
        return ""
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    match = _SAN_RE.search(first_line)
    if not match:
        return first_line
    return match.group(0).replace("0-0-0", "O-O-O").replace("0-0", "O-O")


def _parse(board: chess.Board, san) -> Optional[chess.Move]:
    try:
        return board.parse_san(san)
    except Exception:
        return None


_SAN_PARTS_RE = re.compile(r"^([KQRBN])?([a-h])?([1-8])?x?([a-h][1-8])(?:=?([QRBN]))?[+#]?$")
_PIECES = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT}


def _ambiguous_candidates(board: chess.Board, san):
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


def analyze_san_predictions(results_df, fens):
    """Adds move level columns (legal move, same move, right squares, right
    piece) to the Task 2/3 results.

    fens must be the fen column of the dataset, in the same order as
    results_df.
    """
    df = results_df.copy().reset_index(drop=True)
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


def san_error_breakdown(analyzed_df):
    """Counts how many predictions fall into each type of error."""
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


FEN_DIAGNOSTIC_COLUMNS = [
    "board_exact_match", "board_character_error_rate",
    "side_to_move_match", "castling_match", "en_passant_match", "clocks_match",
]


def _levenshtein(a, b):
    try:
        import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def analyze_fen_predictions(results_df):
    """Adds one column per FEN field to the Task 1 results. Only the board part
    can really be seen in the image.
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


def fen_error_breakdown(results_df, n_examples=5):
    """
    Qualitative error analysis for Task 1 (board field only, what the image shows):
      wrong_squares_per_board: how many of the 64 squares are wrong per prediction
                                ("unparseable" if the board field isn't valid FEN)
      confusions:              (true -> predicted) piece pairs over all wrong squares,
                                '.' = empty square, most frequent first
      examples:                a few failed predictions with their wrong squares
    """
    import chess

    def board_of(fen):
        try:
            return chess.Board(str(fen).split()[0] + " w - - 0 1")
        except (ValueError, IndexError):
            return None

    per_board, confusions, examples = [], [], []
    for gt, pred in zip(results_df["ground_truth"].astype(str), results_df["predicted"].fillna("").astype(str)):
        g, p = board_of(gt), board_of(pred)
        if g is None:
            continue
        if p is None:
            per_board.append("unparseable")
            if len(examples) < n_examples:
                examples.append({"ground_truth": gt.split()[0], "predicted": pred, "wrong_squares": "unparseable"})
            continue
        wrong = []
        for sq in chess.SQUARES:
            a, b = g.piece_at(sq), p.piece_at(sq)
            if a != b:
                pair = f"{a.symbol() if a else '.'} -> {b.symbol() if b else '.'}"
                confusions.append(pair)
                wrong.append(f"{chess.square_name(sq)}: {pair}")
        per_board.append(len(wrong))
        if wrong and len(examples) < n_examples:
            examples.append({"ground_truth": gt.split()[0], "predicted": pred.split()[0], "wrong_squares": ", ".join(wrong)})
    return {
        "wrong_squares_per_board": pd.Series(per_board, name="boards").value_counts(),
        "confusions": pd.Series(confusions, name="count").value_counts(),
        "examples": pd.DataFrame(examples),
    }


# Main metrics in display order. Task 1: FEN exact match and CER on the board
# field, the only part of a FEN the image shows. Task 2/3: Exact Match is
# em_extracted, the move on the answer line.
HEADLINE_COLUMNS = {
    "task1": {
        "board_exact_match": "FEN Exact Match (board)",
        "board_character_error_rate": "CER (board)",
        "square_by_square_accuracy": "Square-by-Square Acc",
        "fen_exact_match": "FEN EM (full string)",
        "character_error_rate": "CER (full string)",
        "levenshtein_distance": "Levenshtein (full string)",
    },
    "task2": {
        "em_extracted": "Exact Match",
        "same_move": "Same move (any notation)",
        "legal": "Legal move",
        "exact_match": "EM raw output",
    },
}
HEADLINE_COLUMNS["task3"] = HEADLINE_COLUMNS["task2"]


def experiment_setup(model_name):
    """Which experimental setup a results row belongs to, from its name tag."""
    m = re.search(r"\[([^\]]+)\]\s*$", str(model_name))
    if not m:
        return "legacy: pixel-tile reordering, 512px"
    tag = m.group(1)
    t = re.fullmatch(r"tok(\d+)", tag)
    return f"token reordering, {t.group(1)}px" if t else tag


def headline_table(table, task):
    """
    The comparison table with the main metrics first (renamed), a `setup`
    column (token-level runs vs the legacy pixel-tile runs), best first.
    """
    cols = HEADLINE_COLUMNS[task]
    out = table.copy()
    out.insert(1, "setup", out["model_name"].map(experiment_setup))
    keep = ["model_name", "setup"] + [c for c in cols if c in out.columns]
    out = out[keep].rename(columns=cols)
    sort_by = next((name for col, name in cols.items() if col in table.columns), None)
    if sort_by is not None and sort_by.startswith("CER"):
        return out.sort_values(sort_by).reset_index(drop=True)
    return (out.sort_values(sort_by, ascending=False) if sort_by else out).reset_index(drop=True)


def backfill_cached_metrics(
    table,
    task,
    fens=None,
    hf_org_prefix="bdatm-project",
    drop_models=(),
):
    """Recomputes the diagnostic columns of the results table from the
    per-sample files on the Hub. Models in drop_models are removed.
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


def training_health_report(repo_id, jump_threshold=0.4, verbose=True):
    """Reads the training logs of a run from the Hub and flags sudden jumps in
    the eval loss.
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
        from training.checkpoints import TRAINING_RECIPE, read_training_recipe
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


def _split_frame_patches(pixel_values, image_grid_thw):
    sizes = [int(n) for n in image_grid_thw.prod(dim=-1).tolist()]
    if len(sizes) != 2 or sizes[0] != sizes[1] or pixel_values.shape[0] != sum(sizes):
        return None
    return pixel_values[: sizes[0]], pixel_values[sizes[0]:]


def frame_ablation_losses(model, processor, dataset_split, n_samples=50):
    """Task 3 check: compares the loss with the two frames in the right order,
    swapped or repeated. If the model really uses both frames, the right
    order should have the lowest loss.
    """
    import torch
    from tqdm.auto import tqdm
    from training.preprocessing import preprocess_function

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


DRIVE_ROOT = "/content/drive/MyDrive/bdatm_diagnostics"


def save_diagnostics(df, name, task, local_root="./diagnostics"):
    """Saves a DataFrame as CSV, on Google Drive if it is mounted, otherwise in
    a local folder.
    """
    root = Path(DRIVE_ROOT) if os.path.isdir("/content/drive/MyDrive") else Path(local_root)
    out = root / task / f"{re.sub(r'[^A-Za-z0-9_.+-]+', '_', name)}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved diagnostics to {out}")
    return out
