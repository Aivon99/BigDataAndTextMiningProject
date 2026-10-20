"""Creates the chessboard images and metadata for the three task datasets."""

import io
import json
import multiprocessing
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cairosvg
import chess
import chess.svg
import pandas as pd
from PIL import Image
from tqdm import tqdm


PROMPTS = {
    "task1": (
        "You are a specialized model for chessboard understanding.\n"
        "Your goal is to extract the exact board state from the provided chessboard image.\n"
        "Input:\n"
        "- Board Image: The visual representation of the chessboard.\n"
        "Output Format:\n"
        "Return only the valid FEN string representing the position of all pieces on the board."
    ),
    "task2": (
        "You are a specialized model for chessboard understanding.\n"
        "Your goal is to predict the last move played on the board based on the highlighted squares.\n"
        "Input:\n"
        "- Board Image: The visual representation with the last move highlighted.\n"
        "Output Format:\n"
        "Return only the move in Standard Algebraic Notation (SAN)."
    ),
    "task3": (
        "You are a specialized model for chessboard temporal reasoning.\n"
        "Your goal is to identify the move that transitioned the chessboard from State t to State t+1.\n"
        "Input:\n"
        "- Frame 1: Chessboard state at time t.\n"
        "- Frame 2: Chessboard state at time t+1.\n"
        "Output Format:\n"
        "Return only the move in Standard Algebraic Notation (SAN)."
    ),
}


def render_board_svg(
    board: chess.Board,
    size=512,
    lastmove: Optional[chess.Move] = None,
) -> Image.Image:
    """Draws the board as a PIL image. Coordinates are turned off so the 8x8
    squares fill the whole image.
    """
    svg_data = chess.svg.board(board=board, size=size, lastmove=lastmove, coordinates=False)
    png_bytes = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def build_sample(
    fen,
    task,
    sample_id,
    moves=None,
    puzzle_id=None,
    output_dir=None,
    image_size=512,
) -> Dict[str, Union[List[Image.Image], dict]]:
    """Builds one sample (images, prompt and target) for task1, task2 or task3."""
    if task not in PROMPTS:
        raise ValueError(f"Unsupported task '{task}'. Expected one of: {list(PROMPTS.keys())}")

    board = chess.Board(fen)
    first_move_uci = moves.split()[0] if moves and isinstance(moves, str) else None
    first_move = chess.Move.from_uci(first_move_uci) if first_move_uci else None

    images: List[Image.Image] = []
    saved_filenames: List[str] = []

    if task == "task1":
        img = render_board_svg(board=board, size=image_size)
        images.append(img)
        saved_filenames.append("board.png")
        target = fen

    elif task == "task2":
        if first_move is None:
            raise ValueError(f"Task 2 requires a valid move sequence, none found for FEN: {fen}")

        target = board.san(first_move)
        board.push(first_move)
        img = render_board_svg(board=board, size=image_size, lastmove=first_move)

        images.append(img)
        saved_filenames.append("board.png")

    elif task == "task3":
        if first_move is None:
            raise ValueError(f"Task 3 requires a valid move sequence, none found for FEN: {fen}")
        target = board.san(first_move)

        img_t = render_board_svg(board=board, size=image_size)
        board.push(first_move)
        img_t1 = render_board_svg(board=board, size=image_size)

        images.extend([img_t, img_t1])
        saved_filenames.extend(["board_t.png", "board_t_plus_1.png"])

    metadata = {
        "sample_id": sample_id,
        "puzzle_id": puzzle_id if puzzle_id is not None else sample_id,
        "task": task,
        "fen": fen,
        "prompt": PROMPTS[task],
        "target": target,
        "image_count": len(images),
        "image_files": saved_filenames,
        "image_size": image_size,
        "patch_order": "row_major",
        "patch_size": 16,
    }

    if output_dir is not None:
        sample_dir = Path(output_dir) / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        for img, fname in zip(images, saved_filenames):
            img.save(sample_dir / fname)

    return {"images": images, "metadata": metadata}


def _render_sample_to_disk(args):
    fen, task, sample_id, moves, puzzle_id, output_dir, image_size = args
    result = build_sample(
        fen=fen,
        task=task,
        sample_id=sample_id,
        moves=moves,
        puzzle_id=puzzle_id,
        output_dir=output_dir,
        image_size=image_size,
    )
    return result["metadata"]


RENDER_VERSION = "no-coordinates-v2"


def _stamp_path(output_root, task, split_name):
    return output_root / ".render_stamps" / f"{task}_{split_name}"


def _existing_sample_count(output_dir):
    metadata_path = output_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return 0
    with open(metadata_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def generate_datasets_parallel(
    task_splits: Dict[str, Dict[str, pd.DataFrame]],
    output_root=".",
    image_size=512,
    num_workers=None,
    skip_existing=True,
):
    """Generates all the task/split datasets in parallel and writes a
    metadata.jsonl file for each one.

    If skip_existing is True, a split that is already complete on disk is
    skipped.
    """
    output_root = Path(output_root)

    work_items: List[tuple] = []
    job_dirs: List[Path] = []

    for task, splits in task_splits.items():
        for split_name, split_df in splits.items():
            output_dir = output_root / f"dataset_{task}" / split_name
            output_dir.mkdir(parents=True, exist_ok=True)

            stamp = _stamp_path(output_root, task, split_name)
            stamp_ok = stamp.exists() and stamp.read_text().strip() == RENDER_VERSION
            if skip_existing and stamp_ok and _existing_sample_count(output_dir) == len(split_df):
                print(
                    f"[{task}/{split_name}] already has {len(split_df)} sample(s) "
                    f"rendered with '{RENDER_VERSION}', skipping."
                )
                continue
            if skip_existing and _existing_sample_count(output_dir) == len(split_df):
                print(
                    f"[{task}/{split_name}] has {len(split_df)} sample(s) from an older render "
                    f"version, re-rendering."
                )

            for idx, (_, row) in enumerate(split_df.iterrows()):
                sample_id = f"sample_{idx:06d}"
                work_items.append((
                    row["FEN"],
                    task,
                    sample_id,
                    row.get("Moves", None),
                    str(row.get("PuzzleId", sample_id)),
                    output_dir,
                    image_size,
                ))
                job_dirs.append(output_dir)

    if not work_items:
        print("Nothing to generate, every (task, split) combination is already up to date.")
        return

    num_workers = num_workers or os.cpu_count() or 1
    print(
        f"Rendering {len(work_items)} samples across {len(task_splits)} task(s) "
        f"using {num_workers} worker process(es)..."
    )

    records_by_dir: Dict[Path, List[dict]] = {}
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.imap(_render_sample_to_disk, work_items, chunksize=8)
        for output_dir, metadata in tqdm(
            zip(job_dirs, results), total=len(work_items), desc="Generating datasets"
        ):
            record = {
                "sample_id": metadata["sample_id"],
                "puzzle_id": metadata["puzzle_id"],
                "task": metadata["task"],
                "fen": metadata["fen"],
                "prompt": metadata["prompt"],
                "target": metadata["target"],
                "file_name": f"{metadata['sample_id']}/{metadata['image_files'][0]}",
                "file_name_t1": (
                    f"{metadata['sample_id']}/{metadata['image_files'][1]}"
                    if len(metadata["image_files"]) > 1 else ""
                ),
            }
            records_by_dir.setdefault(output_dir, []).append(record)

    for output_dir, records in records_by_dir.items():
        metadata_path = output_dir / "metadata.jsonl"
        with open(metadata_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    for task, splits in task_splits.items():
        for split_name in splits:
            if output_root / f"dataset_{task}" / split_name in records_by_dir:
                stamp = _stamp_path(output_root, task, split_name)
                stamp.parent.mkdir(parents=True, exist_ok=True)
                stamp.write_text(RENDER_VERSION)

    print("All task/split datasets generated.")
