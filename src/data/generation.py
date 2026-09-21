import io
import json
import multiprocessing
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import cairosvg
import chess
import chess.svg
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
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
    size: int = 512,
    lastmove: Optional[chess.Move] = None,
) -> Image.Image:
    """Renders a chess.Board state to an RGB PIL Image via SVG rasterization.

    coordinates=False on purpose: with python-chess's default labels the board
    sits inside a ~20px margin, so squares are ~59px instead of size/8 and no
    longer line up with the ViT patch grid or the 8x8 tiles used for patch
    reordering. Without labels the 8x8 squares fill the image exactly.
    """
    svg_data = chess.svg.board(board=board, size=size, lastmove=lastmove, coordinates=False)
    png_bytes = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def build_sample(
    fen: str,
    task: str,
    sample_id: str,
    moves: Optional[str] = None,
    puzzle_id: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    image_size: int = 512,
) -> Dict[str, Union[List[Image.Image], dict]]:
    """Builds a single multimodal sample for Task 1, Task 2, or Task 3."""
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


def iter_task_samples(
    df: pd.DataFrame,
    task: str,
    image_size: int = 512,
) -> Iterator[Dict[str, Union[Image.Image, str]]]:
    """
    Lazily yields one flattened sample per row for `task`, rendering images
    on the fly and never writing to disk. Feed directly into
    `datasets.Dataset.from_generator` so a full task/split dataset can be
    built and pushed to the Hub without holding all images in memory at once.
    """
    if task not in PROMPTS:
        raise ValueError(f"Unsupported task '{task}'. Expected one of: {list(PROMPTS.keys())}")

    for idx, (_, row) in enumerate(df.iterrows()):
        sample_id = f"sample_{idx:06d}"
        fen = row["FEN"]
        moves = row.get("Moves", None)
        puzzle_id = str(row.get("PuzzleId", sample_id))

        board = chess.Board(fen)
        first_move_uci = moves.split()[0] if moves and isinstance(moves, str) else None
        first_move = chess.Move.from_uci(first_move_uci) if first_move_uci else None

        base = {
            "sample_id": sample_id,
            "puzzle_id": puzzle_id,
            "fen": fen,
            "prompt": PROMPTS[task],
        }

        if task == "task1":
            yield {
                **base,
                "image": render_board_svg(board=board, size=image_size),
                "target": fen,
            }

        elif task == "task2":
            if first_move is None:
                raise ValueError(f"Task 2 requires a valid move sequence, none found for FEN: {fen}")
            yield {
                **base,
                "image": render_board_svg(board=board, size=image_size, lastmove=first_move),
                "target": board.san(first_move),
            }

        elif task == "task3":
            if first_move is None:
                raise ValueError(f"Task 3 requires a valid move sequence, none found for FEN: {fen}")
            target = board.san(first_move)
            image_t = render_board_svg(board=board, size=image_size)
            board.push(first_move)
            image_t1 = render_board_svg(board=board, size=image_size)
            yield {
                **base,
                "image_t": image_t,
                "image_t1": image_t1,
                "target": target,
            }


def generate_dataset(
    df: pd.DataFrame,
    task: str,
    output_dir: Union[str, Path],
    image_size: int = 512,
) -> None:
    """Generates a batch dataset for a specific task and creates a clean metadata.jsonl for Hugging Face ImageFolder."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_records = []

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"Generating {task}")):
        sample_id = f"sample_{idx:06d}"
        
        result = build_sample(
            fen=row["FEN"],
            moves=row.get("Moves", None),
            task=task,
            sample_id=sample_id,
            puzzle_id=str(row.get("PuzzleId", sample_id)),
            output_dir=output_path,
            image_size=image_size,
        )

        metadata = result["metadata"]
        saved_files = metadata["image_files"]

        # Base record with uniform schema across all tasks to prevent CastError
        record = {
            "sample_id": sample_id,
            "puzzle_id": metadata["puzzle_id"],
            "task": task,
            "fen": metadata["fen"],
            "prompt": metadata["prompt"],
            "target": metadata["target"],
            "file_name": f"{sample_id}/{saved_files[0]}",
            "file_name_t1": f"{sample_id}/{saved_files[1]}" if len(saved_files) > 1 else ""
        }

        dataset_records.append(record)

    metadata_file_path = output_path / "metadata.jsonl"
    with open(metadata_file_path, "w", encoding="utf-8") as f:
        for record in dataset_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _render_sample_to_disk(args: tuple) -> dict:
    """
    Worker entry point for `generate_datasets_parallel`. Renders one sample
    straight to disk via `build_sample` and returns only its metadata dict —
    never the PIL images — so results stay cheap to ship back from a worker
    process to the main one over IPC.
    """
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


# Bump whenever the rendered images change (e.g. board coordinates removed) so
# `skip_existing` doesn't keep stale images that only *look* up to date.
RENDER_VERSION = "no-coordinates-v2"


def _stamp_path(output_root: Path, task: str, split_name: str) -> Path:
    # Kept outside dataset_<task>/ so the stamp never gets uploaded to the Hub.
    return output_root / ".render_stamps" / f"{task}_{split_name}"


def _existing_sample_count(output_dir: Path) -> int:
    """
    Number of samples already recorded in `output_dir/metadata.jsonl`, or 0
    if that file doesn't exist yet.
    """
    metadata_path = output_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return 0
    with open(metadata_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def generate_datasets_parallel(
    task_splits: Dict[str, Dict[str, pd.DataFrame]],
    output_root: Union[str, Path] = ".",
    image_size: int = 512,
    num_workers: Optional[int] = None,
    skip_existing: bool = True,
) -> None:
    """
    Generates every (task, split) dataset in `task_splits` using a single
    pool of worker processes, instead of looping over tasks and splits one
    at a time. Each worker renders one sample directly to disk; the
    per-(task, split) `metadata.jsonl` files are written once, from the main
    process, after the whole pool has finished.

    `task_splits` is a mapping like {"task1": {"train": df, ...}, "task2": ...}.
    The same `splits` dict can be reused for every task, since a given puzzle
    row is independent across tasks.

    If `skip_existing` is True (the default), a (task, split) combination is
    skipped entirely when its output folder already has a `metadata.jsonl`
    with exactly as many records as `split_df` has rows — so re-running this
    after an interruption (a kernel restart, a network blip mid-upload, ...)
    doesn't re-render work that's already on disk.
    """
    output_root = Path(output_root)

    work_items: List[tuple] = []
    job_dirs: List[Path] = []  # parallel to work_items: output_dir for each item

    for task, splits in task_splits.items():
        for split_name, split_df in splits.items():
            output_dir = output_root / f"dataset_{task}" / split_name
            output_dir.mkdir(parents=True, exist_ok=True)

            stamp = _stamp_path(output_root, task, split_name)
            stamp_ok = stamp.exists() and stamp.read_text().strip() == RENDER_VERSION
            if skip_existing and stamp_ok and _existing_sample_count(output_dir) == len(split_df):
                print(
                    f"[{task}/{split_name}] already has {len(split_df)} sample(s) "
                    f"rendered with '{RENDER_VERSION}' — skipping."
                )
                continue
            if skip_existing and _existing_sample_count(output_dir) == len(split_df):
                print(
                    f"[{task}/{split_name}] has {len(split_df)} sample(s) from an older render "
                    f"version — re-rendering."
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
        print("Nothing to generate — every (task, split) combination is already up to date.")
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
