import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import cairosvg
import chess
import chess.svg
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Standardized prompts according to project specifications (Section 3.2)
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
    """
    Renders a chess.Board state to an RGB PIL Image via SVG rasterization.
    """
    svg_data = chess.svg.board(board=board, size=size, lastmove=lastmove)
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
    """
    Builds a single multimodal sample for Task 1, Task 2, or Task 3.
    """
    if task not in PROMPTS:
        raise ValueError(f"Unsupported task '{task}'. Expected one of: {list(PROMPTS.keys())}")

    board = chess.Board(fen)
    first_move_uci = moves.split()[0] if moves and isinstance(moves, str) else None
    first_move = chess.Move.from_uci(first_move_uci) if first_move_uci else None

    images: List[Image.Image] = []
    saved_filenames: List[str] = []

    # ---------------------------------------------------------
    # Task 1: Image-to-FEN (Single clean board image)
    # ---------------------------------------------------------
    if task == "task1":
        img = render_board_svg(board=board, size=image_size)
        images.append(img)
        saved_filenames.append("board.png")
        target = fen

    # ---------------------------------------------------------
    # Task 2: Move Prediction (Single board image with last move highlighted)
    # ---------------------------------------------------------
    elif task == "task2":
        if first_move is None:
            raise ValueError(f"Task 2 requires a valid move sequence, none found for FEN: {fen}")

        # Compute SAN notation before applying move
        target = board.san(first_move)

        # Apply move to render updated board with highlighted last move
        board.push(first_move)
        img = render_board_svg(board=board, size=image_size, lastmove=first_move)

        images.append(img)
        saved_filenames.append("board.png")

    # ---------------------------------------------------------
    # Task 3: Dual-Image Delta Move (Consecutive frame pair [t, t+1])
    # ---------------------------------------------------------
    elif task == "task3":
        if first_move is None:
            raise ValueError(f"Task 3 requires a valid move sequence, none found for FEN: {fen}")
        target = board.san(first_move)

        # Render Frame t
        img_t = render_board_svg(board=board, size=image_size)

        # Apply move and render Frame t+1
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

    # Save to disk if output directory is provided
    if output_dir is not None:
        sample_dir = Path(output_dir) / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        for img, fname in zip(images, saved_filenames):
            img.save(sample_dir / fname)

        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return {"images": images, "metadata": metadata}


def generate_dataset(
    df: pd.DataFrame,
    task: str,
    output_dir: Union[str, Path],
    image_size: int = 512,
) -> None:
    """
    Generates a batch dataset for a specific task using dataframe records.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"Generating {task}")):
        sample_id = f"sample_{idx:06d}"
        build_sample(
            fen=row["FEN"],
            moves=row.get("Moves", None),
            task=task,
            sample_id=sample_id,
            puzzle_id=str(row.get("PuzzleId", sample_id)),
            output_dir=output_path,
            image_size=image_size,
        )
