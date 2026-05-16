from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from .rendering import render_board
from .serialization import save_json
from .prompts import (
    FEN_PROMPT,
    ASCII_PROMPT,
    SAN_PROMPT
)


def build_sample(
    fen: str,
    sample_id: str,
    output_dir: Path = None,
    image_size: int = 512,
    task: str = "fen"
):
    """
    Build one multimodal sample.
    """

    image = render_board(
        fen=fen,
        size=image_size
    )

    if task == "fen":
        prompt = FEN_PROMPT
        target = fen

    elif task == "ascii":
        prompt = ASCII_PROMPT
        target = None

    else:
        prompt = ""
        target = ""

    metadata = {
        "sample_id": sample_id,
        "task": task,
        "fen": fen,
        "prompt": prompt,
        "target": target,
        "image_size": image_size,
        "patch_order": "row_major",
        "patch_size": 16
    }

    if output_dir is not None:

        sample_dir = output_dir / sample_id
        sample_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        image_path = sample_dir / "board.png"
        metadata_path = sample_dir / "metadata.json"

        image.save(image_path)

        save_json(
            metadata,
            metadata_path
        )

    return {
        "image": image,
        "metadata": metadata
    }

def generate_dataset(
    fens: Iterable[str],
    output_dir: str,
    image_size: int = 512,
    task: str = "fen"
):
    """
    Offline dataset generation.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for idx, fen in enumerate(tqdm(fens)):

        sample_id = f"sample_{idx:06d}"

        build_sample(
            fen=fen,
            sample_id=sample_id,
            output_dir=output_dir,
            image_size=image_size,
            task=task
        )


class ChessDatasetGenerator:

    def __init__(
        self,
        fens,
        image_size=512,
        task="fen"
    ):

        self.fens = fens
        self.image_size = image_size
        self.task = task

    def __iter__(self):

        for idx, fen in enumerate(self.fens):

            sample_id = f"sample_{idx:06d}"

            yield build_sample(
                fen=fen,
                sample_id=sample_id,
                image_size=self.image_size,
                task=self.task
            )

















