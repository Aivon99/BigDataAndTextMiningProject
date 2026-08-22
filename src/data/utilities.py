import pandas as pd
import os
from functools import partial
from pathlib import Path
from typing import Dict, Optional

from datasets import Dataset, Features, Value
from datasets import Image as HFImage
from huggingface_hub import HfApi, login, whoami
from huggingface_hub.utils import LocalTokenNotFoundError
from sklearn.model_selection import train_test_split

try:
    from .generation import iter_task_samples
except ImportError:
    from generation import iter_task_samples


def load_lichess_csv(
    csv_path: str,
    max_samples: int = None,
    seed: int = 42
):
    """
    Load Lichess puzzle CSV.
    """

    df = pd.read_csv(csv_path)

    if max_samples is not None:

        df = df.sample(
            n=max_samples,
            random_state=seed
        )

    return df


def balanced_turn_split(
    df: pd.DataFrame,
    n_per_turn: int,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Dict[str, pd.DataFrame]:
    """
    Balances puzzles equally between White-to-move and Black-to-move
    (n_per_turn each), then splits into train/validation/test, stratified
    by turn (default 80/10/10).
    """
    df = df.copy()
    df["turn"] = df["FEN"].apply(lambda f: f.split(" ")[1] if len(f.split(" ")) > 1 else "w")

    df_white = df[df["turn"] == "w"].sample(n=n_per_turn, random_state=seed)
    df_black = df[df["turn"] == "b"].sample(n=n_per_turn, random_state=seed)
    df_balanced = (
        pd.concat([df_white, df_black])
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    train_df, temp_df = train_test_split(
        df_balanced, test_size=(1 - train_frac), random_state=seed, stratify=df_balanced["turn"]
    )
    val_ratio_of_temp = val_frac / (1 - train_frac)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio_of_temp), random_state=seed, stratify=temp_df["turn"]
    )

    return {"train": train_df, "validation": val_df, "test": test_df}


TASK_FEATURES: Dict[str, Features] = {
    "task1": Features({
        "sample_id": Value("string"),
        "puzzle_id": Value("string"),
        "fen": Value("string"),
        "prompt": Value("string"),
        "target": Value("string"),
        "image": HFImage(),
    }),
    "task2": Features({
        "sample_id": Value("string"),
        "puzzle_id": Value("string"),
        "fen": Value("string"),
        "prompt": Value("string"),
        "target": Value("string"),
        "image": HFImage(),
    }),
    "task3": Features({
        "sample_id": Value("string"),
        "puzzle_id": Value("string"),
        "fen": Value("string"),
        "prompt": Value("string"),
        "target": Value("string"),
        "image_t": HFImage(),
        "image_t1": HFImage(),
    }),
}


def build_hf_dataset(
    df: pd.DataFrame,
    task: str,
    image_size: int = 512,
) -> Dataset:
    """
    Builds a HF `Dataset` for `task` by streaming samples from
    `iter_task_samples` through `Dataset.from_generator`, so images are
    rendered and encoded one at a time instead of all held in memory.
    """
    if task not in TASK_FEATURES:
        raise ValueError(f"Unsupported task '{task}'. Expected one of: {list(TASK_FEATURES.keys())}")

    generator_fn = partial(iter_task_samples, df=df, task=task, image_size=image_size)
    return Dataset.from_generator(generator_fn, features=TASK_FEATURES[task])


def push_task_dataset_to_hub(
    df: pd.DataFrame,
    task: str,
    dataset_name: str,
    split: str,
    namespace: Optional[str] = None,
    image_size: int = 512,
    private: bool = False,
) -> str:
    """
    Builds a task/split dataset in-memory (streamed, low footprint) and
    pushes it to the Hugging Face Hub as a Parquet-backed dataset, which
    supports `load_dataset(..., streaming=True)` on read.
    """
    _, active_namespace = authenticate_hf(target_namespace=namespace)
    repo_id = f"{active_namespace}/{dataset_name}"

    dataset = build_hf_dataset(df=df, task=task, image_size=image_size)
    dataset.push_to_hub(repo_id, split=split, private=private)

    print(f"Pushed {len(dataset)} '{split}' samples for {task} -> https://huggingface.co/datasets/{repo_id}")
    return repo_id


def authenticate_hf(target_namespace: str | None = None) -> tuple[HfApi, str]:
    """
    Handles authentication via local cache or terminal prompt if needed.
    
    Args:
        target_namespace: Specific organization or user namespace. If None,
                          defaults to the currently authenticated user.
                          
    Returns:
        tuple[HfApi, str]: Instantiated API client and active namespace.
    """
    try:
        user_info = whoami()
    except (LocalTokenNotFoundError, Exception):
        print("\n--- Hugging Face Authentication Required ---")
        login()
        user_info = whoami()

    username = user_info["name"]
    active_namespace = target_namespace if target_namespace else username
    
    print(f"Authenticated as: {username}")
    print(f"Target namespace: {active_namespace}")

    api = HfApi()
    return api, active_namespace


def upload_dataset_to_hub(
    local_dir: str | Path,
    dataset_name: str,
    namespace: str | None = None,
    private: bool = False,
    commit_message: str = "Upload dataset",
) -> str:
    """
    Uploads a local directory to a Hugging Face Dataset.
    Creates the dataset on Hugging Face if it does not already exist.

    Args:
        local_dir: Local path to the dataset folder.
        dataset_name: Name of the dataset (e.g., "chess-puzzle-task1").
        namespace: Organization or user account name. If None, uses current user.
        private: Whether the dataset should be private or public.
        commit_message: Commit message for the upload.

    Returns:
        str: The full dataset ID on Hugging Face (e.g., "org/chess-puzzle-task1").
    """
    api, active_namespace = authenticate_hf(target_namespace=namespace)
    
    local_path = Path(local_dir)
    if not local_path.exists() or not local_path.is_dir():
        raise FileNotFoundError(f"Local directory not found: {local_path.resolve()}")

    repo_id = f"{active_namespace}/{dataset_name}"
    print(f"\nTarget Dataset URL: https://huggingface.co/datasets/{repo_id}")

    # Creates the Dataset repo on Hugging Face if it doesn't exist yet
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,  # Does not raise error if dataset already exists
    )

    # Uploads the folder contents to the dataset
    print(f"Uploading files from '{local_path}'...")
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=".",
        commit_message=commit_message,
    )

    print(f"Dataset '{repo_id}' successfully synchronized!")
    return repo_id
