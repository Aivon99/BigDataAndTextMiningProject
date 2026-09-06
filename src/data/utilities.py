import os
import random
from pathlib import Path

from huggingface_hub import HfApi, login, whoami
from huggingface_hub.utils import LocalTokenNotFoundError
import numpy as np
import pandas as pd
import torch
from transformers import set_seed



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

def set_all_seeds(seed_value=42):
    """
    Set the seed for reproducibility across all libraries (Python, NumPy, PyTorch, Transformers).
    """
    # Python standard library
    random.seed(seed_value)
    
    # NumPy
    np.random.seed(seed_value)
    
    # PyTorch 
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Hugging Face Transformers
    set_seed(seed_value)

    # Python environment hash seed
    import os
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    
    print(f"All seeds successfully set to {seed_value} for full pipeline reproducibility.")
