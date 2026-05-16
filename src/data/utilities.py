import pandas as pd


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