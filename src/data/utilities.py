from pathlib import Path

import yaml
import pandas as pd


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    """
    Load the project's YAML config (dataset generation sizes, HF namespace, ...).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_lichess_csv(
    csv_path: str,
    max_samples: int = None,
    seed: int = 42
):
    """
    Load Lichess puzzle CSV. If `max_samples` exceeds the number of rows
    actually available, it's clamped down to that count (with a warning)
    instead of letting pandas raise on `.sample(n=...)`.
    """

    df = pd.read_csv(csv_path)

    if max_samples is not None:
        if max_samples > len(df):
            print(
                f"[load_lichess_csv] Requested max_samples={max_samples} exceeds "
                f"the {len(df)} rows available in '{csv_path}'; clamping to {len(df)}."
            )
            max_samples = len(df)

        df = df.sample(
            n=max_samples,
            random_state=seed
        )

    return df


def clamp_balanced_turn_counts(
    df: pd.DataFrame,
    n_per_turn: int,
    turn_col: str = "turn",
) -> int:
    """
    Clamp `n_per_turn` down to whatever is actually available for the
    scarcer of the two turns ('w'/'b') in `df`, so a subsequent
    `.sample(n=n_per_turn)` per turn never overflows the population.
    Prints a warning and returns the reduced value if clamping was needed;
    otherwise returns `n_per_turn` unchanged.
    """
    available_white = int((df[turn_col] == "w").sum())
    available_black = int((df[turn_col] == "b").sum())
    max_possible = min(available_white, available_black)

    if n_per_turn > max_possible:
        print(
            f"[clamp_balanced_turn_counts] Requested n_per_turn={n_per_turn} exceeds "
            f"what's available (white={available_white}, black={available_black}); "
            f"clamping to {max_possible}."
        )
        return max_possible

    return n_per_turn
