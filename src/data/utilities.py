"""Helpers to load the config file and the Lichess puzzle CSV."""

import yaml
import pandas as pd


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_lichess_csv(
    csv_path,
    max_samples=None,
    seed=42
):
    """Loads the Lichess CSV and takes a random sample of max_samples rows."""

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
    df,
    n_per_turn,
    turn_col="turn",
):
    """Lowers n_per_turn if there are not enough white or black puzzles."""
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
