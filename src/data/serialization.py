import json
from pathlib import Path


def save_json(
    data: dict,
    path: Path
):
    """
    save dictionary as JSON.
    """

    with open(path, "w") as f:
        json.dump(
            data,
            f,
            indent=2
        )