"""Loads the REOrder code (d3tk/REOrder, pinned commit): its Plackett-Luce
policy and its scan orders.

external/REOrder is cloned on first use if it is missing (e.g. on Colab).
REOrder's package is also called `src`, so its modules are imported under a
temporary `src` package and sys.modules is restored afterwards.
"""

import importlib
import subprocess
import sys
import types
from functools import lru_cache
from pathlib import Path

REORDER_URL = "https://github.com/d3tk/REOrder"
REORDER_COMMIT = "ddb290409c8bfbe49f7a5b1bcc19e2da0a94ded6"
REORDER_DIR = Path(__file__).resolve().parents[2] / "external" / "REOrder"


def ensure_reorder_checkout(reorder_dir=REORDER_DIR):
    """Clones REOrder at the pinned commit into external/REOrder, unless it is already there."""
    marker = reorder_dir / "src" / "models" / "layers" / "plackett_luce.py"
    if marker.exists():
        return reorder_dir
    reorder_dir.mkdir(parents=True, exist_ok=True)
    if any(reorder_dir.iterdir()):
        raise RuntimeError(f"{reorder_dir} exists but is not a REOrder checkout; remove it and retry.")
    print(f"[reorder] Fetching {REORDER_URL}@{REORDER_COMMIT[:7]} into {reorder_dir} ...")
    for cmd in (
        ["git", "init", "-q", str(reorder_dir)],
        ["git", "-C", str(reorder_dir), "fetch", "-q", "--depth", "1", REORDER_URL, REORDER_COMMIT],
        ["git", "-C", str(reorder_dir), "checkout", "-q", "FETCH_HEAD"],
    ):
        subprocess.run(cmd, check=True)
    return reorder_dir


@lru_cache(maxsize=1)
def load_reorder() -> types.SimpleNamespace:
    """
    Returns a namespace with REOrder's `PlackettLucePolicy`,
    `get_permute_indices` and the `hilbert` scan-order module.
    """
    root = ensure_reorder_checkout() / "src"
    saved = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    for k in saved:
        del sys.modules[k]
    try:
        for name, path in (("src", root), ("src.models", root / "models"),
                           ("src.models.layers", root / "models" / "layers"), ("src.utils", root / "utils")):
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(path)]
            sys.modules[name] = pkg
        layers = importlib.import_module("src.models.layers.layers")
        hilbert = importlib.import_module("src.models.layers.hilbert")
        plackett_luce = importlib.import_module("src.models.layers.plackett_luce")
    finally:
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    return types.SimpleNamespace(
        PlackettLucePolicy=plackett_luce.PlackettLucePolicy,
        get_permute_indices=layers.get_permute_indices,
        hilbert=hilbert,
    )


def reorder_scan_order(name, grid_size=8):
    """
    A REOrder scan order over a grid_size x grid_size grid, as a list of
    raster indices in visiting order (same convention as
    `get_patch_reordering_indices`). Names as in REOrder's configs:
    "hilbert-curve", "spiral-curve", "diagonal", "snake", "column-major".
    """
    order = load_reorder().get_permute_indices(order_needed=name, num_patches=grid_size * grid_size)
    if order is None:
        raise ValueError(f"REOrder has no fixed scan order called {name!r}")
    order = [int(i) for i in order]
    if sorted(order) != list(range(grid_size * grid_size)):
        raise ValueError(f"REOrder order {name!r} is not a permutation of the {grid_size}x{grid_size} grid")
    return order
