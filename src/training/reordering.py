"""Scan orders over the 64 board squares."""

import numpy as np
import torch

# Orders taken from REOrder (external/REOrder), by their name there.
_REORDER_ORDERS = {"hilbert": "hilbert-curve", "diagonal": "diagonal", "snake_diagonal": "snake"}


def get_patch_reordering_indices(strategy="raster", grid_size=8):
    """Returns the squares (raster index, 0 = a8, 63 = h1) in the order they are read.

    raster / rank_wise (the same order), file_wise, zigzag, spiral_center_out,
    spiral (outside-in), and REOrder's hilbert, diagonal and snake_diagonal.
    """
    if strategy in _REORDER_ORDERS:
        from training.reorder_lib import reorder_scan_order
        return reorder_scan_order(_REORDER_ORDERS[strategy], grid_size=grid_size)
    if strategy == "spiral_center_out":
        return get_patch_reordering_indices("spiral", grid_size)[::-1]

    total_patches = grid_size * grid_size
    indices = np.arange(total_patches).reshape(grid_size, grid_size)

    if strategy == "raster":
        return [int(x) for x in indices.flatten()]

    elif strategy == "zigzag":
        reordered = []
        for r in range(grid_size):
            row = indices[r, :]
            if r % 2 == 1:
                row = row[::-1]
            reordered.extend(row)
        return [int(x) for x in reordered]

    elif strategy == "spiral":
        reordered = []
        top, bottom, left, right = 0, grid_size - 1, 0, grid_size - 1
        while top <= bottom and left <= right:
            for c in range(left, right + 1):
                reordered.append(indices[top, c])
            top += 1
            for r in range(top, bottom + 1):
                reordered.append(indices[r, right])
            right -= 1
            if top <= bottom:
                for c in range(right, left - 1, -1):
                    reordered.append(indices[bottom, c])
                bottom -= 1
            if left <= right:
                for r in range(bottom, top - 1, -1):
                    reordered.append(indices[r, left])
                left += 1
        return [int(x) for x in reordered]

    elif strategy == "file_wise":
        return [int(x) for x in indices.T.flatten()]

    elif strategy == "rank_wise":
        return [int(x) for x in indices.flatten()]

    else:
        raise ValueError(f"Unknown reordering strategy: {strategy}")


def resolve_square_order(order, grid_size=8):
    """Strategy name or square permutation -> list of squares, or None for raster."""
    if order is None:
        return None
    if isinstance(order, str):
        order = get_patch_reordering_indices(order, grid_size=grid_size)
    order = [int(i) for i in (order.tolist() if torch.is_tensor(order) else order)]
    return None if order == list(range(grid_size * grid_size)) else order

