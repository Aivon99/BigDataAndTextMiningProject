import numpy as np
from PIL import Image


def get_patch_reordering_indices(strategy="raster", grid_size=8):
    """
    Generates patch reordering index maps for an 8x8 chessboard grid.
    Strategies supported: 'raster', 'zigzag', 'spiral', 'file_wise', 'rank_wise'
    """
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

    elif strategy == "file_wise": # Column-wise
        return [int(x) for x in indices.T.flatten()]

    elif strategy == "rank_wise": # Row-wise (same as raster)
        return [int(x) for x in indices.flatten()]

    else:
        raise ValueError(f"Unknown reordering strategy: {strategy}")

def apply_patch_permutation(image, permutation, grid_size=8, img_size=512):
    """
    Slices `image` into a grid_size x grid_size grid of tiles and rearranges
    them according to `permutation` (a length grid_size**2 sequence of tile
    indices in raster order, e.g. from `get_patch_reordering_indices()` or
    sampled from a learned policy such as `PlackettLucePatchPolicy`).

    This is the low-level primitive both `reorder_chessboard_image` (fixed,
    named strategies) and the learned-reordering training loop
    (`src/training/learned_reordering.py`, a different permutation per step)
    build on.
    """
    image = image.resize((img_size, img_size))
    tile_size = img_size // grid_size

    # 1. Split image into individual square tiles
    tiles = []
    for r in range(grid_size):
        for c in range(grid_size):
            box = (c * tile_size, r * tile_size, (c + 1) * tile_size, (r + 1) * tile_size)
            tiles.append(image.crop(box))

    # 2. Rearrange tiles based on the given permutation
    reordered_tiles = [tiles[int(i)] for i in permutation]

    # 3. Stitch tiles back together into a new image
    new_image = Image.new("RGB", (img_size, img_size))
    for idx, tile in enumerate(reordered_tiles):
        r = idx // grid_size
        c = idx % grid_size
        new_image.paste(tile, (c * tile_size, r * tile_size))

    return new_image


def reorder_chessboard_image(image, strategy="raster", grid_size=8, img_size=512):
    """
    Slices a chessboard PIL Image into an 8x8 grid of tiles and rearranges
    them according to one of the fixed, named strategies from
    `get_patch_reordering_indices()`.
    """
    reorder_indices = get_patch_reordering_indices(strategy=strategy, grid_size=grid_size)
    return apply_patch_permutation(image, reorder_indices, grid_size=grid_size, img_size=img_size)
