"""
Patch Resolution Alignment (project spec, REOrder methodology section).

Goal: configure the rendered chessboard's resolution so that the ViT's
patch grid lines up exactly with the 8x8 chess-square grid -- ideally
1 vision-token per square, or at least an integer number of tokens per
square, so no single token straddles two different squares.

Unlike REOrder's own from-scratch models (external/REOrder/src/models/vit.py),
where `patch_size` is just a constructor argument you pick freely, Qwen2.5-VL
is a *pretrained* model: its vision tower's patch size is baked into already-
trained weights and can't be changed. The only lever available here is
choosing the rendered image's resolution so it's a clean multiple of Qwen's
actual patch pitch -- this module exists to compute and verify that.

IMPORTANT: the defaults below (patch_size=14, merge_size=2, i.e. an effective
28px-per-visual-token pitch) are the standard values for the Qwen2-VL /
Qwen2.5-VL family, but this has NOT been verified against whichever exact
checkpoint this project ends up using (the notebooks currently reference
"Qwen/Qwen3.5-0.8B", which is not a released checkpoint name as of writing --
confirm the real model id and pull its actual values via `get_vision_token_pitch`
below before trusting the defaults for anything more than a first draft).
"""

from typing import Optional

# Standard Qwen2-VL / Qwen2.5-VL vision tower values. Verify against the
# actual checkpoint in use (see `get_vision_token_pitch`) before relying on
# these for a real training run.
DEFAULT_VIT_PATCH_SIZE = 14
DEFAULT_MERGE_SIZE = 2

BOARD_SQUARES_PER_SIDE = 8


def get_vision_token_pitch(processor=None) -> int:
    """
    Returns the pixel size of one "vision token" -- i.e. how many pixels of
    the input image correspond to one patch embedding the LLM actually sees,
    after Qwen2-VL/2.5-VL's spatial merge (patch_size * merge_size).

    If `processor` is given (a Hugging Face `AutoProcessor` for a Qwen-VL
    checkpoint), reads `patch_size`/`merge_size` from its image processor
    config. Falls back to the family defaults (with a warning) if `processor`
    is None or doesn't expose those attributes -- e.g. before a model has
    been loaded, or for a quick estimate.
    """
    if processor is not None:
        image_processor = getattr(processor, "image_processor", None)
        patch_size = getattr(image_processor, "patch_size", None)
        merge_size = getattr(image_processor, "merge_size", None)
        if patch_size is not None and merge_size is not None:
            return int(patch_size) * int(merge_size)
        print(
            "[patch_geometry] processor did not expose patch_size/merge_size; "
            f"falling back to defaults ({DEFAULT_VIT_PATCH_SIZE}x{DEFAULT_MERGE_SIZE})."
        )

    print(
        "[patch_geometry] No processor given; using unverified Qwen2-VL/2.5-VL "
        f"family defaults (patch_size={DEFAULT_VIT_PATCH_SIZE}, merge_size={DEFAULT_MERGE_SIZE}). "
        "Pass the actual model's processor to confirm this before a real run."
    )
    return DEFAULT_VIT_PATCH_SIZE * DEFAULT_MERGE_SIZE


def aligned_image_size(
    tokens_per_square: int = 1,
    board_squares: int = BOARD_SQUARES_PER_SIDE,
    processor: Optional[object] = None,
) -> int:
    """
    Recommended square render resolution (pixels) so that each chess square
    maps to exactly `tokens_per_square` vision tokens on a side (so
    `tokens_per_square=1` means the ideal 1-patch-per-square alignment the
    spec calls for; `tokens_per_square=2` trades that off for more visual
    detail per square, e.g. if 1:1 renders pieces too small to read clearly).
    """
    pitch = get_vision_token_pitch(processor)
    return board_squares * tokens_per_square * pitch


def check_alignment(
    image_size: int,
    board_squares: int = BOARD_SQUARES_PER_SIDE,
    processor: Optional[object] = None,
) -> dict:
    """
    Diagnoses whether `image_size` (e.g. `configs/config.yaml`'s
    `dataset_generation.image_size`) actually aligns the ViT's patch grid
    with the 8x8 chess grid. Returns a dict rather than raising, so this can
    be used as a one-off sanity check in a notebook cell.
    """
    pitch = get_vision_token_pitch(processor)
    tokens_per_side = image_size / pitch
    square_px = image_size / board_squares

    is_aligned = (
        image_size % pitch == 0
        and (tokens_per_side / board_squares).is_integer()
    )

    result = {
        "image_size": image_size,
        "vision_token_pitch_px": pitch,
        "tokens_per_side": tokens_per_side,
        "pixels_per_square": square_px,
        "tokens_per_square": tokens_per_side / board_squares,
        "is_aligned": is_aligned,
    }

    if is_aligned:
        print(
            f"[patch_geometry] image_size={image_size} is aligned: "
            f"{result['tokens_per_square']:.0f} vision token(s) per square, "
            f"no square straddles a token boundary."
        )
    else:
        recommended = aligned_image_size(processor=processor, board_squares=board_squares)
        print(
            f"[patch_geometry] image_size={image_size} is NOT aligned with the "
            f"{pitch}px vision-token pitch ({result['tokens_per_square']:.2f} tokens/square, "
            f"non-integer). Consider image_size={recommended} instead."
        )

    return result
