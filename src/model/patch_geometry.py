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

VERIFIED against the real checkpoint ("Qwen/Qwen3.5-0.8B") via `tests.ipynb`:
- `processor.image_processor.patch_size = 16`, `merge_size = 2`.
- BUT the naive `patch_size * merge_size = 32px` formula for the vision-token
  pitch is WRONG: an empirical check (rendering a real 512px image and reading
  `image_grid_thw` off the actual processor output) measured a 32x32 grid --
  i.e. an effective pitch of 512/32 = 16px, not 32px. `image_grid_thw` reports
  the *pre-merge* patch grid; the 2x2 spatial merge happens later, inside the
  model, and isn't reflected in that tensor. So `patch_size` alone (16px) is
  the right pitch to reason about for *this* checkpoint -- but given the
  formula was already wrong once, don't trust arithmetic here in general.
  `measure_alignment_empirically()` below (what caught this) is the reliable
  method; `get_vision_token_pitch`/`check_alignment`'s arithmetic is now only
  a rough fallback for when no processor is loaded yet.

For the record: at image_size=512, board_squares=8, this checkpoint gives a
clean 4 vision-tokens-per-square (32x32 grid) -- 512 is aligned, confirmed by
`tests.ipynb`. No dataset regeneration needed.
"""

from typing import Optional

# Family defaults, used only as a rough guess when no processor is available
# yet. Verified WRONG for the actual pitch (see module docstring) -- prefer
# `measure_alignment_empirically()` whenever a processor is loaded.
DEFAULT_VIT_PATCH_SIZE = 16
DEFAULT_MERGE_SIZE = 2

BOARD_SQUARES_PER_SIDE = 8


def get_vision_token_pitch(processor=None) -> int:
    """
    Rough, ARITHMETIC estimate of the pixel pitch per vision token
    (patch_size * merge_size). Kept for a quick guess before a model is
    loaded, but this formula was empirically found to be wrong for the real
    checkpoint (see module docstring) -- prefer
    `measure_alignment_empirically()` whenever `processor` is available.
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
        "[patch_geometry] No processor given; using unverified arithmetic "
        f"defaults (patch_size={DEFAULT_VIT_PATCH_SIZE}, merge_size={DEFAULT_MERGE_SIZE}). "
        "This formula is known to be unreliable -- use measure_alignment_empirically() "
        "once a real processor is loaded."
    )
    return DEFAULT_VIT_PATCH_SIZE * DEFAULT_MERGE_SIZE


def aligned_image_size(
    tokens_per_square: int = 1,
    board_squares: int = BOARD_SQUARES_PER_SIDE,
    processor: Optional[object] = None,
) -> int:
    """
    ARITHMETIC estimate of a square render resolution (pixels) so each chess
    square maps to `tokens_per_square` vision tokens on a side. Same caveat
    as `get_vision_token_pitch`: this is a rough guess, not verified against
    a real processed image. Cross-check any recommendation from this with
    `measure_alignment_empirically()` before trusting it for a real run.
    """
    pitch = get_vision_token_pitch(processor)
    return board_squares * tokens_per_square * pitch


def check_alignment(
    image_size: int,
    board_squares: int = BOARD_SQUARES_PER_SIDE,
    processor: Optional[object] = None,
) -> dict:
    """
    ARITHMETIC diagnosis of whether `image_size` aligns the ViT's patch grid
    with the 8x8 chess grid, using the (unreliable -- see module docstring)
    `get_vision_token_pitch` formula. Kept for a quick guess with no images
    on hand; prefer `measure_alignment_empirically()` when possible.
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
            f"[patch_geometry] (arithmetic estimate) image_size={image_size} is aligned: "
            f"{result['tokens_per_square']:.0f} vision token(s) per square. "
            f"Cross-check with measure_alignment_empirically() before trusting this."
        )
    else:
        recommended = aligned_image_size(processor=processor, board_squares=board_squares)
        print(
            f"[patch_geometry] (arithmetic estimate) image_size={image_size} is NOT aligned "
            f"with the {pitch}px pitch ({result['tokens_per_square']:.2f} tokens/square, "
            f"non-integer). Consider image_size={recommended} -- but verify empirically first."
        )

    return result


def measure_alignment_empirically(
    processor,
    image,
    board_squares: int = BOARD_SQUARES_PER_SIDE,
) -> dict:
    """
    The reliable check: runs `image` (a PIL Image, e.g. from
    `data.generation.render_board_svg`) through the real `processor` and
    reads the actual `image_grid_thw` off the result, instead of trusting
    arithmetic. This is what caught `get_vision_token_pitch`'s formula being
    wrong for the real checkpoint (see module docstring) -- use this, not
    `check_alignment`, whenever a processor and a sample image are available.

    Mirrors the inline check in `tests.ipynb`'s "Test 3: Empirical
    patch-alignment check" -- promoted here so other notebooks/scripts don't
    have to hand-roll it.
    """
    inputs = processor(text=["<|image_pad|>"], images=[image], return_tensors="pt")
    grid_thw = inputs.get("image_grid_thw")

    if grid_thw is None:
        raise ValueError(
            "processor(...) output has no 'image_grid_thw' -- this checkpoint's "
            "processor may not expose per-image patch grid info the way "
            "Qwen2-VL/2.5-VL's does; can't verify alignment this way."
        )

    _, grid_h, grid_w = grid_thw[0].tolist()
    tokens_per_side = grid_h
    tokens_per_square = tokens_per_side / board_squares
    is_aligned = grid_h == grid_w and tokens_per_square.is_integer()

    result = {
        "image_size": image.size[0],
        "grid_h": grid_h,
        "grid_w": grid_w,
        "tokens_per_side": tokens_per_side,
        "tokens_per_square": tokens_per_square,
        "is_aligned": is_aligned,
    }

    if is_aligned:
        print(
            f"[patch_geometry] (empirical) image_size={result['image_size']} is aligned: "
            f"{tokens_per_square:.0f} vision token(s) per square (grid {grid_h}x{grid_w})."
        )
    else:
        print(
            f"[patch_geometry] (empirical) image_size={result['image_size']} is NOT aligned: "
            f"{tokens_per_square:.2f} tokens/square, non-integer (grid {grid_h}x{grid_w})."
        )

    return result
