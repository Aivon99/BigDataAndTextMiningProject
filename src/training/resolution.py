"""Sets the image resolution the model sees (patch resolution alignment)."""

BOARD_SQUARES_PER_SIDE = 8


def configure_image_resolution(processor, image_size, board_squares=BOARD_SQUARES_PER_SIDE):
    """Makes the processor resize every image to image_size x image_size, so the
    512px dataset can be used at 256px without rendering it again.

    The LLM gets one token per 2x2 block of 16px patches (32px). At 256px this
    is an 8x8 grid: one token per chess square. 256 is also the processor's
    minimum size. Checks the result on a blank image and returns the grids.
    """
    from PIL import Image

    ip = processor.image_processor
    pixels = image_size * image_size
    size = dict(getattr(ip, "size", None) or {})
    size.update({"shortest_edge": min(size.get("shortest_edge", pixels), pixels), "longest_edge": pixels})
    ip.size = size
    for attr, value in (("max_pixels", pixels), ("min_pixels", size["shortest_edge"])):
        if hasattr(ip, attr):
            setattr(ip, attr, value)

    merge = int(getattr(ip, "merge_size", 2))
    grid = ip(images=[Image.new("RGB", (512, 512), "white")], return_tensors="pt")["image_grid_thw"][0].tolist()
    llm_grid = (grid[1] // merge, grid[2] // merge)
    if grid[1] * ip.patch_size != image_size or grid[2] * ip.patch_size != image_size:
        raise RuntimeError(
            f"Processor did not resize to {image_size}px (patch grid {grid}); this transformers "
            "version may configure the image size differently."
        )
    if llm_grid[0] % board_squares:
        raise RuntimeError(f"{image_size}px gives a {llm_grid} token grid, not a multiple of {board_squares}.")
    result = {
        "image_size": image_size,
        "patch_grid": tuple(grid[1:]),
        "llm_token_grid": llm_grid,
        "tokens_per_square": (llm_grid[0] // board_squares) ** 2,
    }
    print(f"[resolution] processor set to {image_size}px: {result}")
    return result
