"""Token-level patch reordering for Qwen3.5 (the REOrder setting).

The image is never changed: the visual tokens are read by the LLM in another
order, each one keeping its true 2D position. Two methods of the inner
Qwen3_5Model are patched with the same permutation: get_image_features
reorders the token embeddings, get_rope_index reorders their M-RoPE positions.
The order matters because 18 of the 24 LLM layers are Gated DeltaNet, a
recurrent scan without positional encoding.

Orders are permutations of the 64 squares (0 = a8, 63 = h1). At 256px a square
is one LLM token; at 512px it is a 2x2 block of tokens, moved as a unit.

    reorderer = install_token_reordering(model)
    reorderer.set_order(get_patch_reordering_indices("zigzag"))
    with reorderer.order(order): ...
"""

from contextlib import contextmanager
from typing import Optional, Sequence

import torch


def _find_core_model(model):
    """The module that owns `visual`, `get_image_features` and `get_rope_index`
    (Qwen3_5Model), also when wrapped by PEFT / ForConditionalGeneration."""
    for module in model.modules():
        if all(hasattr(module, a) for a in ("visual", "get_image_features", "get_rope_index")):
            return module
    raise ValueError("No Qwen-VL core model (with visual/get_image_features/get_rope_index) found in `model`.")


class TokenReorderer:
    def __init__(self, model, grid_size: int = 8):
        self.core = _find_core_model(model)
        self.grid_size = grid_size
        self.merge = int(self.core.config.vision_config.spatial_merge_size)
        self._square_order: Optional[list] = None
        self._cache = {}
        self._orig_image_features = self.core.get_image_features
        self._orig_rope_index = self.core.get_rope_index
        self.core.get_image_features = self._image_features
        self.core.get_rope_index = self._rope_index
        self.core._token_reorderer = self

    def set_order(self, square_order: Optional[Sequence[int]]) -> None:
        """Square-level order (length grid_size**2), or None for the model's native raster order."""
        if square_order is None:
            self._square_order = None
            return
        order = [int(i) for i in (square_order.tolist() if torch.is_tensor(square_order) else square_order)]
        n = self.grid_size ** 2
        if sorted(order) != list(range(n)):
            raise ValueError(f"square_order must be a permutation of 0..{n - 1}")
        self._square_order = None if order == list(range(n)) else order
        self._cache.clear()

    @contextmanager
    def order(self, square_order: Optional[Sequence[int]]):
        previous = self._square_order
        self.set_order(square_order)
        try:
            yield self
        finally:
            self._square_order = previous
            self._cache.clear()

    def token_permutation(self, grid_h: int, grid_w: int, device=None) -> torch.Tensor:
        """
        Token-level permutation for one image whose merged token grid is
        grid_h x grid_w (raster order, as produced by the vision merger):
        each square's k x k token block in the square order, raster inside
        the block.
        """
        key = (grid_h, grid_w)
        if key not in self._cache:
            g = self.grid_size
            if grid_h != grid_w or grid_h % g:
                raise ValueError(
                    f"Token grid {grid_h}x{grid_w} is not a square multiple of the {g}x{g} board: "
                    "use an image size where the board maps to whole tokens (256 or 512 px)."
                )
            k = grid_h // g
            perm = []
            for s in self._square_order:
                r, c = divmod(s, g)
                for dr in range(k):
                    for dc in range(k):
                        perm.append((r * k + dr) * grid_w + (c * k + dc))
            self._cache[key] = torch.tensor(perm, dtype=torch.long)
        perm = self._cache[key]
        return perm.to(device) if device is not None else perm

    def _llm_grid(self, thw) -> tuple:
        t, h, w = (int(x) for x in thw)
        if t != 1:
            raise ValueError("Token reordering supports still images only (t == 1).")
        return h // self.merge, w // self.merge

    def _image_features(self, pixel_values, image_grid_thw=None, *args, **kwargs):
        out = self._orig_image_features(pixel_values, image_grid_thw, *args, **kwargs)
        if self._square_order is None:
            return out
        embeds = out.pooler_output if hasattr(out, "pooler_output") else out
        reordered = tuple(
            e[self.token_permutation(*self._llm_grid(thw), device=e.device)]
            for e, thw in zip(embeds, image_grid_thw)
        )
        if hasattr(out, "pooler_output"):
            out.pooler_output = reordered
            return out
        return reordered

    def _rope_index(self, input_ids, *args, **kwargs):
        position_ids, deltas = self._orig_rope_index(input_ids, *args, **kwargs)
        # get_rope_index(input_ids, mm_token_type_ids, image_grid_thw=None, ...)
        mm_token_type_ids = kwargs.get("mm_token_type_ids", args[0] if len(args) > 0 else None)
        image_grid_thw = kwargs.get("image_grid_thw", args[1] if len(args) > 1 else None)
        if self._square_order is None or image_grid_thw is None:
            return position_ids, deltas
        if mm_token_type_ids is None:
            raise ValueError("Token reordering needs `mm_token_type_ids` (returned by the processor).")
        grids = iter(image_grid_thw)
        for b in range(input_ids.shape[0]):
            slots = (mm_token_type_ids[b] == 1).nonzero(as_tuple=True)[0]
            start = 0
            while start < len(slots):
                gh, gw = self._llm_grid(next(grids))
                n = gh * gw
                idx = slots[start:start + n].to(position_ids.device)
                perm = self.token_permutation(gh, gw, device=position_ids.device)
                position_ids[:, b, idx] = position_ids[:, b, idx[perm]]
                start += n
        return position_ids, deltas

    def remove(self) -> None:
        """Restores the original methods."""
        self.core.get_image_features = self._orig_image_features
        self.core.get_rope_index = self._orig_rope_index
        del self.core._token_reorderer


def install_token_reordering(model, grid_size: int = 8) -> TokenReorderer:
    """Installs (once) and returns the TokenReorderer of `model`."""
    core = _find_core_model(model)
    existing = getattr(core, "_token_reorderer", None)
    return existing if existing is not None else TokenReorderer(model, grid_size=grid_size)


@torch.no_grad()
def verify_token_reordering(model, processor, image, square_order, prompt: str = "Describe the board.") -> dict:
    """
    Self-check to run once per environment (transformers versions differ):
    1. with no order set, outputs are identical to the unpatched model;
    2. with `square_order`, the LLM receives, at every image slot j, the
       embedding AND the M-RoPE position of token perm[j] (captured from the
       language model's inputs);
    3. the logits change (the DeltaNet layers do see the order).
    Raises AssertionError on failure, returns a small report otherwise.
    """
    reorderer = install_token_reordering(model)
    core = reorderer.core
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    captured = {}

    def hook(_module, args, kwargs):
        captured["embeds"] = kwargs["inputs_embeds"].detach().clone()
        captured["pos"] = kwargs["position_ids"].detach().clone()

    handle = core.language_model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with reorderer.order(None):
            base_logits = model(**inputs).logits[0, -1].float()
            base_embeds, base_pos = captured["embeds"], captured["pos"]
        reorderer.remove()
        unpatched_logits = model(**inputs).logits[0, -1].float()
        reorderer = install_token_reordering(model)
        with reorderer.order(square_order):
            perm_logits = model(**inputs).logits[0, -1].float()
            perm_embeds, perm_pos = captured["embeds"], captured["pos"]
    finally:
        handle.remove()

    assert torch.allclose(base_logits, unpatched_logits, atol=1e-4), "no-order output differs from the unpatched model"
    slots = (inputs["mm_token_type_ids"][0] == 1).nonzero(as_tuple=True)[0]
    gh, gw = reorderer._llm_grid(inputs["image_grid_thw"][0])
    with reorderer.order(square_order):
        perm = reorderer.token_permutation(gh, gw, device=slots.device)
    src = slots[perm]
    assert torch.equal(perm_embeds[0, slots], base_embeds[0, src]), "image embeddings not permuted consistently"
    assert torch.equal(perm_pos[:, 0, slots], base_pos[:, 0, src]), "M-RoPE positions not permuted with the embeddings"
    text_slots = (inputs["mm_token_type_ids"][0] != 1).nonzero(as_tuple=True)[0]
    assert torch.equal(perm_embeds[0, text_slots], base_embeds[0, text_slots]), "text tokens were modified"
    assert torch.equal(perm_pos[:, 0, text_slots], base_pos[:, 0, text_slots]), "text positions were modified"
    diff = (perm_logits - base_logits).abs().max().item()
    assert diff > 0, "logits identical under a different order: the reordering has no effect"
    report = {"image_tokens": int(len(slots)), "token_grid": (gh, gw), "max_logit_change": diff}
    print(f"[token reordering] OK: {report}")
    return report
