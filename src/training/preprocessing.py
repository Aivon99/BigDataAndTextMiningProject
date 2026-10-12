import torch

from training.reordering import reorder_chessboard_image


def preprocess_function(sample, processor):
    task = sample.get("task", "task1")
    prompt_text = sample["prompt"]
    target_text = sample["target"]

    if task in ["task1", "task2"]:
        board_image = sample["image"]
        chat_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": board_image},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target_text},
                ],
            }
        ]
        images_input = [board_image]

    elif task == "task3":
        img_t = sample.get("image") or sample.get("image_t")
        img_t1 = sample.get("image_t1")

        chat_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_t},
                    {"type": "image", "image": img_t1},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target_text},
                ],
            }
        ]
        images_input = [img_t, img_t1]

    else:
        raise ValueError(f"Unsupported task type: '{task}'")

    # Genera la stringa tramite il template nativo
    text = processor.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Processa testo e immagini insieme
    batch = processor(
        text=[text],
        images=images_input,
        padding=False,
        return_tensors="pt"
    )

    # Estrazione sicura dei tensori rimuovendo la dimensione del batch iniziale
    result = {
        "input_ids": batch["input_ids"][0],
        "attention_mask": batch["attention_mask"][0],
        "pixel_values": batch["pixel_values"],
    }

    if "mm_token_type_ids" in batch:
        result["mm_token_type_ids"] = batch["mm_token_type_ids"][0]

    if "image_grid_thw" in batch:
        result["image_grid_thw"] = batch["image_grid_thw"]

    # Labels for the Causal LM: loss must cover ONLY the assistant's reply,
    # not the prompt (instructions + images + question). Without this mask
    # -- the previous behaviour -- the model also gets gradient on copying
    # back its own prompt, which it already predicts almost perfectly under
    # teacher forcing; for short targets like SAN moves (Task 2/3, a handful
    # of tokens) this dilutes the useful signal in the reported average loss
    # to near-invisibility (prompt+images vastly outnumber the target in
    # token count), making the training loss hard to read and likely
    # slowing learning on the part that actually matters. The prompt/reply
    # boundary is found via the literal "<|im_start|>assistant\n" marker in
    # the same string already tokenized above -- not a fresh
    # apply_chat_template(..., add_generation_prompt=True) call, which for
    # Qwen3.5 in non-thinking mode would also insert a
    # "<think>\n\n</think>\n\n" block that isn't present in the real
    # sequence, throwing off the token count.
    assistant_marker = "<|im_start|>assistant\n"
    marker_pos = text.index(assistant_marker)
    prompt_only_text = text[: marker_pos + len(assistant_marker)]
    prompt_batch = processor(
        text=[prompt_only_text],
        images=images_input,
        padding=False,
        return_tensors="pt",
    )
    prompt_len = prompt_batch["input_ids"].shape[1]

    labels = result["input_ids"].clone()
    labels[:prompt_len] = -100
    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100
    result["labels"] = labels

    return result


class Qwen35VisionDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id

    def __call__(self, examples):
        # Convert lists saved by .map() back to PyTorch tensors
        input_ids = [torch.tensor(ex["input_ids"]) if not isinstance(ex["input_ids"], torch.Tensor) else ex["input_ids"] for ex in examples]
        labels = [torch.tensor(ex["labels"]) if not isinstance(ex["labels"], torch.Tensor) else ex["labels"] for ex in examples]
        attention_mask = [torch.tensor(ex["attention_mask"]) if not isinstance(ex["attention_mask"], torch.Tensor) else ex["attention_mask"] for ex in examples]
        
        # Pad text sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        
        # Handle mm_token_type_ids (pad with 0)
        if "mm_token_type_ids" in examples[0] and examples[0]["mm_token_type_ids"] is not None:
            mm_token_type_ids = [torch.tensor(ex["mm_token_type_ids"]) if not isinstance(ex["mm_token_type_ids"], torch.Tensor) else ex["mm_token_type_ids"] for ex in examples]
            batch["mm_token_type_ids"] = torch.nn.utils.rnn.pad_sequence(mm_token_type_ids, batch_first=True, padding_value=0)
        
        # Handle pixel_values
        if "pixel_values" in examples[0] and examples[0]["pixel_values"] is not None:
            pixel_values_list = [torch.tensor(ex["pixel_values"]) if not isinstance(ex["pixel_values"], torch.Tensor) else ex["pixel_values"] for ex in examples]
            batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)
            
        # Handle image_grid_thw (concatenation along dimension 0)
        if "image_grid_thw" in examples[0] and examples[0]["image_grid_thw"] is not None:
            grid_list = [torch.tensor(ex["image_grid_thw"]) if not isinstance(ex["image_grid_thw"], torch.Tensor) else ex["image_grid_thw"] for ex in examples]
            batch["image_grid_thw"] = torch.cat(grid_list, dim=0)

        return batch


class Qwen35OnTheFlyCollator(Qwen35VisionDataCollator):
    """
    Takes RAW dataset rows (PIL `image`/`image_t1`, `prompt`, `target`, `task`)
    and runs the image processor per batch, optionally reordering patches first.

    Avoids `dataset.map(preprocess_function)`, which materialises every image's
    `pixel_values` (~6 MB each at 512px, ~20 GB for a 3200-sample train split)
    into an Arrow cache and forces the collator to rebuild tensors from nested
    Python lists on every step. Use with `remove_unused_columns=False` and
    `dataloader_num_workers > 0` so preprocessing overlaps with GPU compute.
    """

    def __init__(self, processor, strategy=None, grid_size=8):
        super().__init__(processor)
        self.strategy = strategy
        self.grid_size = grid_size

    def __call__(self, examples):
        processed = []
        for ex in examples:
            ex = dict(ex)
            if self.strategy is not None:
                for key in ("image", "image_t1"):
                    if ex.get(key) is not None:
                        ex[key] = reorder_chessboard_image(
                            ex[key], strategy=self.strategy, grid_size=self.grid_size
                        )
            processed.append(preprocess_function(ex, self.processor))
        return super().__call__(processed)
