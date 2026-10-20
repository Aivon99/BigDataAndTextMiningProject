"""Turns the dataset rows into model inputs for training."""

import torch


def preprocess_function(sample, processor):
    """Builds the model inputs for one sample. The loss is computed only on the
    answer.
    """
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

    text = processor.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=False
    )

    batch = processor(
        text=[text],
        images=images_input,
        padding=False,
        return_tensors="pt"
    )

    result = {
        "input_ids": batch["input_ids"][0],
        "attention_mask": batch["attention_mask"][0],
        "pixel_values": batch["pixel_values"],
    }

    if "mm_token_type_ids" in batch:
        result["mm_token_type_ids"] = batch["mm_token_type_ids"][0]

    if "image_grid_thw" in batch:
        result["image_grid_thw"] = batch["image_grid_thw"]

    # The loss is computed only on the answer: the prompt is tokenized again on its own to find
    # where the answer starts, and every token before it gets the label -100 (ignored).
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
    """Pads the text tensors of a batch and concatenates the image tensors."""

    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id

    def __call__(self, examples):
        input_ids = [torch.tensor(ex["input_ids"]) if not isinstance(ex["input_ids"], torch.Tensor) else ex["input_ids"] for ex in examples]
        labels = [torch.tensor(ex["labels"]) if not isinstance(ex["labels"], torch.Tensor) else ex["labels"] for ex in examples]
        attention_mask = [torch.tensor(ex["attention_mask"]) if not isinstance(ex["attention_mask"], torch.Tensor) else ex["attention_mask"] for ex in examples]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        if "mm_token_type_ids" in examples[0] and examples[0]["mm_token_type_ids"] is not None:
            mm_token_type_ids = [torch.tensor(ex["mm_token_type_ids"]) if not isinstance(ex["mm_token_type_ids"], torch.Tensor) else ex["mm_token_type_ids"] for ex in examples]
            batch["mm_token_type_ids"] = torch.nn.utils.rnn.pad_sequence(mm_token_type_ids, batch_first=True, padding_value=0)

        if "pixel_values" in examples[0] and examples[0]["pixel_values"] is not None:
            pixel_values_list = [torch.tensor(ex["pixel_values"]) if not isinstance(ex["pixel_values"], torch.Tensor) else ex["pixel_values"] for ex in examples]
            batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)

        if "image_grid_thw" in examples[0] and examples[0]["image_grid_thw"] is not None:
            grid_list = [torch.tensor(ex["image_grid_thw"]) if not isinstance(ex["image_grid_thw"], torch.Tensor) else ex["image_grid_thw"] for ex in examples]
            batch["image_grid_thw"] = torch.cat(grid_list, dim=0)

        return batch


class Qwen35OnTheFlyCollator(Qwen35VisionDataCollator):
    """Preprocesses the raw rows batch by batch, so the processed images are
    never stored on disk.
    """

    def __call__(self, examples):
        processed = [preprocess_function(dict(ex), self.processor) for ex in examples]
        return super().__call__(processed)
