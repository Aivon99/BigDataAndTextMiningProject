"""Fine-tunes the model with LoRA for one task and one reordering strategy,
then pushes it to the Hub.
"""

from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from peft import get_peft_model

from training.checkpoints import (
    HubBestAdapterCallback,
    find_resumable_checkpoint,
    train_or_load_finished,
)
from training.preprocessing import Qwen35OnTheFlyCollator
from training.reordering import resolve_square_order
from training.token_reordering import install_token_reordering


def experiment_repo_id(task, strategy, tag=None, hf_org_prefix="bdatm-project"):
    """Hub repo of a LoRA run. The tag (e.g. "tok256") keeps the token-level runs
    apart from the legacy runs, which have no tag.
    """
    return f"{hf_org_prefix}/qwen-{task}-{strategy}{'-' + tag if tag else ''}-lora"


def experiment_model_name(label, tag=None):
    """Name of a row in the results table, e.g. "Qwen + LoRA (Zigzag - SFT) [tok256]"."""
    return f"{label} [{tag}]" if tag else label


def finetune_and_push_chessboard_model(
    strategy_name,
    dataset,
    processor,
    model,
    peft_config,
    task,
    hf_org_prefix="bdatm-project",
    tag=None,
    num_train_epochs=10,
    early_stopping_patience=2,
):
    """LoRA fine-tuning with a fixed scan order (any strategy of
    get_patch_reordering_indices, raster included).

    The order is applied to the visual tokens and the images are unchanged.
    The image size is the one set on the processor (configure_image_resolution).
    Returns the trained model with its reorderer set to the strategy's order.
    """
    if task not in ("task1", "task2", "task3"):
        raise ValueError(f"Unknown task: {task}")
    repo_id_target = experiment_repo_id(task, strategy_name, tag, hf_org_prefix)
    print("\n==============================================")
    print(f"TASK: {task.upper()} | STRATEGY: {strategy_name.upper()} -> {repo_id_target}")
    print("==============================================")

    install_token_reordering(model).set_order(resolve_square_order(strategy_name))
    data_collator = Qwen35OnTheFlyCollator(processor=processor)

    lora_model_instance = get_peft_model(model, peft_config)
    resume_checkpoint = find_resumable_checkpoint(repo_id_target)

    training_args_instance = TrainingArguments(
        output_dir=f"./temp_{task}_{strategy_name}_{tag or 'base'}_output",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=num_train_epochs,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        push_to_hub=True,
        hub_model_id=repo_id_target,
        hub_strategy="checkpoint",
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer_instance = Trainer(
        model=lora_model_instance,
        args=training_args_instance,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
            HubBestAdapterCallback(repo_id_target),
        ],
    )

    print(("Resuming from " + resume_checkpoint) if resume_checkpoint else "Training from scratch...")
    if not train_or_load_finished(trainer_instance, repo_id_target, resume_checkpoint):
        print(f"Nothing new to push for {repo_id_target}.")
        return trainer_instance.model

    print(f"Pushing model and processor to Hugging Face Hub: {repo_id_target}...")
    trainer_instance.model.push_to_hub(
        repo_id_target,
        commit_message=f"Training complete for {task}, {strategy_name} order",
    )
    processor.push_to_hub(repo_id_target)
    print(f"Finished! Successfully uploaded to Hub: {repo_id_target}")
    return trainer_instance.model
