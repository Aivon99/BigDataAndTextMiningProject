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


def finetune_and_push_chessboard_model(
    strategy_name,
    dataset,
    processor,
    model,
    peft_config,
    task,
    hf_org_prefix="bdatm-project",
    repo_root=None,
    num_train_epochs=10,
    early_stopping_patience=2,
):
    print(f"\n==============================================")
    print(
        f"Starting pipeline for TASK: {task.upper()} | STRATEGY:"
        f" {strategy_name.upper()}"
    )
    print(f"==============================================")

    if task not in ("task1", "task2", "task3"):
        raise ValueError(f"Unknown task: {task}")

    print("Applying LoRA to the provided model...")
    lora_model_instance = get_peft_model(model, peft_config)

    repo_id_target = f"{hf_org_prefix}/qwen-{task}-{strategy_name}-lora"
    resume_checkpoint = find_resumable_checkpoint(repo_id_target)

    training_args_instance = TrainingArguments(
        output_dir=f"./temp_{task}_{strategy_name}_output",
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

    data_collator = Qwen35OnTheFlyCollator(processor=processor, strategy=strategy_name)

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

    if resume_checkpoint:
        print(f"Resuming training for {task} ({strategy_name} reordering) from {resume_checkpoint}...")
    else:
        print(f"Training model for {task} with {strategy_name} reordering...")
    if not train_or_load_finished(trainer_instance, repo_id_target, resume_checkpoint):
        print(f"Nothing new to push for {repo_id_target}.")
        return trainer_instance.model

    print(
        f"Pushing model and processor to Hugging Face Hub: {repo_id_target}..."
    )

    trainer_instance.model.push_to_hub(
        repo_id_target,
        commit_message=(
            f"Training complete for {task} using {strategy_name} reordering"
            " strategy"
        ),
    )
    processor.push_to_hub(repo_id_target)

    print(f"Finished! Successfully uploaded to Hub: {repo_id_target}")

    return trainer_instance.model
