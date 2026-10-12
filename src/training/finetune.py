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

    # 1-2. Reordering and preprocessing happen on the fly inside the collator
    # (both frames for task3), so no reordered/tokenized copy of the dataset
    # is ever materialised on disk.

    # 3. Apply PEFT/LoRA to the provided model instance
    print("Applying LoRA to the provided model...")
    lora_model_instance = get_peft_model(model, peft_config)

    # 4. Resolve the target Hub repo now (needed before training starts) and
    # check whether an earlier, interrupted run already left a resumable
    # checkpoint there (e.g. after a Colab disconnect wiped the local runtime).
    repo_id_target = f"{hf_org_prefix}/qwen-{task}-{strategy_name}-lora"
    resume_checkpoint = find_resumable_checkpoint(repo_id_target)

    # 5. Configure Training Arguments for this specific run. Trains for up to
    # `num_train_epochs`, but relies on the validation set (via
    # EarlyStoppingCallback below) to stop once eval_loss stops improving,
    # and to keep the best-performing checkpoint rather than just the last one.
    # push_to_hub + hub_strategy="checkpoint" uploads a fully resumable
    # checkpoint (optimizer/scheduler/RNG state included) after every epoch,
    # so an interruption at epoch i loses at most that epoch's progress.
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

    # 6. Initialize Data Collator and Trainer
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

    # 7. Train (or resume) the model
    if resume_checkpoint:
        print(f"Resuming training for {task} ({strategy_name} reordering) from {resume_checkpoint}...")
    else:
        print(f"Training model for {task} with {strategy_name} reordering...")
    if not train_or_load_finished(trainer_instance, repo_id_target, resume_checkpoint):
        print(f"Nothing new to push for {repo_id_target}.")
        return trainer_instance.model

    # 8. Push final weights and processor directly to Hugging Face Hub
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
