import os
import pathlib

import wandb
from huggingface_hub import login
from transformers import (
    Trainer,
    TrainingArguments,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

from helper import collate_fn, compute_metrics, prepare_dataset
from label_and_id import ID2LABEL, LABEL2ID


os.environ["WANDB_PROJECT"] = "ckpt-6764-275"


def train_model(
    model_name,
    model,
    image_processor,
    train_dataset,
    val_dataset,
    batch_size=4,
    num_epochs=30,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    metric_for_best_model="loss",
):

    # Train and eval
    # args = TrainingArguments(
    #     model_name,
    #     remove_unused_columns=False,
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=learning_rate,
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=batch_size,
    #     warmup_ratio=warmup_ratio,
    #     logging_steps=10,
    #     load_best_model_at_end=True,
    #     metric_for_best_model=metric_for_best_model,
    #     push_to_hub=True,
    #     max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
    # )

    # Only train
    args = TrainingArguments(
        model_name,
        remove_unused_columns=False,
        evaluation_strategy="no",
        do_eval=False,
        save_strategy="epoch",
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        push_to_hub=False,
        max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    trainer.train()


def run_train(dataset_root_path, train_tsv_path, val_tsv_path):

    # ======== Preparing the model ========
    # MODEL_CKPT = "MCG-NJU/videomae-base" # base model/only pretrain -> 94.2M params
    # MODEL_CKPT = "MCG-NJU/videomae-large"  # large model/only pretrain -> 343M params
    # MODEL_CKPT = "MCG-NJU/videomae-base-finetuned-kinetics" # base model/finetuned on kinetics -> 86.5M params
    MODEL_CKPT = "MCG-NJU/videomae-large-finetuned-kinetics"  # large model/finetuned on kinetics -> 304M params

    image_processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_CKPT,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )

    # ======== Preparing the dataset ========
    train_dataset, val_dataset = prepare_dataset(
        dataset_root_path=dataset_root_path,
        train_tsv_path=train_tsv_path,
        val_tsv_path=val_tsv_path,
        image_processor=image_processor,
        model=model,
    )

    # ======== Training the model ========
    MODEL_NAME = "/HDD1/manhckv/_manhckv/ckpt/ai4life-pt"
    NUM_EPOCHS = 50
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-5
    WARMUP_RATIO = 0.1
    # METRIC_FOR_BEST_MODEL = "loss"
    METRIC_FOR_BEST_MODEL = "accuracy"
    train_model(
        model_name=MODEL_NAME,
        model=model,
        image_processor=image_processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
    )


if __name__ == "__main__":

    # ======== Logging in to Hugging Face Hub and Weights & Biases ========
    hf_token = "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"
    login(hf_token, add_to_git_credential=False)

    # ======== Logging in to Weights & Biases ========
    # Get key at https://wandb.ai/authorize
    wandb_key = "7cef42da986b9a35aabf18181bc73a867a875b8f"
    wandb.login(key=wandb_key)

    # ======== Preparing path ========

    dataset_root_path = pathlib.Path("/HDD1/manhckv/_manhckv")
    train_tsv_path = "/home/manhckv/manhckv/ai4life/data_csv/train.csv"
    val_tsv_path = "/home/manhckv/manhckv/ai4life/data_csv/val.csv"

    run_train(dataset_root_path, train_tsv_path, val_tsv_path)
