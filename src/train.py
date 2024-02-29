import pathlib

import torch
from huggingface_hub import login
from transformers import (
    Trainer,
    TrainingArguments,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

import wandb
from helper import collate_fn, compute_metrics, prepare_dataset
from label_and_id import ID2LABEL, LABEL2ID


torch.set_num_threads(2)


def train_model(
    model_name,
    model,
    image_processor,
    train_dataset,
    val_dataset,
    batch_size=4,
    num_epochs=30,
    learning_rate=5e-5,
):

    args = TrainingArguments(
        model_name,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True,
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

    # # ======== Downloading the dataset from Kaggle ========

    # if not dataset_root_path.exists():
    #     print("Downloading dataset from Kaggle...")
    #     kaggle.api.dataset_download_files(
    #         "hasyimabdillah/workoutfitness-video",
    #         str(dataset_root_path),
    #         unzip=True,
    #     )

    # ======== Preparing the model ========
    MODEL_CKPT = "MCG-NJU/videomae-base"

    image_processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_CKPT,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
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
    MODEL_NAME = "/HDD1/manhckv/_manhckv/ckpt/ai4life-personal-trainer"
    NUM_EPOCHS = 50
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-5
    train_model(
        model_name=MODEL_NAME,
        model=model,
        image_processor=image_processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
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
