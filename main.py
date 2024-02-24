import os
import pathlib

import evaluate
import kaggle
import numpy as np
import pytorchvideo.data
import torch
from huggingface_hub import login
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
from transformers import (
    Trainer,
    TrainingArguments,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    pipeline,
)

import wandb


torch.set_num_threads(2)


ID2LABEL = {
    0: 'barbell biceps curl',
    1: 'bench press',
    2: 'chest fly machine',
    3: 'deadlift',
    4: 'decline bench press',
    5: 'hammer curl',
    6: 'hip thrust',
    7: 'incline bench press',
    8: 'lat pulldown',
    9: 'lateral raise',
    10: 'leg extension',
    11: 'leg raises',
    12: 'plank',
    13: 'pull Up',
    14: 'push-up',
    15: 'romanian deadlift',
    16: 'russian twist',
    17: 'shoulder press',
    18: 'squat',
    19: 't bar row',
    20: 'tricep Pushdown',
    21: 'tricep dips',
}


LABEL2ID = {
    'barbell biceps curl': 0,
    'bench press': 1,
    'chest fly machine': 2,
    'deadlift': 3,
    'decline bench press': 4,
    'hammer curl': 5,
    'hip thrust': 6,
    'incline bench press': 7,
    'lat pulldown': 8,
    'lateral raise': 9,
    'leg extension': 10,
    'leg raises': 11,
    'plank': 12,
    'pull Up': 13,
    'push-up': 14,
    'romanian deadlift': 15,
    'russian twist': 16,
    'shoulder press': 17,
    'squat': 18,
    't bar row': 19,
    'tricep Pushdown': 20,
    'tricep dips': 21,
}


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions."""
    # the compute_metrics function takes a Named Tuple as input:
    # predictions, which are the logits of the model as Numpy arrays,
    # and label_ids, which are the ground-truth labels as Numpy arrays.
    metric = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    """The collation function to be used by `Trainer` to prepare data batches."""
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def prepare_dataset(dataset_root_path, image_processor, model, sample_rate=4, fps=30):
    mean = image_processor.image_mean
    std = image_processor.image_std
    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)

    num_frames_to_sample = model.config.num_frames

    clip_duration = num_frames_to_sample * sample_rate / fps
    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(resize_to),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )

    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
                    ]
                ),
            ),
        ]
    )

    train_dataset = pytorchvideo.data.labeled_video_dataset(
        data_path="./train_info.tsv",
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
        video_path_prefix=str(dataset_root_path),
    )

    val_dataset = pytorchvideo.data.labeled_video_dataset(
        data_path="./val_info.tsv",
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
        video_path_prefix=str(dataset_root_path),
    )

    return train_dataset, val_dataset


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


def inference(model, video):
    """Utility to run inference given a model and test video.

    The video is assumed to be preprocessed already.
    """
    # (num_frames, num_channels, height, width)
    perumuted_sample_test_video = video.permute(1, 0, 2, 3)

    inputs = {
        "pixel_values": perumuted_sample_test_video.unsqueeze(0),
        # "labels": torch.tensor(
        #     [sample_test_video["label"]]
        # ),  # this can be skipped if you don't have labels available.
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    class_label = model.config.id2label[predicted_class_idx]
    return class_label


def run_train():

    # ======== Logging in to Hugging Face Hub and Weights & Biases ========
    hf_token = "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"
    login(hf_token, add_to_git_credential=False)

    # ======== Logging in to Weights & Biases ========
    # Get key at https://wandb.ai/authorize
    wandb_key = "7cef42da986b9a35aabf18181bc73a867a875b8f"
    wandb.login(key=wandb_key)

    # ======== Downloading the dataset from Kaggle ========
    dataset_root_path = pathlib.Path("/HDD1/manhckv/_manhckv/workoutfitness-video")

    if not dataset_root_path.exists():
        print("Downloading dataset from Kaggle...")
        kaggle.api.dataset_download_files(
            "hasyimabdillah/workoutfitness-video",
            str(dataset_root_path),
            unzip=True,
        )

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
        image_processor=image_processor,
        model=model,
    )

    # ======== Training the model ========
    MODEL_NAME = "ai4life-personal-trainer"
    NUM_EPOCHS = 30
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


def run_test():
    pass


def run_inference():
    model_ckpt = "./resources/videomae-base-finetuned-workoutfitness-subset"
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )
    video_cls = pipeline(
        model=model, task='video-classification', feature_extractor=image_processor
    )
    print(video_cls("./resources/bench.mp4"))


if __name__ == "__main__":
    # run_train()
    run_inference()
