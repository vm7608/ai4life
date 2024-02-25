import evaluate
import numpy as np
import pytorchvideo.data
import torch
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)


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


def prepare_dataset(
    dataset_root_path,
    train_tsv_path,
    val_tsv_path,
    image_processor,
    model,
    sample_rate=4,
    fps=30,
):
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
        data_path=train_tsv_path,
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
        video_path_prefix=str(dataset_root_path),
    )

    val_dataset = pytorchvideo.data.labeled_video_dataset(
        data_path=val_tsv_path,
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
        video_path_prefix=str(dataset_root_path),
    )

    return train_dataset, val_dataset
