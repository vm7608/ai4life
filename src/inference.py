import torch
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    pipeline,
)

from label_and_id import ID2LABEL, LABEL2ID


def run_inference(model_ckpt, video_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )
    video_cls = pipeline(
        model=model,
        task='video-classification',
        feature_extractor=image_processor,
        device=device,
    )
    print(video_cls(video_path))


if __name__ == "__main__":
    model_ckpt = "/home/manhckv/manhckv/ai4life/ai4life-personal-trainer"
    video_path = "/home/manhckv/manhckv/ai4life/romanian deadlift_9.mp4"
    run_inference(model_ckpt, video_path)
