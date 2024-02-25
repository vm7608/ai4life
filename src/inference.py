from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    pipeline,
)

from label_and_id import ID2LABEL, LABEL2ID


def run_inference():
    model_ckpt = "/home/manhckv/manhckv/ai4life/ai4life-personal-trainer"
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
        device="cuda",
    )
    print(video_cls("/home/manhckv/manhckv/ai4life/deadlift_25.mp4"))


if __name__ == "__main__":
    run_inference()
