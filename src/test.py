import pathlib

import evaluate
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from helper import prepare_test_set
from label_and_id import ID2LABEL, LABEL2ID


torch.set_num_threads(2)


def run_test():
    dataset_root_path = pathlib.Path("/HDD1/manhckv/_manhckv/workoutfitness-video")

    model_ckpt = "/home/manhckv/manhckv/ai4life/ai4life-personal-trainer"
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )

    test_dataset = prepare_test_set(
        dataset_root_path=dataset_root_path,
        image_processor=image_processor,
        model=model,
    )

    # test model and report accuracy, precision, recall, f1-score, and confusion matrix
    evaluate.evaluate(model, test_dataset)


if __name__ == "__main__":
    run_test()
