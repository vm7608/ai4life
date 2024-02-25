# turn off all warnings
import warnings

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    pipeline,
)

from label_and_id import ID2LABEL, LABEL2ID


warnings.filterwarnings("ignore")

torch.set_num_threads(2)


def run_test(model_ckpt, test_root_path, test_tsv_path):
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

    test_data = pd.read_csv(test_tsv_path, sep="\t", header=None)
    columns = ["file_path", "label"]
    test_data.columns = columns

    ground_truth = []
    predictions = []
    # loop through the test data
    for row in tqdm(test_data.iterrows(), total=len(test_data)):
        _, (file_path, label) = row
        video_path = test_root_path + "/" + file_path
        # print(video_cls(video_path))
        ground_truth.append(label)
        prediction = video_cls(video_path)
        predictions.append(LABEL2ID[prediction[0]["label"]])

    # calculate the accuracy
    accuracy = accuracy_score(ground_truth, predictions)
    print(f"Accuracy: {accuracy}")
    # print the classification report
    print(classification_report(ground_truth, predictions))


if __name__ == "__main__":

    model_ckpt = "/home/manhckv/manhckv/ai4life/ai4life-personal-trainer"
    test_root_path = "/HDD1/manhckv/_manhckv/workoutfitness-video"
    test_tsv_path = "/home/manhckv/manhckv/ai4life/data/train_info.tsv"

    run_test(model_ckpt, test_root_path, test_tsv_path)
