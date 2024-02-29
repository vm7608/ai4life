# turn off all warnings
import warnings

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    pipeline,
)

from label_and_id import ID2LABEL, LABEL2ID


warnings.filterwarnings("ignore")

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
        try:
            prediction = video_cls(video_path)
        except Exception as e:
            print(f"Error: {e}")
            print(f"File path: {video_path}")
            exit(0)
        predictions.append(LABEL2ID[prediction[0]["label"]])

    # calculate the accuracy
    accuracy = accuracy_score(ground_truth, predictions)
    print(f"Accuracy: {accuracy}")
    # print the classification report
    print(classification_report(ground_truth, predictions))


if __name__ == "__main__":

    model_ckpt = "/home/manhckv/manhckv/ai4life/checkpoint-6764"
    test_root_path = "/HDD1/manhckv/_manhckv/ai4life-data"
    test_tsv_path = "/home/manhckv/manhckv/ai4life/275_train_info.tsv"

    run_test(model_ckpt, test_root_path, test_tsv_path)
