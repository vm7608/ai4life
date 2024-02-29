# turn off all warnings
import os
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
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


warnings.filterwarnings("ignore")


def run_test(model_ckpt, test_root_dir):
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

    ground_truth = []
    predictions = []
    # loop through the test data
    for folder in tqdm(os.listdir(test_root_dir), desc="Processing"):
        folder_path = os.path.join(test_root_dir, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            label_id = LABEL2ID[folder]
            ground_truth.append(label_id)

            try:
                prediction = video_cls(file_path)
            except Exception as e:
                print(f"Error: {e}")
                print(f"File path: {file_path}")
                exit(0)

            predictions.append(LABEL2ID[prediction[0]["label"]])

    # calculate the accuracy
    accuracy = accuracy_score(ground_truth, predictions)
    print(f"Accuracy: {accuracy}")
    f1 = f1_score(ground_truth, predictions, average="weighted")
    print(f"F1 score: {f1}")
    print(classification_report(ground_truth, predictions))

    # plot the confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    print("======== Confusion Matrix ========")
    print(cm)
    print("=================================")

    plt.figure(figsize=(25, 25))
    sns.set_theme(
        font_scale=2,
    )
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=ID2LABEL.values(),
        yticklabels=ID2LABEL.values(),
        cbar=False,
    )
    plt.title("Confusion Matrix of kikikiki")
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.savefig("confusion_matrix.png")


if __name__ == "__main__":
    # path to the model checkpoint
    model_ckpt = "/home/manhckv/manhckv/ai4life/checkpoints/checkpoint-6764"

    # path to the test data
    test_root_dir = "/HDD1/manhckv/_manhckv/ai4life-data/data-crawl"

    # run the test
    run_test(model_ckpt, test_root_dir)
