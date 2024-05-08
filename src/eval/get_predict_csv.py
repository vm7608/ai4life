import os
import shutil
import uuid
import warnings

import ffmpeg
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
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


def crop_video(video_file, start_time, end_time):
    output_filename = os.path.join(CROPPED_DIR, f"{uuid.uuid4()}.mp4")
    output_filename = str(output_filename)

    # get the start and end time of input video, if start time is 0 and end time is the length of the video, then no need to crop
    video_length = int(float(ffmpeg.probe(video_file)["format"]["duration"]))
    if start_time == 0 and end_time == video_length:
        # copy the video to the output file
        shutil.copy(video_file, output_filename)
        return output_filename

    (ffmpeg.input(video_file, ss=start_time, to=end_time).output(output_filename).run())
    return output_filename


def infer(model_ckpt, video_path, crop_length=5):
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

    # get video duration
    video_length = int(float(ffmpeg.probe(video_path)["format"]["duration"]))

    # crop the video in to 10s consecutive clips
    list_of_clips = []
    for i in range(0, video_length, crop_length):
        start_time = i
        end_time = i + crop_length
        if end_time > video_length:
            end_time = video_length
        clip_path = crop_video(video_path, start_time, end_time)
        list_of_clips.append(clip_path)

    all_clip_predictions = {}
    for clip in list_of_clips:
        prediction = video_cls(clip)
        for pred in prediction:
            all_clip_predictions[pred["label"]] = (
                all_clip_predictions.get(pred["label"], 0) + pred["score"]
            )

    final_prediction = max(all_clip_predictions, key=all_clip_predictions.get)
    return final_prediction


if __name__ == "__main__":
    # CROPPED_DIR = "/HDD1/manhckv/_manhckv/ai4life-data/temp"
    # model_ckpt = "/home/manhckv/manhckv/ai4life/checkpoints/checkpoint-1242"
    # VIDEO_DIR = "/HDD1/manhckv/_manhckv/ai4life-data/test_btc_single"

    # all_results = []
    # for video_file in tqdm(os.listdir(VIDEO_DIR)):
    #     video_path = os.path.join(VIDEO_DIR, video_file)
    #     try:
    #         predict = infer(model_ckpt, video_path)
    #         all_results.append((video_file, predict))
    #     except Exception as e:
    #         print(f"Error processing {video_file}: {e}")
    #         all_results.append((video_file, "error"))
    #         continue

    # df = pd.DataFrame(all_results, columns=["video", "Dự đoán"])

    # df.to_csv("predict.csv", index=False)

    # Test accuracy and f1 score
    df = pd.read_csv("/home/manhckv/manhckv/ai4life/kikiki-predict.csv")
    gt_df = pd.read_csv("/home/manhckv/manhckv/ai4life/submit/gt_of_test2.csv")

    predictions = df["Dự đoán"].tolist()
    ground_truth = gt_df["label"].tolist()

    accuracy = accuracy_score(ground_truth, predictions)
    print(f"Accuracy: {accuracy}")
    f1 = f1_score(ground_truth, predictions, average="weighted")
    print(f"F1 score: {f1}")
