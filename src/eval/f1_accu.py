# turn off all warnings
import math
import os
import shutil
import uuid
import warnings

import ffmpeg
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    pipeline,
)


warnings.filterwarnings("ignore")


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


def segment_video(video_length, chunk_size=10, max_overlap=5):
    # if video_length <= chunk_size + max_overlap:
    #     return [(0, video_length)]
    if video_length <= chunk_size:
        return [(0, video_length)]
    num_chunks = math.ceil((video_length - max_overlap) / (chunk_size - max_overlap))
    chunk_start = 0
    segments = []

    for i in range(num_chunks):
        chunk_end = min(chunk_start + chunk_size, video_length)
        segments.append((chunk_start, chunk_end))
        chunk_start += chunk_size - max_overlap

    return segments


def crop_video(video_file, start_time, end_time):
    output_filename = os.path.join(CROPPED_DIR, f"{uuid.uuid4()}.mp4")
    output_filename = str(output_filename)

    video_length = int(float(ffmpeg.probe(video_file)["format"]["duration"]))
    if start_time == 0 and end_time == video_length:
        shutil.copy(video_file, output_filename)
        return output_filename

    (
        ffmpeg.input(video_file, ss=start_time, to=end_time)
        .output(output_filename)
        .run(quiet=True)
    )
    return output_filename


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

    print(model_ckpt)
    print(test_root_dir)
    print(f"Accuracy: {accuracy_score(ground_truth, predictions)}")
    print(f"F1 macro: {f1_score(ground_truth, predictions, average='macro')}")
    print(f"F1 weighted: {f1_score(ground_truth, predictions, average='weighted')}")


def run_test_segment(model_ckpt, test_root_dir):
    # crop the video in to 10s consecutive clips
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
    for folder in tqdm(os.listdir(test_root_dir), desc="Processing"):
        folder_path = os.path.join(test_root_dir, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            video_length = int(float(ffmpeg.probe(file_path)["format"]["duration"]))
            video_segments = segment_video(video_length)

            label_id = LABEL2ID[folder]
            ground_truth.append(label_id)

            list_of_clips = []
            for start_time, end_time in video_segments:
                clip_path = crop_video(file_path, start_time, end_time)
                list_of_clips.append(clip_path)

            all_clip_predictions = {}
            for clip in list_of_clips:
                prediction = video_cls(clip)
                # if prediction[0]["score"] < 0.5:
                #     continue
                for pred in prediction:
                    all_clip_predictions[pred["label"]] = (
                        all_clip_predictions.get(pred["label"], 0) + pred["score"]
                    )

            final_prediction = max(all_clip_predictions, key=all_clip_predictions.get)
            predictions.append(LABEL2ID[final_prediction])

    print(model_ckpt)
    print(test_root_dir)
    print(f"Accuracy: {accuracy_score(ground_truth, predictions)}")
    print(f"F1 macro: {f1_score(ground_truth, predictions, average='macro')}")
    print(f"F1 weighted: {f1_score(ground_truth, predictions, average='weighted')}")


def run_test_crop(model_ckpt, test_root_dir, crop_length=5):
    # crop the video in to 10s consecutive clips
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
    for folder in tqdm(os.listdir(test_root_dir), desc="Processing"):
        folder_path = os.path.join(test_root_dir, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            video_length = int(float(ffmpeg.probe(file_path)["format"]["duration"]))

            label_id = LABEL2ID[folder]
            ground_truth.append(label_id)

            # crop the video in to small consecutive clips
            list_of_clips = []
            for i in range(0, video_length, crop_length):
                start_time = i
                end_time = i + crop_length
                if end_time > video_length:
                    end_time = video_length
                clip_path = crop_video(file_path, start_time, end_time)
                list_of_clips.append(clip_path)

            all_clip_predictions = {}
            for clip in list_of_clips:
                prediction = video_cls(clip)
                for pred in prediction:
                    all_clip_predictions[pred["label"]] = (
                        all_clip_predictions.get(pred["label"], 0) + pred["score"]
                    )

            final_prediction = max(all_clip_predictions, key=all_clip_predictions.get)
            predictions.append(LABEL2ID[final_prediction])

    print(model_ckpt)
    print(test_root_dir)
    print(f"Accuracy: {accuracy_score(ground_truth, predictions)}")
    print(f"F1 macro: {f1_score(ground_truth, predictions, average='macro')}")
    print(f"F1 weighted: {f1_score(ground_truth, predictions, average='weighted')}")


if __name__ == "__main__":
    CROPPED_DIR = "/HDD1/manhckv/_manhckv/ai4life-data/temp"

    model_ckpts = [
        # "/home/manhckv/manhckv/ai4life/checkpoints/ai4life-personal-trainer/checkpoint-1951",
        # "/home/manhckv/manhckv/ai4life/checkpoints/checkpoint-6764",
        "/home/manhckv/manhckv/ai4life/checkpoints/checkpoint-1242",
        # --------------------------------------------------------------
        "/home/manhckv/manhckv/ai4life/checkpoints/checkpoint-9633",
        # --------------------------------------------------------------
        # "/home/manhckv/manhckv/ai4life/checkpoints/checkpoint-828",
    ]

    # path to the test data
    test_root_dir = "/HDD1/manhckv/_manhckv/ai4life-data/test_btc"
    # test_root_dir = "/HDD1/manhckv/_manhckv/ai4life-data/data-btc"
    # test_root_dir = "/HDD1/manhckv/_manhckv/ai4life-data/data-crawl"

    # test_root_dir = "/HDD1/manhckv/_manhckv/ai4life-data/data_btc_10s"
    # test_root_dir = "/HDD1/manhckv/_manhckv/ai4life-data/data_crawl_10s"

    # run the test
    for model_ckpt in model_ckpts:
        print("*" * 50)
        # run_test(model_ckpt, test_root_dir)
        # run_test_crop(model_ckpt, test_root_dir, crop_length=10)
        run_test_segment(model_ckpt, test_root_dir)
        print("*" * 50)
