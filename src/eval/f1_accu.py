import os
import warnings

import ffmpeg
import torch
from eval_tools import ID2LABEL, LABEL2ID, crop_video, print_results, segment_video
from tqdm import tqdm
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    pipeline,
)


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
    print_results(ground_truth, predictions)


def run_test_segment(model_ckpt, test_root_dir):
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
                clip_path = crop_video(file_path, start_time, end_time, CROPPED_DIR)
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
    print_results(ground_truth, predictions)


def run_test_crop(model_ckpt, test_root_dir, crop_length=5):
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
                clip_path = crop_video(file_path, start_time, end_time, CROPPED_DIR)
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
    print_results(ground_truth, predictions)


if __name__ == "__main__":
    CROPPED_DIR = "/HDD1/manhckv/_manhckv/ai4life-data/temp"

    model_ckpts = [
        "/home/manhckv/manhckv/ai4life/checkpoints/checkpoint-1242",
        # ----------------------------------------------------------
        "/home/manhckv/manhckv/ai4life/checkpoints/checkpoint-3537",
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

        run_test_crop(model_ckpt, test_root_dir, crop_length=5)
        run_test_segment(model_ckpt, test_root_dir)
        print("*" * 50)
