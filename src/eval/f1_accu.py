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
    """Infer không cắt video"""
    print("Run test without cropping")
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


def run_test_segment(model_ckpt, test_root_dir, chunk_size, overlap):
    """Infer chia video thành các đoạn nhỏ nối tiếp overlap nhau 3s"""
    print(f"Run test with segmenting with {chunk_size}s and overlap {overlap}s")
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
            video_segments = segment_video(video_length, chunk_size, overlap)

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
    """Infer chia video thành các đoạn nhỏ nối tiếp nhau"""
    print(f"Run test with cropping with {crop_length}s")
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
    os.makedirs(CROPPED_DIR, exist_ok=True)

    model_ckpts = [
        # "/home/manhckv/manhckv/ai4life/checkpoints/checkpoint-1242",
        # "/home/manhckv/manhckv/ai4life/checkpoints/checkpoint-3537",
        # "/home/manhckv/manhckv/ai4life/checkpoints/checkpoint-5000",
        # "/home/manhckv/manhckv/ai4life/checkpoints/checkpoint-5500",
        "/home/manhckv/manhckv/ai4life/checkpoints/checkpoint-6000",
    ]

    # path to the test data
    test_dirs = [
        # "/HDD1/manhckv/_manhckv/ai4life-data/test_btc",
        # "/HDD1/manhckv/_manhckv/ai4life-data/data_btc_10s",
        # "/HDD1/manhckv/_manhckv/ai4life-data/data_crawl_10s",
        # "/HDD1/manhckv/_manhckv/ai4life-data/data-btc",
        "/HDD1/manhckv/_manhckv/ai4life-data/data-crawl",
    ]

    # run the test
    for model_ckpt in model_ckpts:
        for test_dir in test_dirs:
            print("*" * 50)
            # run_test(model_ckpt, test_dir)
            # run_test_crop(model_ckpt, test_dir, crop_length=5)
            # run_test_segment(model_ckpt, test_dir, chunk_size=5, overlap=1)

            # run_test_crop(model_ckpt, test_dir, crop_length=10)
            run_test_segment(model_ckpt, test_dir, chunk_size=10, overlap=3)
            print("*" * 50)
        print("=" * 50)
