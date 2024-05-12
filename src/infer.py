import os
import warnings

import ffmpeg
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    pipeline,
)

from eval_tools import crop_video, segment_video
from label_and_id import ID2LABEL, LABEL2ID


warnings.filterwarnings("ignore")


def infer_single(model_ckpt, video_path):
    print("Run single...")

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
    prediction = video_cls(video_path)
    final_prediction = prediction[0]["label"]
    return final_prediction


def infer_crop(model_ckpt, video_path, crop_length=5):
    print(f"Run infer crop with {crop_length}s")

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

    list_of_clips = []
    for i in range(0, video_length, crop_length):
        start_time = i
        end_time = i + crop_length
        if end_time > video_length:
            end_time = video_length
        clip_path = crop_video(video_path, start_time, end_time, CROPPED_DIR)
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


def infer_segment(model_ckpt, video_path, chunk_size=5, overlap=1):
    print(f"Run infer segment with {chunk_size}s and overlap {overlap}s")

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

    video_length = int(float(ffmpeg.probe(video_path)["format"]["duration"]))
    video_segments = segment_video(video_length, chunk_size, overlap)

    list_of_clips = []
    for start_time, end_time in video_segments:
        clip_path = crop_video(video_path, start_time, end_time, CROPPED_DIR)
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
    CROPPED_DIR = "/HDD1/manhckv/_manhckv/ai4life-data/temp"
    model_ckpt = "/home/manhckv/manhckv/ai4life/checkpoints/checkpoint-6000"
    save_dir = "/home/manhckv/manhckv/ai4life/soutput"

    TEST_DIR = "/home/manhckv/manhckv/ai4life/Dữ liệu kiểm thử vòng chung kết"

    import time

    start_time = time.perf_counter()
    all_results = []
    for video_file in tqdm(os.listdir(TEST_DIR)):
        video_path = os.path.join(TEST_DIR, video_file)
        try:
            # predict = infer_single(model_ckpt, video_path)
            predict = infer_crop(model_ckpt, video_path, crop_length=5)
            # predict = infer_segment(model_ckpt, video_path, chunk_size=5, overlap=1)

            all_results.append((video_file, predict))
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            all_results.append((video_file, "error"))
            continue

    print(f"Total time taken: {time.perf_counter() - start_time:.2f}s")
    df = pd.DataFrame(all_results, columns=["video", "predict"])

    # file_name = "predict_single.csv"
    file_name = "predict_crop.csv"
    # file_name = "predict_segment.csv"

    df.to_csv(os.path.join(save_dir, file_name), index=False)
