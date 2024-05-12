import math
import os
import shutil
import uuid

import ffmpeg
from tqdm import tqdm


def crop_and_save_video(video_file, save_dir, start_time, end_time):
    output_filename = os.path.join(save_dir, f"{uuid.uuid4()}.mp4")
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


def segment_video(video_length, chunk_size=10, max_overlap=3):
    if video_length <= chunk_size + max_overlap:
        return [(0, video_length)]

    num_chunks = math.ceil((video_length - max_overlap) / (chunk_size - max_overlap))
    chunk_start = 0
    segments = []

    for i in range(num_chunks):
        chunk_end = min(chunk_start + chunk_size, video_length)
        segments.append((chunk_start, chunk_end))
        chunk_start += chunk_size - max_overlap

    return segments


if __name__ == "__main__":

    # INPUT_DIR = "D:\\Project\\06_ai4life\\ai4life-workspace\\ai4life-data\\data-crawl"
    # OUTPUT_DIR = "D:\\Project\\06_ai4life\\ai4life-workspace\\data_crawl_10s"
    INPUT_DIR = "D:\\Project\\06_ai4life\\ai4life-workspace\\ai4life-data\\data-btc"
    OUTPUT_DIR = "D:\\Project\\06_ai4life\\ai4life-workspace\\data_btc_10s"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for folder in tqdm(os.listdir(INPUT_DIR), desc="Processing"):
        folder_path = os.path.join(INPUT_DIR, folder)
        output_folder = os.path.join(OUTPUT_DIR, folder)
        os.makedirs(output_folder, exist_ok=True)

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            video_length = int(float(ffmpeg.probe(file_path)["format"]["duration"]))
            video_segments = segment_video(video_length)
            for start_time, end_time in video_segments:
                crop_and_save_video(file_path, output_folder, start_time, end_time)
