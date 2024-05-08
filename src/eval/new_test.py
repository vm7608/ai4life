# turn off all warnings
import math
import os
import shutil
import uuid
import warnings

import av
import ffmpeg
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor


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
    if video_length < chunk_size + max_overlap:
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


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def infer_one(model_ckpt, video_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    container = av.open(video_path)

    # sample 16 frames
    indices = sample_frame_indices(
        clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames
    )
    video = read_video_pyav(container, indices)

    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )

    model = model.to(device)

    inputs = image_processor(list(video), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_label = logits.argmax(-1).item()
    return predicted_label


def run_test_segment(model_ckpt, test_root_dir):
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

            all_clip_predictions = []
            for clip in list_of_clips:
                prediction = infer_one(model_ckpt, clip)
                all_clip_predictions.append(prediction)

            final_prediction = max(
                set(all_clip_predictions), key=all_clip_predictions.count
            )
            predictions.append(final_prediction)

    print(model_ckpt)
    print(test_root_dir)
    print(f"Accuracy: {accuracy_score(ground_truth, predictions)}")
    print(f"F1 macro: {f1_score(ground_truth, predictions, average='macro')}")
    print(f"F1 weighted: {f1_score(ground_truth, predictions, average='weighted')}")


def run_test(model_ckpt, test_root_dir):
    ground_truth = []
    predictions = []
    for folder in tqdm(os.listdir(test_root_dir), desc="Processing"):
        folder_path = os.path.join(test_root_dir, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            label_id = LABEL2ID[folder]
            ground_truth.append(label_id)
            try:
                prediction = infer_one(model_ckpt, file_path)
            except Exception as e:
                print(f"Error: {e}")
                print(f"File path: {file_path}")
                exit(0)

            predictions.append(prediction)

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
        # "/home/manhckv/manhckv/ai4life/checkpoints/checkpoint-9633",
        # --------------------------------------------------------------
        # "/home/manhckv/manhckv/ai4life/checkpoints/checkpoint-828",
    ]

    # path to the test data
    # test_root_dir = "/HDD1/manhckv/_manhckv/ai4life-data/test_btc"
    # test_root_dir = "/HDD1/manhckv/_manhckv/ai4life-data/data-btc"
    test_root_dir = "/HDD1/manhckv/_manhckv/ai4life-data/data-crawl"

    # test_root_dir = "/HDD1/manhckv/_manhckv/ai4life-data/data_btc_10s"
    # test_root_dir = "/HDD1/manhckv/_manhckv/ai4life-data/data_crawl_10s"

    # run the test
    for model_ckpt in model_ckpts:
        print("*" * 50)
        # run_test_segment(model_ckpt, test_root_dir)
        run_test(model_ckpt, test_root_dir)
        print("*" * 50)
