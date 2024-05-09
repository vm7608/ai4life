import math
import os
import shutil
import uuid
import warnings

import ffmpeg
from sklearn.metrics import accuracy_score, classification_report, f1_score


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


def segment_video(video_length, chunk_size=10, overlap=3):
    if video_length < chunk_size + overlap:
        return [(0, video_length)]

    num_chunks = math.ceil((video_length - overlap) / (chunk_size - overlap))
    chunk_start = 0
    segments = []

    for i in range(num_chunks):
        chunk_end = min(chunk_start + chunk_size, video_length)
        segments.append((chunk_start, chunk_end))
        chunk_start += chunk_size - overlap

    return segments


def crop_video(video_file, start_time, end_time, save_dir):
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


def print_results(ground_truth, predictions):
    print(f"GT: {ground_truth}")
    print(f"DT: {predictions}")
    print(f"Accuracy: {accuracy_score(ground_truth, predictions)}")
    print(f"F1 macro: {f1_score(ground_truth, predictions, average='macro')}")
    print(f"F1 weighted: {f1_score(ground_truth, predictions, average='weighted')}")
    print(
        classification_report(ground_truth, predictions, target_names=ID2LABEL.values())
    )
