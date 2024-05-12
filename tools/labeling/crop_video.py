# This script is used to crop the videos in the dataset to a specific time frame.

import shutil

import ffmpeg


def crop_video(video_file, output_file, start_time, end_time):

    video_length = int(float(ffmpeg.probe(video_file)["format"]["duration"]))
    if start_time == 0 and end_time == video_length:
        shutil.copy(video_file, output_file)
        return output_file

    (
        ffmpeg.input(video_file, ss=start_time, to=end_time)
        .output(output_file)
        .run(quiet=True)
    )
    return output_file


input_file = "raw_video/tricep Pushdown/tricep Pushdown_10.mp4"
output_file = "tricep Pushdown_10.mp4"

start_seconds = 0
end_seconds = 8

crop_video(input_file, output_file, start_seconds, end_seconds)
