import math
import os
import shutil

import ffmpeg
import pandas as pd
from tqdm import tqdm


def crop_video(input_file, output_file, start_time, end_time):
    print(f"Cropping {input_file} to {output_file} from {start_time} to {end_time}")
    # if end_time = -1, then crop to the end of the video
    try:
        if end_time == -1:
            ffmpeg.input(input_file, ss=start_time).output(
                output_file, loglevel="quiet"
            ).run(capture_stdout=True, capture_stderr=True)
        else:
            ffmpeg.input(input_file, ss=start_time, to=end_time).output(
                output_file, loglevel="quiet"
            ).run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e


if __name__ == "__main__":

    raw_dir = "raw_video"
    output_dir = "processed_video"
    os.makedirs(output_dir, exist_ok=True)

    ## Process the label file
    label_file = "label.csv"
    label = pd.read_csv(label_file)

    # loop through each row in the label file
    for idx, row in tqdm(label.iterrows(), total=len(label)):
        folder = row["folder"]
        file_name = row["file_name"]

        # get the start and end time
        start_s = row["start_s"]
        end_s = row["end_s"]
        start_s2 = row["start_s2"]
        end_s2 = row["end_s2"]
        start_s3 = row["start_s3"]
        end_s3 = row["end_s3"]

        # check bad sample
        is_bad = int(row["bad_sample"])

        if is_bad == 1:
            print(f"Skipping {file_name} as it is a bad sample")
            continue

        if is_bad == 0:
            print(f"Copying {file_name}")
            # create the folder if it does not exist
            output_folder = os.path.join(output_dir, folder)
            os.makedirs(output_folder, exist_ok=True)

            # create the file path
            file_path = os.path.join(raw_dir, folder, file_name)

            # create the output file path
            output_file = os.path.join(output_folder, file_name.lower())

            # copy the file to the output folder
            print(f"Copy {file_path} to {output_file}")
            shutil.copy(file_path, output_file)

        if is_bad == 2:
            print(f"Cropping {file_path}")
            # create the folder if it does not exist
            output_folder = os.path.join(output_dir, folder)
            os.makedirs(output_folder, exist_ok=True)

            # create the file path
            file_path = os.path.join(raw_dir, folder, file_name)

            # create the output file path
            output_file = os.path.join(output_folder, file_name.lower())

            try:
                if (start_s and end_s) and (
                    not math.isnan(start_s) and not math.isnan(end_s)
                ):
                    crop_video(file_path, output_file, start_s, end_s)

                if (
                    start_s2
                    and end_s2
                    and (not math.isnan(start_s2) and not math.isnan(end_s2))
                ):
                    crop_video(
                        file_path,
                        output_file.replace(".mp4", "_2.mp4"),
                        start_s2,
                        end_s2,
                    )

                if (
                    start_s3
                    and end_s3
                    and (not math.isnan(start_s3) and not math.isnan(end_s3))
                ):
                    crop_video(
                        file_path,
                        output_file.replace(".mp4", "_3.mp4"),
                        start_s3,
                        end_s3,
                    )

            except Exception as e:
                print(f"\nError processing {file_path}: {e}")
