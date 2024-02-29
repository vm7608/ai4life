import os
import shutil

import ffmpeg
from tqdm import tqdm


dataset_dir = "data_ai4life"
output_dir = "raw_video"
os.makedirs(output_dir, exist_ok=True)


for folder in tqdm(os.listdir(dataset_dir)):
    folder_path = os.path.join(dataset_dir, folder)

    output_folder = os.path.join(output_dir, folder)
    os.makedirs(output_folder, exist_ok=True)

    for idx, file in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file)

        # check if video extension is not mp4, then convert it to mp4
        if file_path.endswith(".mp4"):
            # copy the video to the output folder
            output_file = os.path.join(output_folder, "{}_{}.mp4".format(folder, idx))
            shutil.copy(file_path, output_file)

            #

        else:

            output_file = os.path.join(output_folder, "{}_{}.mp4".format(folder, idx))

            # convert the video to mp4
            stream = ffmpeg.input(file_path)
            stream = ffmpeg.output(stream, output_file)
            ffmpeg.run(stream)
