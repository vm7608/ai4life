import pathlib
import pandas
import numpy as np
import tqdm
import os
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import logging

data_paths = [
    "/workspace/ai4life/src/train_07052024.tsv",
]
dataset_root_path = pathlib.Path("/workspace")
image_dataset_root_path = pathlib.Path("/workspace/img")
torchvision.set_video_backend("pyav")

if not image_dataset_root_path.exists():
    os.mkdir(image_dataset_root_path)

executor = ThreadPoolExecutor(max_workers=32)

done = 0
for data_path in data_paths:
    data = pandas.read_csv(data_path, sep="\t", header=None)
    # get the first column and convert to list
    data_list = data[0].tolist()
    print(len(data_list))
    for file_path in tqdm.tqdm(data_list):

        def run():
            global done
            video_path = (dataset_root_path / file_path).resolve()
            bb_name = video_path.parts[-3]
            class_name = video_path.parts[-2]
            file_name = video_path.parts[-1]
            file_name_without_ext = file_name.split(".")[0]
            image_dir = (
                image_dataset_root_path / bb_name / class_name / file_name_without_ext
            )
            if not image_dir.exists():
                image_dir.mkdir(parents=True, exist_ok=True)
                # os.mkdir(image_dir)
            try:
                vr = torchvision.io.VideoReader(str(video_path), "video")
                i = 0
                for frame in vr:
                    frame = frame["data"]
                    frame = frame.permute(1, 2, 0)
                    im = Image.fromarray(frame.numpy())
                    # vid_name = f"{file_name_without_ext}"

                    # image_path = (image_dir/vid_name/f"{i}.jpg").resolve()
                    image_path = image_dir / f"{i}.jpg"
                    im.save(str(image_path))
                    i += 1
            except Exception as e:
                logging.exception(e)
                raise e
            done += 1
            print(done/len(data_list))

        executor.submit(run)

executor.shutdown(wait=True, cancel_futures=False)
print(done)
