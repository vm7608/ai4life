import pathlib
import pandas
import numpy as np
import tqdm
import os
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

data_paths = ["/home/mkhoatd/repos/ai4life2/ai4life/data_csv_2/train_info.csv", "/home/mkhoatd/repos/ai4life2/ai4life/data_csv_2/val_info.csv"]
dataset_root_path = pathlib.Path("/home/mkhoatd/repos/ai4life2/ai4life/data_stable")
image_dataset_root_path = pathlib.Path("./data_stable_images")
done_lst_path = pathlib.Path("./done_lst.txt")
torchvision.set_video_backend("pyav")

if not image_dataset_root_path.exists():
    os.mkdir(image_dataset_root_path)

if os.path.exists(done_lst_path):
    with open(done_lst_path, "r") as f:
        done_lst = f.readlines()
        done_lst = [x.strip() for x in done_lst if not x.strip()==""]
else:
    done_lst = []

for data_path in data_paths:
    data = pandas.read_csv(data_path, sep="\t", header=None)
    # get the first column and convert to list
    data_list = data[0].tolist()
    print(len(data_list))
    for file_path in tqdm.tqdm(data_list):
        video_path = (dataset_root_path/file_path).resolve()
        if video_path in done_lst:
            continue
        class_name = video_path.parts[-2]
        file_name = video_path.parts[-1]
        file_name_without_ext = file_name.split(".")[0]
        image_dir = image_dataset_root_path/class_name
        if not image_dir.exists():
            os.mkdir(image_dir)
        try:
            vr = torchvision.io.VideoReader(str(video_path), "video")
            i = 0
            for frame in vr:
                frame = frame['data']
                frame = frame.permute(1, 2, 0)
                im = Image.fromarray(frame.numpy())
                image_name = f"{file_name_without_ext}_{i}.jpg"
                image_path = (image_dir/image_name).resolve()
                im.save(str(image_path))
                i += 1
            done_lst.append(str(video_path))
            with open(done_lst_path, "a") as f:
                f.write("\n".join(done_lst) + "\n") 
        except Exception as e:
            print(e)
            print(video_path)
            raise e

        # check if the file exists
        if not video_path.exists():
            print("Error: ", data_path, "->", file_path)
            continue
