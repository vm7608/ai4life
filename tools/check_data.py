import pathlib

import pandas
import tqdm
from decord import VideoReader, cpu


data_paths = ['/workspace/ai4life/csv_info/275_train_info.tsv',]


dataset_root_path = pathlib.Path('/workspace/ai4life-data')
for data_path in data_paths:
    data = pandas.read_csv(data_path, sep="\t", header=None)
    # get the first column and convert to list
    data_list = data[0].tolist()
    print(len(data_list))
    for file_path in tqdm.tqdm(data_list):
        video_path = (dataset_root_path / file_path).resolve()
        try:
            vr = VideoReader(str(video_path))
            for i in range(len(vr)):
                frame = vr[i]
        except Exception as e:
            print(e)
            print(video_path)
            raise e

        # check if the file exists
        if not video_path.exists():
            print("Error: ", data_path, "->", file_path)
            continue
