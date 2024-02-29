import os
import pathlib

from src.label_and_id import LABEL2ID


dataset_root_path = pathlib.Path("/HDD1/manhckv/_manhckv/data-ai4life/data-btc")

all_video_file_paths = list(dataset_root_path.glob("*/*"))

all_video_file_paths = [path for path in all_video_file_paths]

all_video_file_paths = [
    str(path.relative_to(dataset_root_path)) for path in all_video_file_paths
]

all_video_labels = [LABEL2ID[str(path).split("/")[0]] for path in all_video_file_paths]
all_path_label = list(zip(all_video_file_paths, all_video_labels))
all_path_label[:5]

import pandas as pd


info_data = pd.DataFrame(all_path_label, columns=["path", "label"])
info_data.info()

train_info_data = info_data.groupby("label").sample(frac=0.8, random_state=1)
val_info_data = info_data.loc[~info_data.index.isin(train_info_data.index)]

train_info_data.to_csv("./train_info.tsv", index=False, header=False, sep="\t")
val_info_data.to_csv("./val_info.tsv", index=False, header=False, sep="\t")
val_info_data.head()
