# %%
import os
import pathlib

import kaggle


# %%
dataset_root_path = pathlib.Path("./workoutfitness-video/")

if not dataset_root_path.exists():
    print("Downloading dataset from Kaggle...")
    kaggle.api.dataset_download_files(
        "hasyimabdillah/workoutfitness-video",
        str(dataset_root_path),
        unzip=True,
    )


# %%
id2label = {
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

label2id = {
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

all_video_file_paths = list(dataset_root_path.glob("*/*"))

all_video_file_paths = [path for path in all_video_file_paths]

all_video_file_paths = [
    str(path.relative_to(dataset_root_path)) for path in all_video_file_paths
]

all_video_labels = [label2id[str(path).split("/")[0]] for path in all_video_file_paths]
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
