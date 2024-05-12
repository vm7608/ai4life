import os

import pandas as pd

from src.label_and_id import ID2LABEL, LABEL2ID


# create 2 pandas dataframe with 3 columns: file_path and label

train_df = pd.DataFrame(columns=["file_path", "label"])

root_dir = "/HDD1/manhckv/_manhckv/ai4life-data"
data_btc_dir = "data_btc_10s"
data_crawl_dir = "data_crawl_10s"

# add all file in data_btc_dir to train_df
for folder in os.listdir(os.path.join(root_dir, data_btc_dir)):
    folder_path = os.path.join(root_dir, data_btc_dir, folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        file_path = file_path.replace(root_dir + "/", "")
        label = LABEL2ID[folder]
        train_df = pd.concat(
            [
                train_df,
                pd.DataFrame(
                    [[file_path, label]],
                    columns=["file_path", "label"],
                ),
            ]
        )

# add all file in data_crawl_dir to train_df
for folder in os.listdir(os.path.join(root_dir, data_crawl_dir)):
    folder_path = os.path.join(root_dir, data_crawl_dir, folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        file_path = file_path.replace(root_dir + "/", "")
        label = LABEL2ID[folder]
        train_df = pd.concat(
            [
                train_df,
                pd.DataFrame(
                    [[file_path, label]],
                    columns=["file_path", "label"],
                ),
            ]
        )
# print length of train_df and test_df
print(len(train_df))

# save to csv
train_df.to_csv("train_07052024.tsv", index=False, header=False, sep="\t")
