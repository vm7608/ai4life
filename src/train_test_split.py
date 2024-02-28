import os

import pandas as pd

from label_and_id import ID2LABEL, LABEL2ID


# create 2 pandas dataframe with 3 columns: file_path and label

train_df = pd.DataFrame(columns=["file_path", "label"])
test_df = pd.DataFrame(columns=["file_path", "label"])


root_dir = "/HDD1/manhckv/_manhckv/data-ai4life"
data_btc_dir = "data-btc"
data_crawl_dir = "data-crawl"
active_learning_csv = "/home/manhckv/manhckv/ai4life/active_learning.csv"

# add all file in data_btc_dir to train_df
for folder in os.listdir(os.path.join(root_dir, data_btc_dir)):
    folder_path = os.path.join(root_dir, data_btc_dir, folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        file_path = file_path.replace(root_dir, "")
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

# for each label in activative_learning.csv, sort by score and add 0.3 of highest score to test_df, 0.7 to train_df
active_learning_df = pd.read_csv(active_learning_csv)
active_learning_df = active_learning_df.sort_values(
    by=["label", "score"], ascending=[True, False]
)
for label in range(22):
    label_df = active_learning_df[active_learning_df["label"] == label]
    label_df_test = label_df.head(int(len(label_df) * 0.3))
    # check data that not in label_df_test and add to train_df
    label_df_train = label_df[~label_df.index.isin(label_df_test.index)]

    # append data_crawl_dir to the begin of file_path in label_df_test and label_df_train
    label_df_test["file_path"] = data_crawl_dir + label_df_test["file_path"]
    label_df_train["file_path"] = data_crawl_dir + label_df_train["file_path"]

    test_df = pd.concat([test_df, label_df_test[["file_path", "label"]]])
    train_df = pd.concat([train_df, label_df_train[["file_path", "label"]]])


# print length of train_df and test_df
print(len(train_df))
print(len(test_df))

# save to csv
train_df.to_csv("train_info.tsv", index=False, header=False, sep="\t")
test_df.to_csv("val_info.tsv", index=False, header=False, sep="\t")
