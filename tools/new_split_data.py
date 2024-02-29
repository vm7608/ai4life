import pandas as pd


csv_path1 = "/home/manhckv/manhckv/ai4life/full-btc-al(6764).csv"
csv_path2 = "/home/manhckv/manhckv/ai4life/full-crawl-al(6764).csv"

train_df = pd.DataFrame(columns=["file_path", "label"])

data1 = pd.read_csv(csv_path1)

# add row that have score < 0.7 to train_df
for row in data1.iterrows():
    _, (file_path, label, score) = row
    if score < 0.8:
        train_df = pd.concat(
            [
                train_df,
                pd.DataFrame(
                    [["data-btc" + file_path, label]],
                    columns=["file_path", "label"],
                ),
            ]
        )


data2 = pd.read_csv(csv_path2)

# add row that have score < 0.7 to train_df
for row in data2.iterrows():
    _, (file_path, label, score) = row
    if score < 0.8:
        train_df = pd.concat(
            [
                train_df,
                pd.DataFrame(
                    [["data-crawl" + file_path, label]],
                    columns=["file_path", "label"],
                ),
            ]
        )


print(len(data1[data1["score"] < 0.8]))
print(len(data2[data2["score"] < 0.8]))
print(len(train_df))

train_df.to_csv("275_train_info.tsv", index=False, header=False, sep="\t")
