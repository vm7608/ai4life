import os

import pandas as pd
import torch
from tqdm import tqdm

from inference import run_inference
from label_and_id import ID2LABEL, LABEL2ID


torch.set_num_threads(4)

MODEL_CKPT = "/home/manhckv/manhckv/ai4life/checkpoint-6764"

data_crawl_root = "/HDD1/manhckv/_manhckv/data-ai4life/data-crawl"

# create a pandas dataframe with 3 columns: file_path, label_id and predict
df = pd.DataFrame(columns=["file_path", "label_id", "score", "prediction"])


for folder in tqdm(os.listdir(data_crawl_root), desc="Processing"):
    folder_path = os.path.join(data_crawl_root, folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        label_id = LABEL2ID[folder]

        _, _, prediction = run_inference(MODEL_CKPT, file_path)

        # get score of label in prediction
        score = 0
        for p in prediction:
            if p["label"] == ID2LABEL[label_id]:
                score = p["score"]
                break

        file_path = file_path.replace(data_crawl_root, "")
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [[file_path, label_id, score, prediction]],
                    columns=["file_path", "label_id", "score", "prediction"],
                ),
            ]
        )


# save to csv
df.to_csv("active_learning.csv", index=False)
