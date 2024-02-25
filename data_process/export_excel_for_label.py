import os

import pandas as pd


dataset_dir = "raw_video"

output_excel = "data_ai4life.xlsx"

# create a dataframe
df = pd.DataFrame(
    columns=[
        "folder",
        "file_name",
        "bad_sample",
        "start_s",
        "end_s",
        "start_s2",
        "end_s2",
        "start_s3",
        "end_s3",
    ]
)
for folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        new_row = pd.DataFrame(
            {
                "folder": folder,
                "file_name": file,
                "bad_sample": "",
                "start_s": "",
                "end_s": "",
                "start_s2": "",
                "end_s2": "",
                "start_s3": "",
                "end_s3": "",
            },
            index=[0],
        )
        df = pd.concat([df, new_row], ignore_index=True)


df.to_excel(output_excel, index=False)
