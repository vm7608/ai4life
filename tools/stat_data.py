import os
from pprint import pprint

import ffmpeg
from tqdm import tqdm


# data_dir = "/HDD1/manhckv/_manhckv/ai4life-data/data-btc"
data_dir = "/HDD1/manhckv/_manhckv/ai4life-data/data_btc_10s"
# data_dir = "/HDD1/manhckv/_manhckv/ai4life-data/data_crawl_10s"
# data_dir = "/HDD1/manhckv/_manhckv/ai4life-data/data-crawl"


data_stats = []

# for folder in tqdm(os.listdir(data_dir), desc="Processing"):
#     folder_path = os.path.join(data_dir, folder)
#     stat = {}
#     stat["class"] = folder
#     stat["num_videos"] = len(os.listdir(folder_path))
#     stat["total_length"] = 0
#     for file in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, file)
#         video_length = int(float(ffmpeg.probe(file_path)["format"]["duration"]))
#         stat["total_length"] += video_length

#     data_stats.append(stat)

data_dir2 = "/HDD1/manhckv/_manhckv/ai4life-data/data_crawl_10s"
for folder in tqdm(os.listdir(data_dir), desc="Processing"):
    folder_path1 = os.path.join(data_dir, folder)
    folder_path2 = os.path.join(data_dir2, folder)
    stat = {}
    stat["class"] = folder
    stat["num_videos"] = len(os.listdir(folder_path1)) + len(os.listdir(folder_path2))
    stat["total_length"] = 0
    for file in os.listdir(folder_path1):
        file_path = os.path.join(folder_path1, file)
        video_length = int(float(ffmpeg.probe(file_path)["format"]["duration"]))
        stat["total_length"] += video_length

    for file in os.listdir(folder_path2):
        file_path = os.path.join(folder_path2, file)
        video_length = int(float(ffmpeg.probe(file_path)["format"]["duration"]))
        stat["total_length"] += video_length

    data_stats.append(stat)

print(f"Data root: {data_dir}")
pprint(data_stats)
print(f"Total classes: {len(data_stats)}")
print(f"Total videos: {sum([stat['num_videos'] for stat in data_stats])}")
print(f"Total length: {sum([stat['total_length'] for stat in data_stats])} seconds")

# create a pandas dataframe with 2 columns class and total videos
import pandas as pd


df = pd.DataFrame(data_stats)
# df = df[["class", "num_videos"]]
df.to_csv("data_stats.csv", index=False)
