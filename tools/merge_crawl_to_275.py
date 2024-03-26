import pandas as pd


csv_275_path = "/home/manhckv/manhckv/ai4life/csv_info/275_train_info.tsv"

full_crawl_path = "/home/manhckv/manhckv/ai4life/csv_info/full_crawl.tsv"


# merge full crawl to 275 and remove duplicates
df_275 = pd.read_csv(csv_275_path, sep="\t", header=None)
df_275.columns = ["file_path", "label"]

df_full_crawl = pd.read_csv(full_crawl_path, sep="\t", header=None)
df_full_crawl.columns = ["file_path", "label"]

df_full_crawl = df_full_crawl[~df_full_crawl["file_path"].isin(df_275["file_path"])]
df_full_crawl = df_full_crawl.drop_duplicates(subset=["file_path"])

df_275 = pd.concat([df_275, df_full_crawl])

print(len(df_275))

df_275.to_csv(
    "275_full_crawl_info.tsv",
    sep="\t",
    index=False,
    header=False,
)
