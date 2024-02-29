import pandas as pd


label = pd.read_csv("raw_label.csv")

# fill nan in the bad_sample column with 2
label["bad_sample"].fillna(2, inplace=True)

# fill nan in start_s, end_s, start_s2, end_s2, start_s3, end_s3 with -9999
label[["start_s", "end_s", "start_s2", "end_s2", "start_s3", "end_s3"]] = label[
    ["start_s", "end_s", "start_s2", "end_s2", "start_s3", "end_s3"]
].fillna(-9999)

# astype int
label[["start_s", "end_s", "start_s2", "end_s2", "start_s3", "end_s3"]] = label[
    ["start_s", "end_s", "start_s2", "end_s2", "start_s3", "end_s3"]
].astype(int)


# convert the bad_sample column to int
label["bad_sample"] = label["bad_sample"].astype(int)


# save the label to a new file
label.to_csv("label.csv", index=False)

label = pd.read_csv("label.csv")

# loop through bad_sample column
for idx, row in label.iterrows():
    if row["bad_sample"] not in [0, 1, 2]:
        print(f"Bad sample value {row['bad_sample']} at index {idx} is not valid")


# loop through start_s, end_s, start_s2, end_s2, start_s3, end_s3 columns and check if the values are valid

for idx, row in label.iterrows():
    # check type
    if (
        not isinstance(row["start_s"], int)
        or not isinstance(row["end_s"], int)
        or not isinstance(row["start_s2"], int)
        or not isinstance(row["end_s2"], int)
        or not isinstance(row["start_s3"], int)
        or not isinstance(row["end_s3"], int)
    ):
        print(f"Invalid type at index {idx}")

    if row["start_s"] > row["end_s"]:
        print(
            f"start_s {row['start_s']} is greater than end_s {row['end_s']} at index {idx}"
        )

    if row["start_s2"] > row["end_s2"]:
        print(
            f"start_s2 {row['start_s2']} is greater than end_s2 {row['end_s2']} at index {idx}"
        )

    if row["start_s3"] > row["end_s3"]:
        print(
            f"start_s3 {row['start_s3']} is greater than end_s3 {row['end_s3']} at index {idx}"
        )
