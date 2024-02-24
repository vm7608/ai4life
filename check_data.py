import pandas


data_paths = ["train_info.tsv", "val_info.tsv"]


for data_path in data_paths:
    data = pandas.read_csv(data_path, sep="\t", header=None)
    # get the first column and convert to list
    data_list = data[0].tolist()

    for file_path in data_list:
        folder_name = file_path.split("/")[0]

        file_name = file_path.split("/")[1]
        file_name = file_name.split("_")[0]

        # print(folder_name, file_name)
        if folder_name.lower() != file_name.lower():
            print("Error: ", data_path, "->", file_path)
