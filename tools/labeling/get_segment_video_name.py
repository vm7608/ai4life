import os


dataset_dir = "new_processed_video"

for folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder)

    list_file = os.listdir(folder_path)

    for file in os.listdir(folder_path):
        if len(file.split("_")) > 2:
            print(folder + "/" + file)
