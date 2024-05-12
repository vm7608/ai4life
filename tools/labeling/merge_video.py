list_path_to_merge = [
    (
        "decline bench press/decline bench press_3_2.mp4",
        "decline bench press/decline bench press_3.mp4",
    ),
    (
        "decline bench press/decline bench press_4_2.mp4",
        "decline bench press/decline bench press_4.mp4",
    ),
    ("hammer curl/hammer curl_6_2.mp4", "hammer curl/hammer curl_6.mp4"),
    ("hip thrust/hip thrust_1_2.mp4", "hip thrust/hip thrust_1.mp4"),
    ("lateral raise/lateral raise_0_2.mp4", "lateral raise/lateral raise_0.mp4"),
    ("leg raises/leg raises_3_2.mp4", "leg raises/leg raises_3.mp4"),
    ("leg raises/leg raises_5_2.mp4", "leg raises/leg raises_5.mp4"),
    ("plank/plank_0_2.mp4", "plank/plank_0.mp4"),
    ("plank/plank_3_2.mp4", "plank/plank_3.mp4"),
    ("pull Up/pull up_1_2.mp4", "pull Up/pull up_1.mp4"),
    ("russian twist/russian twist_4_2.mp4", "russian twist/russian twist_4.mp4"),
    ("shoulder press/shoulder press_2_2.mp4", "shoulder press/shoulder press_2.mp4"),
]

video_dir = "new_processed_video"

# ffmeg command to merge videos
import os
import subprocess


for video in list_path_to_merge:
    if not os.path.exists(video_dir + "/" + video[0]):
        print(f"{video[0]} does not exist")
    if not os.path.exists(video_dir + "/" + video[1]):
        print(f"{video[1]} does not exist")

    with open("concat.txt", "w") as f:
        f.write(f"file '{video_dir}/{video[0]}'\n")
        f.write(f"file '{video_dir}/{video[1]}'\n")

    output_name = video[1].split("/")[-1]

    # command = f"ffmpeg -f concat -safe 0 -i concat.txt -c copy '{output_name}'"
    command = f"ffmpeg -f concat -safe 0 -i concat.txt -c copy"

    # convert command to list
    command = command.split(" ")
    command.append(output_name)
    # print(command)

    subprocess.run(command, shell=True)
