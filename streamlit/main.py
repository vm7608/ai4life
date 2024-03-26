import datetime
import os
import shutil
import uuid
from pathlib import Path

import ffmpeg
import torch
from PIL import Image
from pytube import YouTube

import streamlit as st
from src.inference import run_inference
from src.label_and_id import ID2LABEL


torch.set_num_threads(2)
ROOT = Path("/HDD1/manhckv/_manhckv/temp")

UPLOAD_DIR = ROOT / "uploaded_video"
CROPPED_DIR = ROOT / "cropped_video"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CROPPED_DIR, exist_ok=True)

MODEL_CKPT = "/home/manhckv/manhckv/ai4life/checkpoints/checkpoint-1360"


def crop_video(video_file, start_time, end_time):
    output_filename = CROPPED_DIR / f"{uuid.uuid4()}.mp4"
    output_filename = str(output_filename)

    # get the start and end time of input video, if start time is 0 and end time is the length of the video, then no need to crop
    video_length = int(float(ffmpeg.probe(video_file)["format"]["duration"]))
    if start_time == 0 and end_time == video_length:
        # copy the video to the output file
        shutil.copy(video_file, output_filename)
        return output_filename

    (ffmpeg.input(video_file, ss=start_time, to=end_time).output(output_filename).run())
    return output_filename


def main():
    favicon = Image.open("/home/manhckv/manhckv/ai4life/favicon/favicon.ico")
    st.set_page_config(page_title="AI4LIFE2024", page_icon=favicon)
    st.title("AI4LIFE2024 - kikikiki")

    temp_downloaded_path = UPLOAD_DIR / f"{uuid.uuid4()}.mp4"

    if "download_btn" not in st.session_state:
        st.session_state["download_btn"] = False
    if "crop_btn" not in st.session_state:
        st.session_state["crop_btn"] = False
    if "predict_btn" not in st.session_state:
        st.session_state["predict_btn"] = False
    if "temp_downloaded_path" not in st.session_state:
        st.session_state["temp_downloaded_path"] = ""

    def reset_state():
        st.session_state["download_btn"] = False
        st.session_state["crop_btn"] = False
        st.session_state["predict_btn"] = False
        st.session_state["temp_downloaded_path"] = ""

    def reset_crop_state():
        st.session_state["crop_btn"] = False
        st.session_state["predict_btn"] = False

    # create a choice box for the user to select the input type as upload or youtube link
    input_type = st.radio(
        "Select input type:", ("Upload video", "Youtube link"), on_change=reset_state
    )

    if input_type == "Upload video":
        uploaded_video = st.file_uploader(
            "Upload a video (less than 10 minutes)", type=["mp4", "mov", "avi"]
        )
        if uploaded_video:
            bytes_data = uploaded_video.getvalue()
            with open(temp_downloaded_path, "wb") as file:
                file.write(bytes_data)
            video_length = int(
                float(ffmpeg.probe(temp_downloaded_path)["format"]["duration"])
            )
            hours = video_length // 3600
            minutes = (video_length % 3600) // 60
            seconds = video_length % 60
            if hours > 0 or minutes > 10:
                st.warning(
                    "Video length is greater than 10 minutes. Please upload a shorter video."
                )
                return

            st.session_state["download_btn"] = True
            st.session_state["temp_downloaded_path"] = str(temp_downloaded_path)

    if input_type == "Youtube link":
        youtube_link = st.text_input(
            "Enter the youtube link (video length < 10 minutes)"
        )

        if st.button("Download"):
            yt = YouTube(youtube_link)
            video_length = yt.length
            hours = video_length // 3600
            minutes = (video_length % 3600) // 60
            seconds = video_length % 60
            if hours > 0 or minutes > 10:
                st.warning(
                    "Video length is greater than 10 minutes. Please enter a shorter video URL."
                )
                return
            with st.spinner("Downloading video..."):
                yt = YouTube(youtube_link)
                yt.streams.filter(
                    progressive=True, file_extension="mp4"
                ).first().download(filename=temp_downloaded_path)

            st.session_state["download_btn"] = True
            st.session_state["temp_downloaded_path"] = str(temp_downloaded_path)

    if os.path.exists(st.session_state["temp_downloaded_path"]):
        video_length = int(
            float(
                ffmpeg.probe(st.session_state["temp_downloaded_path"])["format"][
                    "duration"
                ]
            )
        )
        hours = video_length // 3600
        minutes = (video_length % 3600) // 60
        seconds = video_length % 60
        st.video(str(st.session_state["temp_downloaded_path"]))
        st.write(f"Video length: {minutes} minutes {seconds} seconds")

        crop_time = st.slider(
            "Select the time range to crop the video",
            min_value=datetime.time(minute=0, second=0),
            max_value=datetime.time(minute=minutes, second=seconds),
            value=(
                datetime.time(minute=0, second=0),
                datetime.time(minute=minutes, second=seconds),
            ),
            step=datetime.timedelta(seconds=1),
            format="mm:ss",
            on_change=reset_crop_state,
        )

        if st.button("Crop Video"):
            st.session_state["crop_btn"] = True

            start_seconds = (
                crop_time[0].hour * 3600
                + crop_time[0].minute * 60
                + crop_time[0].second
            )
            end_seconds = (
                crop_time[1].hour * 3600
                + crop_time[1].minute * 60
                + crop_time[1].second
            )
            # Check if start time is greater than end time
            if start_seconds >= end_seconds:
                st.error("Start time cannot be greater than end time.")

            else:
                with st.spinner("Cropping video..."):
                    crop_video_path = crop_video(
                        st.session_state["temp_downloaded_path"],
                        start_seconds,
                        end_seconds,
                    )
                st.success("Video cropped successfully!")

                st.video(crop_video_path)
                st.session_state["crop_video_path"] = crop_video_path

        if st.session_state["crop_btn"]:
            if st.button("Predict"):
                with st.spinner("Predicting..."):
                    predict_class, score, predict = run_inference(
                        MODEL_CKPT, st.session_state["crop_video_path"], device="cpu"
                    )
                st.success(f"Predicted class: {ID2LABEL[predict_class]}")
                st.warning(f"Confidence score: {score:.2f}")
                st.write("Top 5 predictions:")
                for p in predict:
                    st.write(f"{p['label']}: {p['score']:.2f}")


if __name__ == "__main__":
    main()
