import datetime
import os
import uuid
from pathlib import Path

import ffmpeg
import torch
from PIL import Image

import streamlit as st
from src.inference import run_inference
from src.label_and_id import ID2LABEL


torch.set_num_threads(2)
ROOT = Path("/HDD1/manhckv/_manhckv/temp")

UPLOAD_DIR = ROOT / "uploaded_video"
CROPPED_DIR = ROOT / "cropped_video"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CROPPED_DIR, exist_ok=True)

MODEL_CKPT = "/home/manhckv/manhckv/ai4life/checkpoint-6764"


def crop_video(video_file, start_time, end_time):
    output_filename = CROPPED_DIR / f"{uuid.uuid4()}.mp4"
    output_filename = str(output_filename)
    (ffmpeg.input(video_file, ss=start_time, to=end_time).output(output_filename).run())
    return output_filename


def main():
    favicon = Image.open("/home/manhckv/manhckv/ai4life/favicon/favicon.ico")
    st.set_page_config(page_title="AI4LIFE2024", page_icon=favicon)
    st.title("AI4LIFE2024 - kikikiki")
    uploaded_video = st.file_uploader(
        "Upload a video (less than 10 minutes)", type=["mp4", "mov", "avi"]
    )

    if uploaded_video:
        # Get uploaded video
        bytes_data = uploaded_video.getvalue()
        temp_downloaded_video = UPLOAD_DIR / f"{uuid.uuid4()}.mp4"
        with open(temp_downloaded_video, "wb") as file:
            file.write(bytes_data)

        video_length = int(
            float(ffmpeg.probe(temp_downloaded_video)["format"]["duration"])
        )

        hours = video_length // 3600
        minutes = (video_length % 3600) // 60
        seconds = video_length % 60
        if hours > 0 or minutes > 10:
            st.warning(
                "Video length is greater than 10 minutes. Please upload a shorter video."
            )
            return

        # display video
        st.video(uploaded_video)
        st.write(f"Video length: {minutes} minutes {seconds} seconds")

        # Get start and end time inputs
        def set_false():
            st.session_state["crop_btn"] = False

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
            on_change=set_false,
        )

        # Crop button
        if "crop_btn" not in st.session_state:
            st.session_state["crop_btn"] = False

        if st.button("Crop Video"):
            st.session_state["crop_btn"] = not st.session_state["crop_btn"]

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
                        temp_downloaded_video, start_seconds, end_seconds
                    )
                st.success("Video cropped successfully!")
                st.video(crop_video_path)
                st.session_state["crop_video_path"] = crop_video_path

        if st.session_state["crop_btn"]:
            if st.button("Predict", key="predict"):
                with st.spinner("Predicting..."):
                    predict_class, score, predict = run_inference(
                        MODEL_CKPT, st.session_state["crop_video_path"], device="cpu"
                    )
                st.success(f"Predicted class: {ID2LABEL[predict_class]}")
                st.warning(f"Confidence score: {score:.2f}")
                for p in predict:
                    st.write(f"{ID2LABEL[p['label']]}: {p['score']:.2f}")


if __name__ == "__main__":
    main()
