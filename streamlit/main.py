import datetime
import os
import uuid
from pathlib import Path

import ffmpeg

import streamlit as st
from src.inference import run_inference


ROOT = Path("/HDD1/manhckv/_manhckv/temp")

UPLOAD_DIR = ROOT / "uploaded_video"
CROPPED_DIR = ROOT / "cropped_video"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CROPPED_DIR, exist_ok=True)

MODEL_CKPT = "/home/manhckv/manhckv/ai4life/ai4life-personal-trainer"


def crop_video(video_file, start_time, end_time):

    # create a random filename
    output_filename = CROPPED_DIR / f"{uuid.uuid4()}.mp4"

    # convert to string
    output_filename = str(output_filename)

    (ffmpeg.input(video_file, ss=start_time, to=end_time).output(output_filename).run())

    return output_filename


def main():
    st.title("Video Cropper")
    uploaded_video = st.file_uploader(
        "Upload a video (less than 1 hour)", type=["mp4", "mov", "avi"]
    )

    if uploaded_video:
        # Get video duration
        # Get uploaded video
        bytes_data = uploaded_video.getvalue()

        # Convert bytes to a file
        # create a temporary file to store the uploaded video
        temp_downloaded_video = UPLOAD_DIR / f"{uuid.uuid4()}.mp4"
        with open(temp_downloaded_video, "wb") as file:
            file.write(bytes_data)

        video_length = int(
            float(ffmpeg.probe(temp_downloaded_video)["format"]["duration"])
        )

        hours = video_length // 3600
        if hours > 0:
            st.warning(
                "Video length is greater than 1 hour. Please upload a shorter video."
            )
            return

        minutes = (video_length % 3600) // 60
        seconds = video_length % 60

        # display video
        st.video(uploaded_video)
        st.write(f"Video length: {minutes} minutes {seconds} seconds")

        # Get start and end time inputs
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
        )

        # Crop button
        if st.button("Crop Video"):

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

                # if st.button("Predict", key="predict"):
                #     st.write("Predicting...")
                #     # with st.spinner("Predicting..."):
                #     #     predict_class, score = run_inference(
                #     #         MODEL_CKPT, crop_video_path
                #     #     )
                #     # st.success(f"Predicted class: {predict_class}")
                #     # st.write(f"Confidence score: {score:.2f}")

                #     # https://discuss.streamlit.io/t/3-nested-buttons/30468 -> fix bằng cách này, không thì 1 nut crop chạy chung với predict luôn


if __name__ == "__main__":
    main()
