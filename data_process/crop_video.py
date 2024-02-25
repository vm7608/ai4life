import ffmpeg


def crop_video(input_file, output_file, start_time, end_time):
    (ffmpeg.input(input_file, ss=start_time).output(output_file).run())
    print("Cropped video saved to", output_file)


input_file = "raw_video/tricep Pushdown/tricep Pushdown_10.mp4"

output_file = "tricep Pushdown_10.mp4"

start_seconds = 0
end_seconds = 8

crop_video(input_file, output_file, start_seconds, end_seconds)
