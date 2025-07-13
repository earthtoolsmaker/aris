"""
This module provides functions for video processing tasks, including encoding videos
with the H.264 codec and extracting the average frame from a video file.

Functions:
- encode_video_with_h264_codec: Encodes a given video file using the H.264 codec.
- get_average_frame: Extracts the average frame from a video file, utilizing a specified
  number of frames or all frames if none are specified.
"""

import logging
from pathlib import Path

import cv2
import ffmpeg
import numpy as np
from numpy.typing import NDArray


def encode_video_with_h264_codec(filepath_input: Path, filepath_output: Path):
    """
    Encode the video with h264 codec.
    """

    assert filepath_input.exists(), "filepath_input does not exist!"

    logging.info(f"Encoding video from {filepath_input} to {filepath_output}")

    try:
        (
            ffmpeg.input(str(filepath_input))
            .output(str(filepath_output), vcodec="libx264", preset="medium")
            .run(capture_stdout=True, capture_stderr=True)
        )
        logging.info("Video encoded successfully.")

    except ffmpeg.Error as e:
        logging.error("An error occurred while encoding the video.")
        logging.error("Error message:", e.stderr.decode())


def get_average_frame(
    filepath_video: Path,
    max_frames: int | None,
) -> NDArray[np.uint8] | None:
    """
    Extract average frame from `filepath_video` using `max_frames` if
    specified, otherwise use all frames.

    Returns:
        array_image or None
    """
    logging.info(f"Opening Video {filepath_video}")
    cap = cv2.VideoCapture(str(filepath_video))
    number_frames_to_use = max_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(
        f"Extracting average frame from {filepath_video} using {number_frames_to_use} frames"
    )

    number_frames_processed = 0
    frame_sum = None

    while cap.isOpened():
        if number_frames_processed >= number_frames_to_use:
            break
        _, frame = cap.read()
        if frame is None:
            break

        frame_sum = (
            frame.astype(float)
            if frame_sum is None
            else (frame_sum + frame.astype(float))
        )
        number_frames_processed += 1

    logging.info(f"Closing video {filepath_video}")
    cap.release()

    if frame_sum is not None:
        average_frame = (frame_sum / number_frames_processed).astype("uint8")
        return average_frame
    else:
        logging.warning(f"Could not generate average frame from {filepath_video}")
        return None
