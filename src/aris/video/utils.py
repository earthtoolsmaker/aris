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
from tqdm import tqdm


def encode_video_with_h264_codec(filepath_input: Path, filepath_output: Path):
    """
    Encode the input video file to the output file using the H.264 codec.

    Args:
        filepath_input (Path): The path to the input video file to be encoded.
        filepath_output (Path): The path where the encoded video file will be saved.

    Raises:
        AssertionError: If the input file does not exist.
        ffmpeg.Error: If an error occurs during the encoding process.
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
    max_frames: int | None = None,
) -> NDArray[np.uint8] | None:
    """
    Extract the average frame from the specified video file.

    This function opens the video file and computes the average of the frames 
    up to a specified maximum number of frames. If no maximum is specified, 
    all frames will be used to compute the average.

    Args:
        filepath_video (Path): The path to the video file from which the average frame will be extracted.
        max_frames (int | None, optional): The maximum number of frames to consider for averaging.
                                            If None, all frames are used.

    Returns:
        NDArray[np.uint8] | None: The average frame as a NumPy array, or None if no frames could be processed.

    Raises:
        Exception: If the video file cannot be opened or read.
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


def get_all_frames(filepath_video: Path) -> list[NDArray[np.uint8]]:
    """
    Extract all frames from the specified video file.

    Args:
        filepath_video (Path): The path to the video file from which frames will be extracted.

    Returns:
        list[NDArray[np.uint8]]: A list of frames represented as NumPy arrays, where each array corresponds to a frame in the video.
    """
    logging.info(f"Opening Video {filepath_video}")
    cap = cv2.VideoCapture(str(filepath_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Extracting all frames from {filepath_video}")

    frames = []

    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        while cap.isOpened():
            _, frame = cap.read()
            if frame is None:
                break
            frames.append(frame.astype("uint8"))
            pbar.update(1)

    cap.release()
    return frames


def get_fps(filepath_video: Path):
    """
    Retrieve the frames per second (FPS) of the specified video file.

    Args:
        filepath_video (Path): The path to the video file.

    Returns:
        float: The FPS of the video.

    Raises:
        Exception: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(str(filepath_video))
    if not cap.isOpened():
        raise Exception(f"Cannot open {filepath_video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def get_video_duration(filepath_video: Path) -> float | None:
    """
    Get the duration of the video in seconds.

    Args:
        filepath_video (Path): The path to the video file for which the duration is to be retrieved.

    Returns:
        float: The duration of the video in seconds.

    Raises:
        Exception: If the video file cannot be opened or if the duration cannot be retrieved.
    """
    try:
        probe = ffmpeg.probe(str(filepath_video))
        duration = float(probe["format"]["duration"])
        return duration
    except Exception as e:
        logging.error(f"Could not get video duration for {filepath_video}")
        return None


def save_frames_to_video(
    frames: list[NDArray[np.uint8]], filepath_save: Path, fps: int = 30
) -> None:
    """
    Save a list of frames as a video file using the specified frames per second (FPS).

    Args:
        frames (list[NDArray[np.uint8]]): A list of frames to be saved, where each frame is represented as a NumPy array.
        filepath_save (Path): The path where the video file will be saved.
        fps (int, optional): The frames per second for the output video. Default is 30.

    Raises:
        ValueError: If the list of frames is empty.
        Exception: If there is an error during video writing.

    Returns:
        None
    """
    if not frames:
        logging.error("No frames to save.")
        return

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(filepath_save), fourcc, fps, (width, height))

    with tqdm(total=len(frames), desc="Saving frames", unit="frame") as pbar:
        for frame in frames:
            video_writer.write(frame)
            pbar.update(1)

    video_writer.release()
    logging.info(f"Video saved to {filepath_save}")
