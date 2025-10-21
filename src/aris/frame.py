"""
Frame extraction and video generation for ARIS sonar data.

This module provides functions for extracting individual frames from ARIS sonar files
and generating MP4 videos from frame sequences. It handles the conversion between
ARIS binary data and standard video formats, including grayscale to RGB conversion,
timestamp extraction, and parallel frame processing for improved performance.

Key Features:
    - Extract frames from ARIS files as NumPy arrays
    - Parallel frame processing using ThreadPoolExecutor
    - Convert frame sequences to MP4 videos (mp4v codec)
    - Grayscale to RGB conversion for video encoding
    - UTC timestamp extraction from ARIS frames

Dependencies:
    - pyARIS: For ARIS file format parsing and frame reading
    - OpenCV (cv2): For video writing and frame processing
    - NumPy: For efficient array operations

Typical Usage:
    # Extract frames from ARIS file
    aris_data, first_frame = pyARIS.DataImport("sonar_data.aris")
    frames = extract_frames_as_numpy_arrays(
        aris_data=aris_data,
        start_frame=0,
        end_frame=100
    )

    # Generate video from frames
    aris_frames_to_mp4v_video(
        aris_frames=frames,
        filepath_save=Path("output.mp4"),
        fps=24
    )

Related Modules:
    - aris.pyARIS.pyARIS: Core ARIS file format parser
    - aris.video.utils: Video processing utilities (encoding, frame extraction)
"""

import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pytz
from numpy.typing import NDArray
from tqdm import tqdm

from .pyARIS import pyARIS


def get_recorded_at_datetime(aris_frame: pyARIS.ARIS_Frame) -> datetime.datetime:
    """
    Extract the datetime when the ARIS frame was recorded.

    Converts the sonar timestamp from microseconds to a timezone-aware
    datetime object in UTC.

    Args:
        aris_frame (pyARIS.ARIS_Frame): ARIS frame object containing
            sonar timestamp metadata.

    Returns:
        datetime.datetime: Timezone-aware datetime in UTC representing
            when the frame was recorded.
    """
    return datetime.datetime.fromtimestamp(
        aris_frame.sonartimestamp / 1000000, pytz.timezone("UTC")
    )


def grayscale_to_rgb(aris_frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Convert an ARIS grayscale frame into RGB by stacking the channels.

    Creates an RGB image by replicating the grayscale channel three times.
    This is necessary for video encoding which typically requires RGB input.

    Args:
        aris_frame (NDArray[np.uint8]): Grayscale ARIS frame with shape (H, W).

    Returns:
        NDArray[np.uint8]: RGB frame with shape (H, W, 3) where all three
            channels contain identical grayscale values.
    """
    return np.stack((aris_frame,) * 3, axis=-1)


def is_grayscale(aris_frame: NDArray[np.uint8]) -> bool:
    """
    Check whether the ARIS frame is grayscale or RGB.

    Determines frame type by checking the number of dimensions:
    grayscale frames have 2 dimensions (H, W), while RGB frames
    have 3 dimensions (H, W, C).

    Args:
        aris_frame (NDArray[np.uint8]): ARIS frame to check.

    Returns:
        bool: True if the frame is grayscale (2D array), False if RGB (3D array).
    """
    return len(aris_frame.shape) == 2


def aris_frames_to_mp4v_video(
    aris_frames: list[NDArray[np.uint8]],
    filepath_save: Path,
    fps: int = 24,
) -> None:
    """
    Generate an MP4 video from a sequence of ARIS frames.

    Encodes a list of ARIS frames into an MP4 video file using the mp4v codec.
    Automatically converts grayscale frames to RGB for video encoding. Creates
    parent directories if they don't exist. Progress is displayed via tqdm.

    Args:
        aris_frames (list[NDArray[np.uint8]]): List of ARIS frames to encode.
            Frames can be either grayscale (H, W) or RGB (H, W, 3).
        filepath_save (Path): Path where the output MP4 video will be saved.
            Parent directories will be created if they don't exist.
        fps (int, optional): Frames per second for the output video. Defaults to 24.
            Should match the original ARIS frame rate for accurate playback.

    Returns:
        None

    Note:
        - All frames must have the same dimensions (height and width)
        - Grayscale frames are automatically converted to RGB
        - Uses OpenCV's mp4v codec (FourCC: 'mp4v')
        - Video writer and windows are properly released after encoding
    """
    height = aris_frames[0].shape[0]
    width = aris_frames[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    filepath_save.parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(str(filepath_save), fourcc, fps, (width, height))
    for i in tqdm(range(len(aris_frames)), desc="Writing frames"):
        frame = aris_frames[i]
        if is_grayscale(frame):
            rgb_frame = grayscale_to_rgb(frame)
            video.write(rgb_frame)
        else:
            video.write(frame)
    video.release()
    cv2.destroyAllWindows()


def extract_frame_as_numpy_array(
    aris_data: pyARIS.ARIS_File, idx: int
) -> NDArray[np.uint8] | None:
    """
    Extract a single frame as a NumPy array from ARIS file data.

    Reads the specified frame from the ARIS data object and returns the
    remapped sonar image as a NumPy array. Remapping converts the raw
    beam/bin sonar data into 2D spatial coordinates for visualization.

    Args:
        aris_data (pyARIS.ARIS_File): ARIS file data object containing
            the sonar frames and metadata.
        idx (int): Zero-indexed frame number to extract.

    Returns:
        NDArray[np.uint8] | None: Remapped sonar frame as a 2D NumPy array,
            or None if extraction fails due to an error.

    Note:
        - Errors during extraction are logged but don't raise exceptions
        - Returns None on failure to support parallel processing pipelines
        - The 'remap' attribute contains the spatially remapped sonar image
    """
    try:
        aris_frame = pyARIS.FrameRead(aris_data, idx)
        return aris_frame.remap
    except Exception as e:
        logging.error(f"Could not extract frame {idx}, {e}")
        return None  # Return None if there's an error


def extract_frames_as_numpy_arrays(
    aris_data: pyARIS.ARIS_File,
    start_frame: int,
    end_frame: int,
    skip_frame: int = 0,
) -> list[NDArray[np.uint8]]:
    """
    Extract multiple frames as NumPy arrays from ARIS file data using parallel processing.

    Extracts a range of frames from the ARIS data object using ThreadPoolExecutor
    for improved performance. Frames are processed in parallel and progress is
    tracked with a tqdm progress bar. Failed frame extractions are logged but
    do not halt the overall process.

    Args:
        aris_data (pyARIS.ARIS_File): ARIS file data object containing
            the sonar frames and metadata.
        start_frame (int): Index of the first frame to extract (inclusive).
        end_frame (int): Index of the last frame to extract (exclusive).
        skip_frame (int, optional): Number of frames to skip between extracted frames.
            Defaults to 0 (extract every frame). For example, skip_frame=1 extracts
            every other frame.

    Returns:
        list[NDArray[np.uint8]]: List of successfully extracted and remapped
            sonar frames. Frames that failed to extract are omitted from the list.

    Note:
        - Uses parallel processing via ThreadPoolExecutor for performance
        - Failed extractions are logged and skipped (not included in output)
        - Progress is displayed via tqdm progress bar
        - Output order may differ from input order due to parallel execution
        - Frame indices follow Python conventions: range(start_frame, end_frame, skip_frame + 1)
    """
    aris_frames = []
    indices = range(start_frame, end_frame, skip_frame + 1)

    with ThreadPoolExecutor() as executor:
        # Submit all frame extraction tasks
        future_to_idx = {
            executor.submit(extract_frame_as_numpy_array, aris_data, idx): idx
            for idx in indices
        }

        for future in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="Processing frames",
        ):
            idx = future_to_idx[future]
            try:
                aris_frame = future.result()
                if aris_frame is not None:
                    aris_frames.append(aris_frame)
            except Exception as e:
                logging.error(f"Error processing frame {idx}: {e}")

    return aris_frames
