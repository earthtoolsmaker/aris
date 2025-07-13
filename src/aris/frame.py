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
    Extract the datetime the aris_frame was recorded.

    Returns:
        datetime in utc
    """
    return datetime.datetime.fromtimestamp(
        aris_frame.sonartimestamp / 1000000, pytz.timezone("UTC")
    )


def grayscale_to_rgb(aris_frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Convert an ARIS grayscale frame into RGB by stacking the channels.

    Returns:
        rgb_frame (NDArray[np.uint8])
    """
    return np.stack((aris_frame,) * 3, axis=-1)


def is_grayscale(aris_frame: NDArray[np.uint8]) -> bool:
    """
    Whether the aris_frame is a grayscale frame.

    Returns:
        is_grayscale (bool)
    """
    return len(aris_frame.shape) == 2


def aris_frames_to_mp4v_video(
    aris_frames: list[NDArray[np.uint8]],
    filepath_save: Path,
    fps: int = 24,
) -> None:
    height = aris_frames[0].shape[0]
    width = aris_frames[0].shape[1]
    cv2.VideoWriter()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    filepath_save.parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(str(filepath_save), fourcc, fps, (width, height))
    for i in range(len(aris_frames)):
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
    Extract a single frame as a numpy array from the `aris_data`.
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
    Extract frames as numpy arrays from the `aris_data`.
    Track the progress with tqdm.

    Returns:
        aris_frames (list[NDArray[np.uint8]])
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

