"""
CLI script to convert ARIS files into mp4 video files.

Arguments:
 --filepath-aris: Path to the ARIS file (required).
 --dir-save: Directory to save the generated video (required).
 --start-frame: Frame number to start extraction (default: 0).
 --end-frame: Frame number to stop extraction (optional).
 --skip-frame: Number of frames to skip during extraction (default: 0).
 --video-fps: Frames per second for the generated video (default: 24).
 --video-codec: Codec for the video (default: "h264").
 -log, --loglevel: Set the logging level (default: "warning").

Example usage:
python convert_aris_to_video.py --filepath-aris path/to/file.aris --dir-save path/to/save --start-frame 0 --end-frame 100 --video-fps 30
"""

import argparse
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from aris.pyARIS import pyARIS


def extract_frame(aris_data, idx: int) -> NDArray[np.uint8]:
    """
    Extract a single frame as a numpy array from the `aris_data`.
    """
    try:
        frame = pyARIS.FrameRead(aris_data, idx)
        return frame.remap
    except Exception as e:
        logging.error(f"Could not extract frame {idx}, {e}")
        return None  # Return None if there's an error


def extract_frames(
    aris_data,
    start_frame: int,
    end_frame: int,
    skip_frame: int = 0,
) -> List[NDArray[np.uint8]]:
    """
    Extract frames as numpy arrays from the `aris_data`.

    Returns:
        aris_frames (list[NDArray[np.uint8]])
    """
    aris_frames = []
    indices = range(start_frame, end_frame, skip_frame + 1)

    with ThreadPoolExecutor() as executor:
        # Submit all frame extraction tasks
        future_to_idx = {
            executor.submit(extract_frame, aris_data, idx): idx for idx in indices
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                frame = future.result()
                if frame is not None:
                    aris_frames.append(frame)
            except Exception as e:
                logging.error(f"Error processing frame {idx}: {e}")

    return aris_frames


# # TODO: parallelize the frame extraction to use multiprocessing
# def extract_frames(
#     aris_data,
#     start_frame: int,
#     end_frame: int,
#     skip_frame: int = 0,
# ) -> list[NDArray[np.uint8]]:
#     """
#     Extract frames as numpy arrays from the `aris_data`.
#
#     Returns:
#         aris_frames (list[NDArray[np.uint8]])
#     """
#     aris_frames = []
#     for idx in range(start_frame, end_frame, skip_frame + 1):
#         try:
#             frame = pyARIS.FrameRead(aris_data, idx)
#             aris_frames.append(frame.remap)
#         except Exception as e:
#             logging.error(f"Could not extract frame {idx}, {e}")
#     return aris_frames


def grayscale_to_rgb(aris_frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Convert an ARIS grayscale frame into RGB by stacking the channels.

    Returns:
        rgb_frame (NDArray[np.uint8])
    """
    return np.stack((aris_frame,) * 3, axis=-1)


def is_grayscale(frame: NDArray[np.uint8]) -> bool:
    """
    Whether the aris_frame is a grayscale frame.

    Returns:
        is_grayscale (bool)
    """
    return len(frame.shape) == 2


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


def sanitize_frame_boundaries(
    start_frame: int | None,
    end_frame: int | None,
    aris_data,
) -> Tuple[int, int]:
    """
    Sanitize the start_frame and end_frame values to not get out of bound
    errors when generating the videos and extracting the frames.
    """
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = int(aris_data.FrameCount - 1)
    return start_frame, end_frame


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath-aris",
        help="filepath of the ARIS file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dir-save",
        help="directory to save the video",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        help="frame start",
        default=0,
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        nargs="?",
        const=None,
        help="frame end",
    )
    parser.add_argument(
        "--skip-frame",
        type=int,
        help="frames to skip in between each frame",
        default=0,
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        help="Frames per second in the generated video",
        default=10,
    )
    parser.add_argument(
        "--video-codec",
        choices=["h264"],
        type=str,
        help="Frames per second in the generated video",
        default="h264",
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """
    Return whether the parsed args are valid.
    """
    if not args["filepath_aris"].exists() or not args["filepath_aris"].is_file():
        logging.error("Invalid --filepath-aris filepath does not exist")
        return False
    return True


def encode_video_with_h264_codec(filepath_input: Path, filepath_output: Path):
    assert filepath_input.exists(), "filepath_input does not exist!"

    command = [
        "ffmpeg",
        "-i",
        str(filepath_input),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        str(filepath_output),
    ]

    logger.info(f"Command to run: {command}")

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info("Command executed successfully.")

    except subprocess.CalledProcessError as e:
        logger.error("An error occurred while executing the command.")
        logger.error("Error message:", e.stderr.decode())


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        logging.error(f"Could not validate the parsed args: {args}")
        exit(1)
    else:
        logger.info(args)
        filepath_aris = args["filepath_aris"]
        dir_save = args["dir_save"]
        start_frame = args["start_frame"]
        end_frame = args["end_frame"]
        skip_frame = args["skip_frame"]
        video_fps = args["video_fps"]
        video_codec = args["video_codec"]
        logger.info(f"Saving results in {dir_save}")
        dir_save.mkdir(parents=True, exist_ok=True)
        logger.info(f"Parsing ARIS file {filepath_aris}")
        aris_data, _ = pyARIS.DataImport(str(filepath_aris))
        aris_data.info()
        start_frame_sanitized, end_frame_sanitized = sanitize_frame_boundaries(
            start_frame,
            end_frame,
            aris_data,
        )
        logger.info(
            f"Parsing frames from frame {start_frame_sanitized} until frame {end_frame_sanitized}"
        )
        frames = extract_frames(
            aris_data=aris_data,
            start_frame=start_frame_sanitized,
            end_frame=end_frame_sanitized,
            skip_frame=skip_frame,
        )
        filepath_save = (
            dir_save
            / f"{filepath_aris.stem}_from_{start_frame_sanitized}_to_{end_frame_sanitized}.mp4"
        )
        aris_frames_to_mp4v_video(
            aris_frames=frames,
            filepath_save=filepath_save,
            fps=video_fps,
        )
        logger.info("Encode video with H.264 codec")
        encode_video_with_h264_codec(
            filepath_input=filepath_save,
            filepath_output=filepath_save.parent / f"encoded_h264_{filepath_save.name}",
        )
        logger.info("Done ✅")
