"""
CLI script to preprocess sonar videos with Gaussian blur and MOG2 background subtraction.

This script applies a preprocessing pipeline to sonar videos based on the approach
from Salmon Computer Vision (https://github.com/Salmon-Computer-Vision/salmon-vision-sonar):
1. Gaussian blur to reduce noise in sonar imagery
2. MOG2 (Mixture of Gaussians) background subtraction to isolate moving objects (fish)

The output is a grayscale video showing the MOG2 background subtraction mask, which
highlights moving objects against the background.

Usage:
    # Process a single video with default parameters
    uv run python ./scripts/preprocess_sonar_video.py \\
        --filepath-video ./data/video.mp4 \\
        --filepath-save ./data/preprocessed.mp4

    # Process with custom parameters
    uv run python ./scripts/preprocess_sonar_video.py \\
        --filepath-video ./data/video.mp4 \\
        --filepath-save ./data/preprocessed.mp4 \\
        --gaussian-kernel 5 \\
        --gaussian-sigma 1.4 \\
        --mog-history 100

    # Process only first 1000 frames
    uv run python ./scripts/preprocess_sonar_video.py \\
        --filepath-video ./data/video.mp4 \\
        --filepath-save ./data/preprocessed.mp4 \\
        --max-frames 1000

Parameters:
    --filepath-video: Path to the input video file (required)
    --filepath-save: Path to save the preprocessed video (required)
    --gaussian-kernel: Gaussian blur kernel size, must be odd (default: 3)
    --gaussian-sigma: Gaussian blur sigma value (default: 1.4)
    --mog-history: MOG background subtractor history length (default: 100)
    --max-frames: Maximum number of frames to process (optional, processes all if not specified)
    -log, --loglevel: Logging level (default: info)
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

import aris.frame as frame_utils
import aris.video.utils as video_utils


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser for the sonar video preprocessing tool.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Preprocess sonar videos with Gaussian blur and MOG2 background subtraction."
    )
    parser.add_argument(
        "--filepath-video",
        type=Path,
        help="Path to the input video file",
        required=True,
    )
    parser.add_argument(
        "--filepath-save",
        required=True,
        type=Path,
        help="Path to save the preprocessed video",
    )
    parser.add_argument(
        "--gaussian-kernel",
        type=int,
        default=3,
        help="Gaussian blur kernel size (must be odd, default: 3)",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=1.4,
        help="Gaussian blur sigma value (default: 1.4, same as Salmon Vision)",
    )
    parser.add_argument(
        "--mog-history",
        type=int,
        default=500,
        help="MOG2 background subtractor history length (default: 500)",
    )
    parser.add_argument(
        "--max-frames",
        nargs="?",
        const=None,
        type=int,
        help="Maximum number of frames to process (optional, processes all if not specified)",
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="info",
        help="Provide logging level. Example --loglevel debug, default=info",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """
    Validate the parsed command-line arguments.

    Args:
        args: Dictionary of parsed arguments

    Returns:
        bool: True if arguments are valid, False otherwise
    """
    if not args["filepath_video"].exists() or not args["filepath_video"].is_file():
        logging.error("Invalid --filepath-video: file does not exist")
        return False

    if args["gaussian_kernel"] <= 0 or args["gaussian_kernel"] % 2 == 0:
        logging.error(
            f"Invalid --gaussian-kernel: must be a positive odd number, got {args['gaussian_kernel']}"
        )
        return False

    if args["gaussian_sigma"] <= 0:
        logging.error(
            f"Invalid --gaussian-sigma: must be positive, got {args['gaussian_sigma']}"
        )
        return False

    if args["mog_history"] <= 0:
        logging.error(
            f"Invalid --mog-history: must be positive, got {args['mog_history']}"
        )
        return False

    return True


def process_frame(
    frame: NDArray[np.uint8],
    bg_subtractor: cv2.BackgroundSubtractorMOG2,
    gaussian_kernel: int,
    gaussian_sigma: float,
) -> NDArray[np.uint8]:
    """
    Process a single frame with Gaussian blur and MOG2 background subtraction.

    This implements steps 1-2 from the Salmon Computer Vision pipeline:
    1. Apply Gaussian blur to reduce noise in sonar imagery
    2. Apply MOG2 background subtraction to isolate moving objects

    Args:
        frame: Input frame (grayscale or RGB)
        bg_subtractor: MOG2 background subtractor instance
        gaussian_kernel: Gaussian blur kernel size (must be odd)
        gaussian_sigma: Gaussian blur sigma value

    Returns:
        Preprocessed frame (grayscale MOG2 mask)
    """
    # Step 1: Apply Gaussian blur to reduce noise
    # This is critical for sonar imagery which has significant speckle noise
    frame_blurred = cv2.GaussianBlur(
        frame, (gaussian_kernel, gaussian_kernel), gaussian_sigma
    )

    # Step 2: Apply MOG2 background subtraction
    # This isolates moving objects (fish) from the static background
    frame_mog = bg_subtractor.apply(frame_blurred)

    return frame_mog


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())

    if not validate_parsed_args(args):
        logging.error(f"Could not validate the parsed args: {args}")
        exit(1)

    logger.info(args)

    # Extract arguments
    filepath_video = args["filepath_video"]
    filepath_save = args["filepath_save"]
    gaussian_kernel = args["gaussian_kernel"]
    gaussian_sigma = args["gaussian_sigma"]
    mog_history = args["mog_history"]
    max_frames = args["max_frames"]

    # Get video metadata
    fps = video_utils.get_fps(filepath_video)
    logger.info(f"Video FPS: {fps}")

    # Initialize MOG2 background subtractor
    logger.info(f"Initializing MOG2 background subtractor with history={mog_history}")
    mog_subtractor = cv2.createBackgroundSubtractorMOG2(history=mog_history, detectShadows=False)

    # Process video frames
    logger.info(f"Processing video: {filepath_video}")
    logger.info(
        f"Gaussian blur parameters: kernel={gaussian_kernel}x{gaussian_kernel}, sigma={gaussian_sigma}"
    )

    cap = cv2.VideoCapture(str(filepath_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(total_frames, max_frames) if max_frames else total_frames

    logger.info(f"Total frames in video: {total_frames}")
    logger.info(f"Frames to process: {frames_to_process}")

    processed_frames = []
    frame_count = 0

    with tqdm(total=frames_to_process, desc="Processing frames", unit="frame") as pbar:
        while cap.isOpened() and frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Process the frame
            frame_processed = process_frame(
                frame=frame,
                bg_subtractor=mog_subtractor,
                gaussian_kernel=gaussian_kernel,
                gaussian_sigma=gaussian_sigma,
            )

            # Convert grayscale MOG output to RGB for video saving
            # (video codecs expect 3-channel input)
            frame_rgb = frame_utils.grayscale_to_rgb(frame_processed)
            processed_frames.append(frame_rgb)

            frame_count += 1
            pbar.update(1)

    cap.release()
    logger.info(f"Successfully processed {frame_count} frames")

    # Save the processed video
    filepath_save.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving preprocessed video to: {filepath_save}")

    video_utils.save_frames_to_video(
        frames=processed_frames,
        filepath_save=filepath_save,
        fps=int(fps),
    )

    logger.info("Done ✅")
