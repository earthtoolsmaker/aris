"""
CLI script to stabilize sonar videos using bidirectional Gaussian temporal smoothing.

This script reduces frame-to-frame jitter in sonar videos by applying temporal smoothing
across a sliding window of frames. Unlike causal temporal smoothing (which only uses
past frames), this bidirectional approach uses both past and future frames for smoother results.

Algorithm:
1. Maintain a sliding window of frames (e.g., 2 before + current + 2 after)
2. Apply Gaussian weights based on temporal distance
3. Compute weighted average to produce stabilized frame
4. Slide window forward and repeat

This is particularly useful for sonar videos which often have temporal noise/jitter from:
- Sensor noise
- Vessel motion
- Sonar beam instabilities

Usage:
    # Basic usage with default 5-frame window
    uv run python ./scripts/stabilize_sonar_video.py \
        --filepath-video ./data/jittery.mp4 \
        --filepath-save ./data/stabilized.mp4

    # Custom window size and sigma
    uv run python ./scripts/stabilize_sonar_video.py \
        --filepath-video ./data/jittery.mp4 \
        --filepath-save ./data/stabilized.mp4 \
        --window-size 7 \
        --sigma 1.5

    # Process only first 100 frames
    uv run python ./scripts/stabilize_sonar_video.py \
        --filepath-video ./data/jittery.mp4 \
        --filepath-save ./data/stabilized.mp4 \
        --max-frames 100

Parameters:
    --filepath-video: Path to the input video file (required)
    --filepath-save: Path to save the stabilized video (required)
    --window-size: Temporal window size in frames, must be odd (default: 5)
    --sigma: Gaussian kernel sigma for temporal weighting (default: 1.0)
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

import aris.video.utils as video_utils


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser for the video stabilization tool.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Stabilize sonar videos using bidirectional Gaussian temporal smoothing."
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
        help="Path to save the stabilized video",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Temporal window size in frames (must be odd, default: 5 = 2 before + current + 2 after)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian kernel sigma for temporal weighting (default: 1.0)",
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

    if args["window_size"] <= 0:
        logging.error(
            f"Invalid --window-size: must be positive, got {args['window_size']}"
        )
        return False

    if args["window_size"] % 2 == 0:
        logging.error(
            f"Invalid --window-size: must be odd (e.g., 3, 5, 7), got {args['window_size']}"
        )
        return False

    if args["sigma"] <= 0:
        logging.error(f"Invalid --sigma: must be positive, got {args['sigma']}")
        return False

    return True


def create_gaussian_kernel(window_size: int, sigma: float) -> NDArray[np.float32]:
    """
    Create a 1D Gaussian kernel for temporal weighting.

    The kernel is centered at the middle of the window, with weights decaying
    based on temporal distance according to a Gaussian function. Weights are
    normalized to sum to 1.0.

    Args:
        window_size: Size of the temporal window (must be odd)
        sigma: Standard deviation of the Gaussian function

    Returns:
        NDArray[np.float32]: 1D array of normalized Gaussian weights

    Example:
        >>> kernel = create_gaussian_kernel(5, 1.0)
        >>> # Returns weights for [t-2, t-1, t, t+1, t+2]
        >>> # with highest weight at center (t)
    """
    center = window_size // 2
    weights = np.array(
        [np.exp(-((i - center) ** 2) / (2 * sigma**2)) for i in range(window_size)],
        dtype=np.float32,
    )
    # Normalize so weights sum to 1
    return weights / np.sum(weights)


def smooth_frames_temporal(
    frames: list[NDArray[np.uint8]], weights: NDArray[np.float32]
) -> NDArray[np.uint8]:
    """
    Apply Gaussian weighted temporal smoothing to a window of frames.

    Computes a weighted average of frames using the provided Gaussian weights.
    This reduces temporal noise while preserving important features.

    Args:
        frames: List of frames in the temporal window
        weights: Gaussian weights for each frame (must have same length as frames)

    Returns:
        NDArray[np.uint8]: Smoothed frame

    Note:
        Weights should already be normalized to sum to 1.0
    """
    # Convert to float for accurate weighted averaging
    smoothed = np.zeros_like(frames[0], dtype=np.float32)

    for frame, weight in zip(frames, weights):
        smoothed += weight * frame.astype(np.float32)

    # Convert back to uint8
    return smoothed.astype(np.uint8)


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
    window_size = args["window_size"]
    sigma = args["sigma"]
    max_frames = args["max_frames"]

    # Get video metadata
    fps = video_utils.get_fps(filepath_video)
    logger.info(f"Video FPS: {fps}")

    # Create Gaussian kernel
    gaussian_weights = create_gaussian_kernel(window_size, sigma)
    center = window_size // 2
    logger.info(
        f"Temporal window: {window_size} frames ({center} before + current + {center} after)"
    )
    logger.info(f"Gaussian sigma: {sigma}")
    logger.info(f"Gaussian weights: {gaussian_weights}")

    # Open video
    cap = cv2.VideoCapture(str(filepath_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(total_frames, max_frames) if max_frames else total_frames

    logger.info(f"Total frames in video: {total_frames}")
    logger.info(f"Frames to process: {frames_to_process}")

    if window_size > frames_to_process:
        logger.warning(
            f"Window size ({window_size}) is larger than video length ({frames_to_process}). "
            f"This will work but may not provide optimal smoothing."
        )

    # Read all frames into memory for sliding window approach
    logger.info("Reading frames from video...")
    all_frames = []
    with tqdm(total=frames_to_process, desc="Reading frames", unit="frame") as pbar:
        for _ in range(frames_to_process):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            all_frames.append(frame)
            pbar.update(1)

    cap.release()
    logger.info(f"Read {len(all_frames)} frames into memory")

    # Process frames with sliding window
    logger.info("Applying Gaussian temporal smoothing...")
    stabilized_frames = []

    with tqdm(total=len(all_frames), desc="Stabilizing frames", unit="frame") as pbar:
        for frame_idx in range(len(all_frames)):
            # Calculate window boundaries for current frame
            # Handle edges: first and last frames have incomplete windows
            window_start = max(0, frame_idx - center)
            window_end = min(len(all_frames), frame_idx + center + 1)

            # Extract frames in window
            window_frames = all_frames[window_start:window_end]

            # Adjust weights for edge frames (incomplete windows)
            if len(window_frames) < window_size:
                # Calculate which weights to use
                if frame_idx < center:
                    # Near start: missing frames before
                    missing_before = center - frame_idx
                    active_weights = gaussian_weights[missing_before:]
                else:
                    # Near end: missing frames after
                    missing_after = (frame_idx + center + 1) - len(all_frames)
                    active_weights = gaussian_weights[: window_size - missing_after]

                # Renormalize weights to sum to 1
                active_weights = active_weights / np.sum(active_weights)
            else:
                active_weights = gaussian_weights

            # Apply weighted smoothing
            stabilized_frame = smooth_frames_temporal(window_frames, active_weights)
            stabilized_frames.append(stabilized_frame)

            pbar.update(1)

    logger.info(f"Successfully stabilized {len(stabilized_frames)} frames")

    # Save the stabilized video
    filepath_save.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving stabilized video to: {filepath_save}")

    video_utils.save_frames_to_video(
        frames=stabilized_frames,
        filepath_save=filepath_save,
        fps=int(fps),
    )

    logger.info("Done ✅")
