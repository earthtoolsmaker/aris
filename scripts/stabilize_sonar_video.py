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
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import aris.preprocessing
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
    gaussian_weights = aris.preprocessing.create_gaussian_kernel(window_size, sigma)
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

    # Initialize video writer for on-the-go output
    # Get frame dimensions from first frame
    ret, first_frame = cap.read()
    if not ret:
        logger.error("Could not read first frame from video")
        exit(1)

    height, width = first_frame.shape[:2]
    filepath_save.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        str(filepath_save), fourcc, int(fps), (width, height)
    )
    logger.info(f"Initialized video writer: {width}x{height} at {fps} fps")

    # Initialize sliding window buffer with first frame
    frame_buffer = deque([first_frame], maxlen=window_size)
    frame_idx = 0
    frames_written = 0

    logger.info("Processing frames with sliding window (memory-efficient)...")

    with tqdm(total=frames_to_process, desc="Stabilizing frames", unit="frame") as pbar:
        # Phase 1: Fill the buffer to have "future" frames (need center frames ahead)
        # Read center frames ahead to fill the right side of the window
        while (
            len(frame_buffer) < min(window_size, center + 1)
            and frame_idx < frames_to_process - 1
        ):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_buffer.append(frame)
            frame_idx += 1

        # Phase 2: Main processing loop - slide window and output frames
        while frame_idx < frames_to_process:
            # Calculate position of output frame within buffer
            # For a full window, output frame is at position 'center'
            # For incomplete windows at start, output frame is at position 0
            buffer_list = list(frame_buffer)
            output_position = min(frames_written, center)

            # Determine which frames to use for smoothing
            # At start: use fewer frames (incomplete window on left)
            # At end: use fewer frames (incomplete window on right)
            if frames_written < center:
                # Near start: use frames from position 0 to (frames_written + center + 1)
                window_frames = buffer_list[
                    : min(len(buffer_list), frames_written + center + 1)
                ]
                # Use weights starting from the appropriate position
                missing_before = center - frames_written
                active_weights = gaussian_weights[
                    missing_before : missing_before + len(window_frames)
                ]
            elif frame_idx >= frames_to_process - 1 and len(buffer_list) < window_size:
                # Near end: incomplete window on right
                window_frames = buffer_list[max(0, len(buffer_list) - window_size) :]
                # Calculate how many frames we're missing on the right
                frames_in_window = len(window_frames)
                active_weights = gaussian_weights[:frames_in_window]
            else:
                # Middle: full window
                window_frames = buffer_list
                active_weights = gaussian_weights

            # Renormalize weights
            active_weights = active_weights / np.sum(active_weights)

            # Apply weighted smoothing
            stabilized_frame = aris.preprocessing.smooth_frames_temporal(
                window_frames, active_weights
            )

            # Write frame immediately (memory-efficient!)
            video_writer.write(stabilized_frame)
            frames_written += 1
            pbar.update(1)

            # Read next frame and slide window
            if frame_idx < frames_to_process - 1:
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_buffer.append(frame)
                frame_idx += 1
            else:
                # No more input frames, but we still need to output remaining buffered frames
                if len(frame_buffer) > 1:
                    frame_buffer.popleft()  # Remove oldest frame
                else:
                    break

        # Phase 3: Output remaining frames in buffer (last 'center' frames)
        while len(frame_buffer) > 0 and frames_written < frames_to_process:
            buffer_list = list(frame_buffer)

            # Use all remaining frames in buffer
            window_frames = buffer_list
            frames_in_window = len(window_frames)

            # Use the first 'frames_in_window' weights
            active_weights = gaussian_weights[:frames_in_window]
            active_weights = active_weights / np.sum(active_weights)

            # Apply weighted smoothing
            stabilized_frame = aris.preprocessing.smooth_frames_temporal(
                window_frames, active_weights
            )

            # Write frame immediately
            video_writer.write(stabilized_frame)
            frames_written += 1
            pbar.update(1)

            # Remove oldest frame from buffer
            if len(frame_buffer) > 1:
                frame_buffer.popleft()
            else:
                break

    # Cleanup
    cap.release()
    video_writer.release()

    logger.info(f"Successfully stabilized and wrote {frames_written} frames")
    logger.info(f"Output saved to: {filepath_save}")
    logger.info("Done ✅")
