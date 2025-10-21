"""
Combined stabilization and preprocessing pipeline for sonar videos.

This script combines bidirectional temporal stabilization and motion detection
preprocessing in a single-pass operation. This is more efficient than running
stabilize_sonar_video.py and preprocess_sonar_video.py separately, as it avoids
intermediate file I/O.

Processing Pipeline:
1. Read frames into stabilization buffer (sliding window)
2. Apply Gaussian temporal smoothing (stabilization)
3. Apply preprocessing pipeline on stabilized frame:
   - Gaussian blur
   - MOG2 background subtraction
   - Guided filter
   - Preprocessing temporal smoothing
4. Create dual-channel visualization (blue=input, red=motion)
5. Write output frame immediately (memory-efficient)

Output Format:
- Blue channel: Gaussian-blurred stabilized frames
- Green channel: Empty (black)
- Red channel: Preprocessed frames showing detected motion

Visual interpretation:
- Pure blue: Static sonar background
- Pure red: Detected moving objects (fish)
- Magenta/purple: Motion overlapping with sonar structures
- Black: No signal

Usage:
    # Standard usage with default parameters
    uv run python ./scripts/stabilize_and_preprocess_sonar_video.py \\
        --filepath-video ./data/raw.mp4 \\
        --filepath-save ./data/processed.mp4

    # Custom parameters
    uv run python ./scripts/stabilize_and_preprocess_sonar_video.py \\
        --filepath-video ./data/raw.mp4 \\
        --filepath-save ./data/processed.mp4 \\
        --stabilize-window-size 7 \\
        --stabilize-sigma 1.5 \\
        --gaussian-kernel 5 \\
        --guided-radius 15

    # Process limited frames
    uv run python ./scripts/stabilize_and_preprocess_sonar_video.py \\
        --filepath-video ./data/raw.mp4 \\
        --filepath-save ./data/processed.mp4 \\
        --max-frames 100

Parameters:
    Stabilization:
        --stabilize-window-size: Temporal window for stabilization (default: 5)
        --stabilize-sigma: Gaussian sigma for stabilization (default: 1.0)

    Preprocessing:
        --gaussian-kernel: Blur kernel size (default: 3)
        --gaussian-sigma: Blur sigma (default: 1.4)
        --mog-history: MOG2 history length (default: 500)
        --guided-radius: Guided filter radius (default: 10)
        --guided-eps: Guided filter epsilon (default: 0.01)
        --temporal-weight: Weight for temporal smoothing (default: 0.8)

    Common:
        --filepath-video: Input video path (required)
        --filepath-save: Output video path (required)
        --max-frames: Max frames to process (optional)
        --loglevel: Logging level (default: info)
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
    Make the CLI parser for the combined stabilization and preprocessing tool.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Stabilize and preprocess sonar videos in a single pass."
    )

    # Input/Output
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
        help="Path to save the processed video",
    )

    # Stabilization parameters
    parser.add_argument(
        "--stabilize-window-size",
        type=int,
        default=5,
        help="Stabilization temporal window size (must be odd, default: 5)",
    )
    parser.add_argument(
        "--stabilize-sigma",
        type=float,
        default=1.0,
        help="Stabilization Gaussian kernel sigma (default: 1.0)",
    )

    # Preprocessing parameters
    parser.add_argument(
        "--gaussian-kernel",
        type=int,
        default=3,
        help="Preprocessing Gaussian blur kernel size (must be odd, default: 3)",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=1.4,
        help="Preprocessing Gaussian blur sigma (default: 1.4)",
    )
    parser.add_argument(
        "--mog-history",
        type=int,
        default=500,
        help="MOG2 background subtractor history length (default: 500)",
    )
    parser.add_argument(
        "--guided-radius",
        type=int,
        default=10,
        help="Guided filter radius (default: 10)",
    )
    parser.add_argument(
        "--guided-eps",
        type=float,
        default=0.01,
        help="Guided filter epsilon (default: 0.01)",
    )
    parser.add_argument(
        "--temporal-weight",
        type=float,
        default=0.8,
        help="Preprocessing temporal smoothing weight (default: 0.8)",
    )

    # Common parameters
    parser.add_argument(
        "--max-frames",
        nargs="?",
        const=None,
        type=int,
        help="Maximum number of frames to process (optional)",
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="info",
        help="Logging level (default: info)",
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

    if args["stabilize_window_size"] <= 0:
        logging.error(
            f"Invalid --stabilize-window-size: must be positive, got {args['stabilize_window_size']}"
        )
        return False

    if args["stabilize_window_size"] % 2 == 0:
        logging.error(
            f"Invalid --stabilize-window-size: must be odd, got {args['stabilize_window_size']}"
        )
        return False

    if args["stabilize_sigma"] <= 0:
        logging.error(
            f"Invalid --stabilize-sigma: must be positive, got {args['stabilize_sigma']}"
        )
        return False

    if args["gaussian_kernel"] <= 0 or args["gaussian_kernel"] % 2 == 0:
        logging.error(
            f"Invalid --gaussian-kernel: must be positive odd number, got {args['gaussian_kernel']}"
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

    if args["guided_radius"] <= 0:
        logging.error(
            f"Invalid --guided-radius: must be positive, got {args['guided_radius']}"
        )
        return False

    if args["guided_eps"] <= 0:
        logging.error(
            f"Invalid --guided-eps: must be positive, got {args['guided_eps']}"
        )
        return False

    if not (0 < args["temporal_weight"] <= 1.0):
        logging.error(
            f"Invalid --temporal-weight: must be between 0 and 1, got {args['temporal_weight']}"
        )
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
    stabilize_window_size = args["stabilize_window_size"]
    stabilize_sigma = args["stabilize_sigma"]
    gaussian_kernel = args["gaussian_kernel"]
    gaussian_sigma = args["gaussian_sigma"]
    mog_history = args["mog_history"]
    guided_radius = args["guided_radius"]
    guided_eps = args["guided_eps"]
    temporal_weight = args["temporal_weight"]
    max_frames = args["max_frames"]

    # Get video metadata
    fps = video_utils.get_fps(filepath_video)
    logger.info(f"Video FPS: {fps}")

    # Create stabilization Gaussian kernel
    stabilization_weights = aris.preprocessing.create_gaussian_kernel(
        stabilize_window_size, stabilize_sigma
    )
    stabilize_center = stabilize_window_size // 2
    logger.info(
        f"Stabilization window: {stabilize_window_size} frames "
        f"({stabilize_center} before + current + {stabilize_center} after)"
    )
    logger.info(f"Stabilization sigma: {stabilize_sigma}")
    logger.info(f"Stabilization weights: {stabilization_weights}")

    # Initialize MOG2 background subtractor
    logger.info(f"Initializing MOG2 background subtractor with history={mog_history}")
    mog_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=mog_history, detectShadows=False
    )

    # Log preprocessing parameters
    logger.info(
        f"Preprocessing Gaussian blur: kernel={gaussian_kernel}x{gaussian_kernel}, sigma={gaussian_sigma}"
    )
    logger.info(f"Guided filter: radius={guided_radius}, eps={guided_eps}")
    logger.info(
        f"Preprocessing temporal smoothing: weight={temporal_weight} "
        f"(current={temporal_weight*100}%, history={(1-temporal_weight)*100}%)"
    )

    # Open video
    cap = cv2.VideoCapture(str(filepath_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(total_frames, max_frames) if max_frames else total_frames

    logger.info(f"Total frames in video: {total_frames}")
    logger.info(f"Frames to process: {frames_to_process}")

    # Get frame dimensions from first frame
    ret, first_frame = cap.read()
    if not ret:
        logger.error("Could not read first frame from video")
        exit(1)

    height, width = first_frame.shape[:2]
    filepath_save.parent.mkdir(parents=True, exist_ok=True)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        str(filepath_save), fourcc, int(fps), (width, height)
    )
    logger.info(f"Initialized video writer: {width}x{height} at {fps} fps")

    # Initialize buffers and state
    stabilization_buffer = deque([first_frame], maxlen=stabilize_window_size)
    preprocessing_history = None
    frame_idx = 0
    frames_written = 0

    logger.info(
        "Processing frames with combined stabilization and preprocessing (memory-efficient)..."
    )

    with tqdm(total=frames_to_process, desc="Processing frames", unit="frame") as pbar:
        # Phase 1: Fill stabilization buffer with "future" frames
        while (
            len(stabilization_buffer) < min(stabilize_window_size, stabilize_center + 1)
            and frame_idx < frames_to_process - 1
        ):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            stabilization_buffer.append(frame)
            frame_idx += 1

        # Phase 2: Main processing loop
        while frame_idx < frames_to_process:
            buffer_list = list(stabilization_buffer)

            # Determine which frames to use for stabilization
            if frames_written < stabilize_center:
                # Near start: incomplete window on left
                window_frames = buffer_list[
                    : min(len(buffer_list), frames_written + stabilize_center + 1)
                ]
                missing_before = stabilize_center - frames_written
                active_weights = stabilization_weights[
                    missing_before : missing_before + len(window_frames)
                ]
            elif (
                frame_idx >= frames_to_process - 1
                and len(buffer_list) < stabilize_window_size
            ):
                # Near end: incomplete window on right
                window_frames = buffer_list[
                    max(0, len(buffer_list) - stabilize_window_size) :
                ]
                frames_in_window = len(window_frames)
                active_weights = stabilization_weights[:frames_in_window]
            else:
                # Middle: full window
                window_frames = buffer_list
                active_weights = stabilization_weights

            # Renormalize weights
            active_weights = active_weights / np.sum(active_weights)

            # Apply stabilization
            stabilized_frame = aris.preprocessing.smooth_frames_temporal(
                window_frames, active_weights
            )

            # Extract blue channel for preprocessing (sonar videos have identical RGB channels)
            stabilized_gray = stabilized_frame[:, :, 0]

            # Apply preprocessing pipeline
            (
                frame_blurred,
                frame_preprocessed,
                preprocessing_history,
            ) = aris.preprocessing.preprocess_frame(
                frame=stabilized_gray,
                bg_subtractor=mog_subtractor,
                gaussian_kernel=gaussian_kernel,
                gaussian_sigma=gaussian_sigma,
                guided_radius=guided_radius,
                guided_eps=guided_eps,
                frame_history=preprocessing_history,
                frame_count=frames_written,
                temporal_weight=temporal_weight,
            )

            # Create dual-channel visualization
            output_frame = aris.preprocessing.create_dual_channel_visualization(
                frame_blurred, frame_preprocessed
            )

            # Write frame immediately (memory-efficient!)
            video_writer.write(output_frame)
            frames_written += 1
            pbar.update(1)

            # Read next frame and slide stabilization window
            if frame_idx < frames_to_process - 1:
                ret, frame = cap.read()
                if ret and frame is not None:
                    stabilization_buffer.append(frame)
                frame_idx += 1
            else:
                # No more input frames, process remaining buffered frames
                if len(stabilization_buffer) > 1:
                    stabilization_buffer.popleft()
                else:
                    break

        # Phase 3: Drain remaining frames from stabilization buffer
        while len(stabilization_buffer) > 0 and frames_written < frames_to_process:
            buffer_list = list(stabilization_buffer)
            window_frames = buffer_list
            frames_in_window = len(window_frames)

            active_weights = stabilization_weights[:frames_in_window]
            active_weights = active_weights / np.sum(active_weights)

            # Apply stabilization
            stabilized_frame = aris.preprocessing.smooth_frames_temporal(
                window_frames, active_weights
            )

            # Extract blue channel and preprocess
            stabilized_gray = stabilized_frame[:, :, 0]

            (
                frame_blurred,
                frame_preprocessed,
                preprocessing_history,
            ) = aris.preprocessing.preprocess_frame(
                frame=stabilized_gray,
                bg_subtractor=mog_subtractor,
                gaussian_kernel=gaussian_kernel,
                gaussian_sigma=gaussian_sigma,
                guided_radius=guided_radius,
                guided_eps=guided_eps,
                frame_history=preprocessing_history,
                frame_count=frames_written,
                temporal_weight=temporal_weight,
            )

            # Create visualization and write
            output_frame = aris.preprocessing.create_dual_channel_visualization(
                frame_blurred, frame_preprocessed
            )
            video_writer.write(output_frame)
            frames_written += 1
            pbar.update(1)

            # Remove oldest frame
            if len(stabilization_buffer) > 1:
                stabilization_buffer.popleft()
            else:
                break

    # Cleanup
    cap.release()
    video_writer.release()

    logger.info(f"Successfully processed and wrote {frames_written} frames")
    logger.info(f"Output saved to: {filepath_save}")
    logger.info("Done ✅")
