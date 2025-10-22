"""
CLI script to preprocess sonar videos with Gaussian blur, MOG2 background subtraction,
guided filtering, and temporal smoothing.

This script applies a preprocessing pipeline to sonar videos:
1. Gaussian blur to reduce noise in sonar imagery
2. MOG2 (Mixture of Gaussians) background subtraction to isolate moving objects (fish)
3. Bidirectional guided filtering (frame-guided MOG and MOG-guided frame)
4. Edge detection with Canny on both guided outputs
5. Edge intersection to identify high-confidence fish boundaries
6. Temporal smoothing to reduce frame-to-frame noise (0.8*current + 0.2*history)

The output is an RGB video with triple-channel visualization:
- Blue channel: Gaussian-blurred sonar frames (shows input structure after noise reduction)
- Green channel: Edge intersection (high-confidence fish boundaries)
- Red channel: Fully preprocessed frames (shows detected motion)

This visualization makes it easy to identify fish:
- Pure blue: Static sonar background
- Pure green: High-confidence object edges (fish boundaries)
- Pure red: Detected moving regions
- Yellow (green + red): Moving objects with clear boundaries → FISH (highest confidence)
- Cyan (blue + green): Static edges in the scene
- Magenta (blue + red): Motion over background structure

Usage:
    # Process a single video with default parameters
    uv run python ./scripts/preprocess_sonar_video.py \
        --filepath-video ./data/video.mp4 \
        --filepath-save ./data/preprocessed.mp4

    # Process with custom parameters
    uv run python ./scripts/preprocess_sonar_video.py \
        --filepath-video ./data/video.mp4 \
        --filepath-save ./data/preprocessed.mp4 \
        --gaussian-kernel 5 \
        --gaussian-sigma 1.4 \
        --mog-history 100

    # Process only first 1000 frames
    uv run python ./scripts/preprocess_sonar_video.py \
        --filepath-video ./data/video.mp4 \
        --filepath-save ./data/preprocessed.mp4 \
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
from tqdm import tqdm

import aris.preprocessing
import aris.video.utils as video_utils


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser for the sonar video preprocessing tool.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Preprocess sonar videos with Gaussian blur, MOG2, guided filter, and temporal smoothing."
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
        help="Gaussian blur sigma value (default: 1.4)",
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
        help="Guided filter epsilon/regularization (default: 0.01)",
    )
    parser.add_argument(
        "--temporal-weight",
        type=float,
        default=0.8,
        help="Weight for current frame in temporal smoothing (default: 0.8, meaning 80%% current + 20%% history)",
    )
    parser.add_argument(
        "--edge-canny-low",
        type=int,
        default=200,
        help="Canny edge detection lower threshold (default: 200)",
    )
    parser.add_argument(
        "--edge-canny-high",
        type=int,
        default=255,
        help="Canny edge detection upper threshold (default: 255)",
    )
    parser.add_argument(
        "--edge-dilation-size",
        type=int,
        default=2,
        help="Edge dilation size in pixels for tolerance (default: 2)",
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


def main():
    """Main entry point for the preprocess_sonar_video CLI script."""
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
    guided_radius = args["guided_radius"]
    guided_eps = args["guided_eps"]
    temporal_weight = args["temporal_weight"]
    edge_canny_low = args["edge_canny_low"]
    edge_canny_high = args["edge_canny_high"]
    edge_dilation_size = args["edge_dilation_size"]
    max_frames = args["max_frames"]

    # Get video metadata
    fps = video_utils.get_fps(filepath_video)
    logger.info(f"Video FPS: {fps}")

    # Initialize MOG2 background subtractor
    logger.info(f"Initializing MOG2 background subtractor with history={mog_history}")
    mog_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=mog_history, detectShadows=False
    )

    # Process video frames
    logger.info(f"Processing video: {filepath_video}")
    logger.info(
        f"Gaussian blur parameters: kernel={gaussian_kernel}x{gaussian_kernel}, sigma={gaussian_sigma}"
    )
    logger.info(f"Guided filter parameters: radius={guided_radius}, eps={guided_eps}")
    logger.info(
        f"Temporal smoothing: weight={temporal_weight} (current={temporal_weight * 100}%, history={(1 - temporal_weight) * 100}%)"
    )
    logger.info(
        f"Edge detection: Canny thresholds=({edge_canny_low}, {edge_canny_high}), dilation={edge_dilation_size}px"
    )

    cap = cv2.VideoCapture(str(filepath_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(total_frames, max_frames) if max_frames else total_frames

    logger.info(f"Total frames in video: {total_frames}")
    logger.info(f"Frames to process: {frames_to_process}")

    processed_frames = []
    frame_count = 0
    frame_history = None  # Track previous frame for temporal smoothing

    with tqdm(total=frames_to_process, desc="Processing frames", unit="frame") as pbar:
        while cap.isOpened() and frame_count < frames_to_process:
            ret, frame_rgb_input = cap.read()
            if not ret or frame_rgb_input is None:
                break

            # Extract blue channel from RGB input (sonar videos have identical channels)
            # This gives us a grayscale array for processing
            frame_gray = frame_rgb_input[:, :, 0]

            # Process the frame with full pipeline
            result = aris.preprocessing.preprocess_frame(
                frame=frame_gray,
                bg_subtractor=mog_subtractor,
                gaussian_kernel=gaussian_kernel,
                gaussian_sigma=gaussian_sigma,
                guided_radius=guided_radius,
                guided_eps=guided_eps,
                frame_history=frame_history,
                frame_count=frame_count,
                temporal_weight=temporal_weight,
                edge_canny_low=edge_canny_low,
                edge_canny_high=edge_canny_high,
                edge_dilation_size=edge_dilation_size,
            )

            # Update history for next frame
            frame_history = result.history

            # Create triple-channel RGB visualization
            visualization_frame = aris.preprocessing.create_visualization(
                result.blurred, result.edges, result.motion
            )
            processed_frames.append(visualization_frame)

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


if __name__ == "__main__":
    main()
