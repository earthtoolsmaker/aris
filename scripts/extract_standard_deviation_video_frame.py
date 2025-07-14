"""
CLI script to extract the standard deviation frame from a video.

Usage:
    python extract_standard_deviation_video_frame.py --filepath-video <path_to_video> --filepath-save <path_to_save_frame> [--max-frames <max_frames>] [-log <log_level>]

Arguments:
    --filepath-video     Path to the source video (required).
    --filepath-save      Path to the output averaged frame (required).
    --max-frames        Max number of frames to use (optional).
    -log, --loglevel    Provide logging level (default is 'info', options include 'debug', 'info', 'warning', etc.).
"""

import argparse
import logging
from pathlib import Path

import cv2

import aris.video.utils as video_utils


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser(
        description="Arguments for the video frame averaging tool."
    )
    parser.add_argument(
        "--filepath-video",
        type=Path,
        help="Path to the source video.",
        required=True,
    )
    parser.add_argument(
        "--filepath-save",
        required=True,
        type=Path,
        help="Path to the output averaged frame.",
    )
    parser.add_argument(
        "--max-frames",
        nargs="?",
        const=None,
        type=int,
        help="Max number of frames to use.",
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="info",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """
    Return whether the parsed args are valid.
    """
    if not args["filepath_video"].exists() or not args["filepath_video"].is_file():
        logging.error("Invalid --filepath-video filepath does not exist")
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
    else:
        logger.info(args)
        filepath_video = args["filepath_video"]
        filepath_save = args["filepath_save"]
        max_frames = args["max_frames"]
        frame_std = video_utils.get_std_frame(
            filepath_video=filepath_video,
            max_frames=max_frames,
        )
        if frame_std is not None:
            logger.info(f"Saving std frame in {filepath_save}")
            filepath_save.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(filepath_save, frame_std)
            exit(0)
        else:
            logger.error(f"Could not save the std frame from {filepath_video}")
            exit(1)
