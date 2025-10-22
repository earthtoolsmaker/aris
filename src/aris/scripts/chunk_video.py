"""
CLI script to chunk a large video into a sequence of fixed duration videos.

Usage:
    python chunk_video.py --filepath-video <path_to_video> --dir-save <path_to_save_chunked_videos> [-log <log_level>]

Arguments:
    --filepath-video     Path to the source video (optional).
    --dir-videos         Directory containing video files to chunk (optional).
    --dir-save           Path to the chunked videos (required).
    --duration-seconds   Duration in seconds of the video chunks (required).
    -log, --loglevel     Provide logging level (default is 'info', options include 'debug', 'info', 'warning', etc.).
"""

import argparse
import logging
import math
from logging import Logger
from pathlib import Path

import ffmpeg
from tqdm import tqdm

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
    )
    parser.add_argument(
        "--dir-videos",
        type=Path,
        help="directory containing video files to chunk.",
    )
    parser.add_argument(
        "--dir-save",
        required=True,
        type=Path,
        help="Path to the chunked videos.",
    )
    parser.add_argument(
        "--duration-seconds",
        required=True,
        type=int,
        help="Duration in seconds of the video chunks.",
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
    if not args["filepath_video"] and not args["dir_videos"]:
        logging.error(
            "Specify one parameter among --filepath-video to or --dir-videos, none specified."
        )
        return False
    if args["filepath_video"] and (
        not args["filepath_video"].exists() or not args["filepath_video"].is_file()
    ):
        logging.error("Invalid --filepath-video filepath does not exist")
        return False
    if args["dir_videos"] and (
        not args["dir_videos"].exists() or not args["dir_videos"].is_dir()
    ):
        logging.error("Invalid --dir-videos directory does not exist")
        return False
    return True


def chunk_video(
    filepath_video: Path,
    start_time_seconds: int,
    duration_seconds: int,
    filepath_save: Path,
) -> None:
    """
    Chunk the video into a smaller segment.

    Args:
        filepath_video (Path): Path to the source video file.
        start_time_seconds (int): The start time in seconds for the chunk.
        duration_seconds (int): The duration in seconds of the chunk.
        filepath_save (Path): Path where the chunked video will be saved.
    """
    (
        ffmpeg.input(str(filepath_video), ss=start_time_seconds, t=duration_seconds)
        .output(str(filepath_save))
        .run(overwrite_output=True)
    )


def process_video(
    filepath_video: Path,
    duration_seconds: int,
    dir_save: Path,
    logger: Logger,
) -> None:
    """
    Process the video by chunking it into smaller segments.

    Args:
        filepath_video (Path): Path to the source video file.
        duration_seconds (int): Duration in seconds for each video chunk.
        dir_save (Path): Directory where the chunked videos will be saved.
        logger (Logger): Logger instance for logging information and errors.
    """
    video_duration = video_utils.get_video_duration(filepath_video)
    if not video_duration:
        logger.error(
            f"Can not get the video duration for {filepath_video}, skip processing."
        )
    else:
        minutes = int(video_duration // 60)
        seconds = int(video_duration % 60)
        logger.info(
            f"Extracted video duration: {minutes}mn{seconds}s from {filepath_video}"
        )
        start_time_seconds_list = [
            i * duration_seconds
            for i in range(0, math.ceil(video_duration / duration_seconds))
        ]
        logger.info(
            f"Chunking the video in {len(start_time_seconds_list)} clips of {duration_seconds}s long"
        )
        for start_time_seconds in tqdm(start_time_seconds_list):
            filepath_save = (
                dir_save
                / filepath_video.stem
                / f"{filepath_video.stem}_start_{int(start_time_seconds)}_duration_{int(duration_seconds)}.mp4"
            )
            logger.info(f"Saving chunk in {filepath_save}")
            filepath_save.parent.mkdir(exist_ok=True, parents=True)
            logger.info(
                f"Chunking video {filepath_video} starting at {start_time_seconds}s for a duration of {duration_seconds}s"
            )
            chunk_video(
                filepath_video=filepath_video,
                start_time_seconds=int(start_time_seconds),
                duration_seconds=duration_seconds,
                filepath_save=filepath_save,
            )


def main():
    """Main entry point for the chunk_video CLI script."""
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
        dir_videos = args["dir_videos"]
        dir_save = args["dir_save"]
        duration_seconds = args["duration_seconds"]
        filepaths_videos_to_process = (
            [filepath_video] if filepath_video else list(dir_videos.rglob("*.mp4"))
        )
        logger.info(
            f"Found {len(filepaths_videos_to_process)} videos to process: {filepaths_videos_to_process}"
        )
        for fp_video in tqdm(filepaths_videos_to_process):
            logger.info(f"Processing video: {fp_video}")
            process_video(
                filepath_video=fp_video,
                duration_seconds=duration_seconds,
                dir_save=dir_save,
                logger=logger,
            )


if __name__ == "__main__":
    main()
