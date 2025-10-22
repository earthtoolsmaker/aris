"""
CLI script to encode MP4 videos into a different codec.

Usage:
    python encode_video_with_codec.py --filepath-video <video_file_path> --dir-save <save_directory> [--dir-videos <directory_with_videos>] [--video-codec <codec>] [-log <log_level>]

Parameters:
    --filepath-video: The file path of the video file to encode.
    --dir-videos: The directory containing video files to encode (optional).
    --dir-save: The directory to save the encoded videos (required).
    --video-codec: The codec to encode the video with (default: h264).
    -log, --loglevel: Provide logging level (default: info).
"""

import argparse
import logging
import random
from logging import Logger
from pathlib import Path

from tqdm import tqdm

import aris.video.utils as video_utils


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath-video",
        help="filepath of the video file to encode",
        type=Path,
    )
    parser.add_argument(
        "--dir-videos",
        help="directory containing video files to encode",
        type=Path,
    )
    parser.add_argument(
        "--dir-save",
        help="directory to save the videos",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--video-codec",
        choices=["h264"],
        type=str,
        help="Codec to encode the video with",
        default="h264",
        required=True,
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


def process_video_filepath(
    filepath_video: Path,
    dir_save: Path,
    video_codec: str,
    logger: Logger,
    force: bool = False,
) -> None:
    """
    Process a single video file by encoding it with the specified codec.

    This function checks if the output video file already exists and skips
    processing if it does, unless forced to overwrite. It encodes the input
    video file using the specified codec and saves the output in the
    specified directory.

    Parameters:
        filepath_video (Path): The path to the input video file to be encoded.
        dir_save (Path): The directory where the processed video will be saved.
        video_codec (str): The codec to use for encoding the video.
        logger (Logger): The logger instance to record processing information.
        force (bool): If True, will overwrite existing files without checking.
    """
    filepath_video_with_new_codec = dir_save / filepath_video.name
    logger.info(f"filepath_video_with_new_codec: {filepath_video_with_new_codec}")
    if (
        not force
        and filepath_video_with_new_codec
        and filepath_video_with_new_codec.exists()
    ):
        logger.info(
            f"Skipping because the video is already generated in {filepath_video_with_new_codec}"
        )
    elif video_codec == "h264":
        filepath_video_with_new_codec.parent.mkdir(parents=True, exist_ok=True)
        video_utils.encode_video_with_h264_codec(
            filepath_input=filepath_video,
            filepath_output=filepath_video_with_new_codec,
        )
        logger.info(f"Done with video filepath {filepath_video}")
    else:
        logger.error(f"Codec conversion {video_codec} not yet implemented")


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
        dir_videos = args["dir_videos"]
        filepaths_videos_to_process = (
            [filepath_video] if filepath_video else list(dir_videos.rglob("*.mp4"))
        )
        filepaths_videos_to_process_shuffled = random.Random().sample(
            filepaths_videos_to_process,
            k=len(filepaths_videos_to_process),
        )
        logger.info(
            f"Found {len(filepaths_videos_to_process_shuffled)} video files to process"
        )
        dir_save = args["dir_save"]
        video_codec = args["video_codec"]
        logger.info(f"Saving results in {dir_save}")
        dir_save.mkdir(parents=True, exist_ok=True)

        for fp_video in tqdm(filepaths_videos_to_process_shuffled):
            try:
                process_video_filepath(
                    filepath_video=fp_video,
                    dir_save=dir_save,
                    logger=logger,
                    video_codec=video_codec,
                )
            except Exception as e:
                logger.error(f"Error processing {fp_video}: {e}")

        logger.info("Done ✅")
