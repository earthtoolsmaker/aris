"""
CLI script to convert ARIS files into MP4 video files.

This script provides a command-line interface for converting ARIS files, which
contain sonar data, into MP4 video files. Users can specify various parameters,
including the path to the ARIS file, the directory for saving the generated
video, and frame extraction boundaries.

Arguments:

--filepath-aris: Path to the ARIS file (required).
--dir-aris: Directory containing ARIS files to convert (optional).
--dir-save: Directory to save the generated video (required).
--start-frame: Frame number to start extraction (default: 0).
--end-frame: Frame number to stop extraction (optional).
--video-codec: Codec for the video (default: "h264").
-log, --loglevel: Set the logging level (default: "warning").

Example usage:

python convert_aris_to_video.py --filepath-aris data/aris/file.aris --dir-save data/mp4/ --start-frame 0 --end-frame 100

This script facilitates the extraction of frames from ARIS data and encodes
them into a video format, helping users visualize sonar data effectively.
"""

import argparse
import logging
from logging import Logger
from pathlib import Path
from typing import Tuple

import aris.frame as aris_frame
import aris.video.utils as video_utils
from aris.pyARIS import pyARIS


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath-aris",
        help="filepath of the ARIS file to convert",
        type=Path,
    )
    parser.add_argument(
        "--dir-aris",
        help="directory containing ARIS files to convert",
        type=Path,
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
    if not args["filepath_aris"] and not args["dir_aris"]:
        logging.error(
            "Specify one parameter among --filepath-aris to or --dir-aris, none specified."
        )
        return False
    if args["filepath_aris"] and (
        not args["filepath_aris"].exists() or not args["filepath_aris"].is_file()
    ):
        logging.error("Invalid --filepath-aris filepath does not exist")
        return False
    if args["dir_aris"] and (
        not args["dir_aris"].exists() or not args["dir_aris"].is_dir()
    ):
        logging.error("Invalid --dir-aris directory does not exist")
        return False
    return True


def sanitize_frame_boundaries(
    start_frame: int | None,
    end_frame: int | None,
    aris_data,
) -> Tuple[int, int]:
    """
    Sanitize the start_frame and end_frame values to ensure they are within
    the valid range of frame indices for the given ARIS data. This helps to
    prevent out-of-bounds errors during video generation and frame extraction.

    Parameters:
        start_frame (int | None): The frame number to start extraction from.
                                   If None, defaults to 0.
        end_frame (int | None): The frame number to stop extraction at.
                                 If None, it will be set to the last frame
                                 index based on the ARIS data.

        aris_data: The ARIS data object containing information about the
                    total number of frames.

    Returns:
        Tuple[int, int]: A tuple containing the sanitized start_frame and
                         end_frame values.
    """
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = int(aris_data.FrameCount - 1)
    return start_frame, end_frame


def get_filepath_video_save(
    args_cli: dict,
    filepath_aris: Path,
    aris_data: pyARIS.ARIS_File,
    dir_save: Path,
) -> Path:
    """
    Construct a filepath for saving the converted video based on the
    input ARIS file and specified frame boundaries.

    Parameters:
        args_cli (dict): A dictionary containing command line arguments,
                         including start and end frame information.
        aris_data (pyARIS.ARIS_File): The ARIS data object that holds
                                        information about the ARIS file.
        dir_save (Path): The directory where the converted video will be saved.

    Returns:
        Path: The constructed filepath for the converted video.
    """
    start_frame_sanitized, end_frame_sanitized = sanitize_frame_boundaries(
        args_cli["start_frame"],
        args_cli["end_frame"],
        aris_data,
    )
    number_frames = aris_data.FrameCount
    if start_frame_sanitized == 0 and end_frame == number_frames - 1:
        return dir_save / f"{filepath_aris.stem}.mp4"
    else:
        return (
            dir_save
            / f"{filepath_aris.stem}_fromframe_{start_frame_sanitized}_toframe_{end_frame_sanitized}.mp4"
        )


def process_aris_filepath(
    start_frame: int | None,
    end_frame: int | None,
    filepath_aris: Path,
    dir_save: Path,
    logger: Logger,
    force: bool = False,
) -> None:
    """
    Process a single ARIS file by extracting frames, generating a video, and encoding it with H.264 codec.
    The processed video is saved in the specified directory.

    Parameters:
        start_frame (int | None): The frame number to start extraction from. If None, extraction starts from the first frame.
        end_frame (int | None): The frame number to stop extraction at. If None, extraction continues to the last frame.
        filepath_aris (Path): The path to the ARIS file to be processed.
        dir_save (Path): The directory where the generated video will be saved.
        logger (Logger): The logger instance for logging information and errors.
        force (bool): If True, overwrite existing video files. Defaults to False.
    """
    logger.info(f"Parsing ARIS file {filepath_aris}")
    aris_data, first_frame = pyARIS.DataImport(str(filepath_aris))
    logger.info("ARIS data details:")
    aris_data.info()
    filepath_video_save = get_filepath_video_save(
        args_cli=args,
        filepath_aris=filepath_aris,
        aris_data=aris_data,
        dir_save=dir_save,
    )
    logger.info(f"filepath_video_save: {filepath_video_save}")
    filepath_h264_video_save = (
        filepath_video_save.parent / f"encoded_h264_{filepath_video_save.name}"
    )
    logger.info(f"filepath_h264_video_save: {filepath_h264_video_save}")
    if (
        not force
        and filepath_h264_video_save
        and filepath_video_save.exists()
        and filepath_h264_video_save
        and filepath_h264_video_save.exists()
    ):
        logger.info(
            f"Skipping because the video is already generated in {filepath_video_save}"
        )
    else:
        last_frame = pyARIS.FrameRead(aris_data, aris_data.FrameCount - 1)
        logger.info("First frame details:")
        first_frame.info()
        logger.info("Last frame details:")
        last_frame.info()
        frame_rate_aris = first_frame.framerate
        start_frame_sanitized, end_frame_sanitized = sanitize_frame_boundaries(
            start_frame,
            end_frame,
            aris_data,
        )
        logger.info(
            f"Parsing frames from frame {start_frame_sanitized} until frame {end_frame_sanitized}"
        )
        aris_frames = aris_frame.extract_frames_as_numpy_arrays(
            aris_data=aris_data,
            start_frame=start_frame_sanitized,
            end_frame=end_frame_sanitized,
            skip_frame=0,
        )
        logger.info(
            f"Generating a video file from the ARIS file in {filepath_video_save}"
        )
        aris_frame.aris_frames_to_mp4v_video(
            aris_frames=aris_frames,
            filepath_save=filepath_video_save,
            fps=int(frame_rate_aris),
        )
        logger.info(f"Encode video with H.264 codec in {filepath_h264_video_save}")
        video_utils.encode_video_with_h264_codec(
            filepath_input=filepath_video_save,
            filepath_output=filepath_h264_video_save,
        )
        logger.info(f"Done with ARIS filepath {filepath_aris}")


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
        dir_aris = args["dir_aris"]
        filepaths_aris_to_process = filepath_aris or list(dir_aris.rglob("*.aris"))
        logger.info(f"Found {len(filepaths_aris_to_process)} ARIS files to process")
        dir_save = args["dir_save"]
        start_frame = args["start_frame"]
        end_frame = args["end_frame"]
        video_codec = args["video_codec"]
        logger.info(f"Saving results in {dir_save}")
        dir_save.mkdir(parents=True, exist_ok=True)

        for fp_aris in filepaths_aris_to_process:
            try:
                process_aris_filepath(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    filepath_aris=fp_aris,
                    dir_save=dir_save,
                    logger=logger,
                )
            except Exception as e:
                logger.error(f"Error processing {fp_aris}: {e}")

        logger.info("Done ✅")
