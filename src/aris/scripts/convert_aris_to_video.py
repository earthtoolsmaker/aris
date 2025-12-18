"""
CLI script to convert ARIS files into MP4 video files.

This script provides a command-line interface for converting ARIS files, which
contain sonar data, into MP4 video files using ffmpeg via pyARIS.VideoExport().

Arguments:

--filepath-aris: Path to the ARIS file (either --filepath-aris or --dir-aris required).
--dir-aris: Directory containing ARIS files to convert (either --filepath-aris or --dir-aris required).
--dir-save: Directory to save the generated video (required).
-log, --loglevel: Set the logging level (default: "info").

Example usage:

uv run python ./src/aris/scripts/convert_aris_to_video.py --filepath-aris data/aris/file.aris --dir-save data/mp4/
uv run python ./src/aris/scripts/convert_aris_to_video.py --dir-aris data/aris/ --dir-save data/mp4/

This script facilitates the extraction of frames from ARIS data and encodes
them into a video format using ffmpeg, helping users visualize sonar data effectively.

Requirements:
    - ffmpeg must be installed and available in PATH
"""

import argparse
import logging
import shutil
from logging import Logger
from pathlib import Path

from aris.pyARIS import pyARIS


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser(
        description="Convert ARIS sonar files to MP4 videos using ffmpeg."
    )
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
        "-log",
        "--loglevel",
        default="info",
        help="Provide logging level. Example --loglevel debug, default=info",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """
    Return whether the parsed args are valid.
    """
    if not args["filepath_aris"] and not args["dir_aris"]:
        logging.error(
            "Specify one parameter among --filepath-aris or --dir-aris, none specified."
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


def process_aris_filepath(
    filepath_aris: Path,
    dir_save: Path,
    logger: Logger,
    force: bool = False,
) -> bool:
    """
    Process a single ARIS file by converting it to MP4 video using ffmpeg.

    Parameters:
        filepath_aris (Path): The path to the ARIS file to be processed.
        dir_save (Path): The directory where the generated video will be saved.
        logger (Logger): The logger instance for logging information and errors.
        force (bool): If True, overwrite existing video files. Defaults to False.

    Returns:
        bool: True if conversion successful, False otherwise.
    """
    filepath_video_save = dir_save / f"{filepath_aris.stem}.mp4"

    if not force and filepath_video_save.exists():
        logger.info(f"Skipping {filepath_aris.name} - output already exists")
        return True

    try:
        logger.info(f"Loading ARIS file: {filepath_aris.name}")
        aris_data, first_frame = pyARIS.DataImport(str(filepath_aris))

        # Get frame rate from first frame, fallback to 24 FPS
        fps = float(first_frame.framerate) if first_frame.framerate > 0 else 24.0
        logger.info(f"Frame rate: {fps} FPS, Total frames: {aris_data.FrameCount}")

        logger.info(f"Exporting video to: {filepath_video_save}")

        # VideoExport uses ffmpeg subprocess with MPEG4 codec
        pyARIS.VideoExport(
            data=aris_data,
            filename=str(filepath_video_save),
            fps=fps,
            start_frame=1,  # pyARIS uses 1-indexed frames
            end_frame=None,  # Process all frames
            timestamp=False,  # No timestamp overlay
        )

        logger.info(f"Successfully converted {filepath_aris.name}")
        return True

    except Exception as e:
        logger.error(f"Error converting {filepath_aris.name}: {e}")
        return False


def main():
    """Main entry point for the convert_aris_to_video CLI script."""
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=args["loglevel"].upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Validate ffmpeg is available
    if not shutil.which("ffmpeg"):
        logger.error("ffmpeg not found in PATH. Please install ffmpeg.")
        exit(1)

    if not validate_parsed_args(args):
        logging.error(f"Could not validate the parsed args: {args}")
        exit(1)

    filepath_aris = args["filepath_aris"]
    dir_aris = args["dir_aris"]
    filepaths_aris_to_process = (
        [filepath_aris] if filepath_aris else sorted(dir_aris.rglob("*.aris"))
    )

    if not filepaths_aris_to_process:
        logger.warning(f"No ARIS files found to process")
        exit(0)

    logger.info(f"Found {len(filepaths_aris_to_process)} ARIS file(s) to process")

    dir_save = args["dir_save"]
    dir_save.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results in {dir_save}")

    success_count = 0
    failed_count = 0

    for fp_aris in filepaths_aris_to_process:
        if process_aris_filepath(
            filepath_aris=fp_aris,
            dir_save=dir_save,
            logger=logger,
        ):
            success_count += 1
        else:
            failed_count += 1

    logger.info("Conversion complete!")
    logger.info(f"Successfully converted: {success_count}/{len(filepaths_aris_to_process)} files")

    if failed_count > 0:
        logger.warning(f"Failed to convert: {failed_count}/{len(filepaths_aris_to_process)} files")
        exit(1)

    logger.info("Done")


if __name__ == "__main__":
    main()
