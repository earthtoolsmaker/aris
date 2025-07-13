"""
CLI script to convert ARIS files into mp4 video files.

Arguments:
 --filepath-aris: Path to the ARIS file (required).
 --dir-save: Directory to save the generated video (required).
 --start-frame: Frame number to start extraction (default: 0).
 --end-frame: Frame number to stop extraction (optional).
 --skip-frame: Number of frames to skip during extraction (default: 0).
 --video-codec: Codec for the video (default: "h264").
 -log, --loglevel: Set the logging level (default: "warning").

Example usage:
python convert_aris_to_video.py --filepath-aris path/to/file.aris --dir-save path/to/save --start-frame 0 --end-frame 100
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import aris.frame as aris_frame
import aris.video.utils as video_utils
from aris.pyARIS import pyARIS


def sanitize_frame_boundaries(
    start_frame: int | None,
    end_frame: int | None,
    aris_data,
) -> Tuple[int, int]:
    """
    Sanitize the start_frame and end_frame values to not get out of bound
    errors when generating the videos and extracting the frames.
    """
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = int(aris_data.FrameCount - 1)
    return start_frame, end_frame


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath-aris",
        help="filepath of the ARIS file",
        type=Path,
        required=True,
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
        "--skip-frame",
        type=int,
        help="frames to skip in between each frame",
        default=0,
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
    if not args["filepath_aris"].exists() or not args["filepath_aris"].is_file():
        logging.error("Invalid --filepath-aris filepath does not exist")
        return False
    return True


def get_filepath_video_save(
    args_cli: dict,
    aris_data: pyARIS.ARIS_File,
    dir_save: Path,
) -> Path:
    """
    Construct a filepath to save the converted video.
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
        dir_save = args["dir_save"]
        start_frame = args["start_frame"]
        end_frame = args["end_frame"]
        skip_frame = args["skip_frame"]
        video_codec = args["video_codec"]
        logger.info(f"Saving results in {dir_save}")
        dir_save.mkdir(parents=True, exist_ok=True)
        logger.info(f"Parsing ARIS file {filepath_aris}")
        aris_data, first_frame = pyARIS.DataImport(str(filepath_aris))
        logger.info("ARIS data details:")
        aris_data.info()
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
            skip_frame=skip_frame,
        )
        filepath_video_save = get_filepath_video_save(
            args_cli=args,
            aris_data=aris_data,
            dir_save=dir_save,
        )
        logger.info(
            f"Generating a video file from the ARIS file in {filepath_video_save}"
        )
        aris_frame.aris_frames_to_mp4v_video(
            aris_frames=aris_frames,
            filepath_save=filepath_video_save,
            fps=int(frame_rate_aris),
        )
        filepath_h264_video_save = (
            filepath_video_save.parent / f"encoded_h264_{filepath_video_save.name}"
        )
        logger.info("Encode video with H.264 codec in {filepath_h264_video_save}")
        video_utils.encode_video_with_h264_codec(
            filepath_input=filepath_video_save,
            filepath_output=filepath_h264_video_save,
        )
        logger.info("Done ✅")

## REPL
# filepath_aris = Path(
#     "../bc-hydro-sonar/data/01_raw/jansen-lake-2025/ARIS_2025_05_06/2025-05-06_151500.aris"
# )
# filepath_aris.exists()
#
# aris_data, first_frame = pyARIS.DataImport(str(filepath_aris))
# aris_data
# aris_data.RecordInterval
# aris_data.RadioSeconds
# aris_data.strDate
# aris_data.FrameInterval
#
# foo.tsminute
# foo.tshour
# import datetime
#
# import pytz
#
# foo.frametime
# foo.sonartimestamp
#
# get_datetime(foo)
# get_datetime(last_frame)
#
# datetime.datetime.fromtimestamp(foo.sonartimestamp / 1000000, pytz.timezone("UTC"))
# datetime.datetime.utcfromtimestamp(foo.frametime / 1000000)
# datetime.datetime
# # foo.latitude
# # foo.longitude
# # foo.sonarx
# # foo.daygps
# # foo.samplerate
# # foo.sampleperiod
# foo.framerate
# # foo.cputempfault
# # foo.uptime
# # foo.arisappverionmajor
# # foo.sonartimestamp
# # foo.frameindex
# # foo.frametime
# # foo.status
# # foo.transmitmode
# # foo.receivergain
# # foo.humidity
# # foo.focus
# # foo.depth
# # foo.pitch
# #
# # foo.info()
# #
# # aris_data.FrameCount
# # first_frame.info()
# # last_frame = pyARIS.FrameRead(aris_data, aris_data.FrameCount - 1)
# # last_frame.info()
