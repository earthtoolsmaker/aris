import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

import aris.frame as frame_utils
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
        help="Path to the background subtracted video.",
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
        iterator_frames = video_utils.get_all_frames(filepath_video)
        # logger.info(f"Extracted {len(frames)} frames from the video {filepath_video}")
        fps = video_utils.get_fps(filepath_video)
        logger.info(f"Extracted FPS {fps} from video {filepath_video}")
        # logger.info(frames[0])
        filepath_save.parent.mkdir(parents=True, exist_ok=True)
        # Initialize MOG2 background subtractor
        # mog = cv2.bgsegm.createBackgroundSubtractorMOG(100)
        mog = cv2.createBackgroundSubtractorMOG2(history=200, detectShadows=False)
        frames_mog = []
        # logger.info(frames[0].shape)
        # logger.info(frame_utils.grayscale_to_rgb(frames[0]).shape)
        for frame in iterator_frames:
            # WHY do we need to gaussian blur here?
            frame_blur = cv2.GaussianBlur(frame, (3, 3), 1.4)
            frame_mog = mog.apply(frame_blur)
            frame_rgb = frame_utils.grayscale_to_rgb(frame_mog)
            frames_mog.append(frame_rgb)

        # print(frames_mog)
        video_utils.save_frames_to_video(
            frames=frames_mog,
            filepath_save=filepath_save,
            fps=int(fps),
        )

        # average_frame = video_utils.get_average_frame(filepath_video=filepath_video)

        # frames_with_bakground_subtracted = []
        # if average_frame is not None:
        #     for frame in frames:
        #         frame_with_bakground_subtracted = np.subtract(frame.copy().astype("int16"), average_frame.astype("int16"))
        #         frames_with_bakground_subtracted.append(frame_with_bakground_subtracted.astype("uint8"))
        #     video_utils.save_frames_to_video(frames=frames_with_bakground_subtracted, filepath_save=filepath_save, fps=int(fps))
        #
