from bg_sub.BgUtility import BgUtility
from bg_sub.FrameExtract import FrameExtract
from pyARIS import pyARIS

from pathlib import Path


def sanitize_frame_boundaries(
    frame_start: int | None, frame_end: int | None, aris_data
):
    if frame_start is None:
        frame_start = 0
    if frame_end is None:
        frame_end = aris_data.FrameCount - 1
    return frame_start, frame_end


def extract_frames(
    source_filepath: Path,
    output_filepath: Path,
    frame_start: int | None,
    frame_end: int | None,
    skip: int,
    fps: int = 24,
    invert: bool = False,
):
    aris_data, _ = pyARIS.DataImport(source_filepath)
    frame_start, frame_end = sanitize_frame_boundaries(
        frame_start,
        frame_end,
        aris_data,
    )
    frames = FrameExtract(aris_data).extract_frames(
        frame_start,
        frame_end,
        skipFrame=skip,
    )
    BgUtility.export_video(frames, output_filepath, invert_color=invert, fps=fps)
