"""
ARIS - Advanced Recording Information System Sonar Processing Toolkit.

This package provides tools for processing ARIS sonar files, including frame extraction,
video generation, and various preprocessing operations for underwater imaging analysis.

Modules:
    frame: Frame extraction and video generation from ARIS sonar data
    video: Video processing utilities (encoding, frame extraction, metadata)
    preprocessing: Sonar video preprocessing and stabilization functions
    pyARIS: ARIS file format parser (DDF_05 format handling)

Primary Use Cases:
    1. Convert ARIS sonar files to MP4 videos
    2. Extract and process individual sonar frames
    3. Apply video stabilization and preprocessing
    4. Background subtraction for motion detection
    5. Batch processing of sonar datasets

Example:
    # Basic ARIS to video conversion
    from aris import DataImport, extract_frames_as_numpy_arrays, aris_frames_to_mp4v_video
    from pathlib import Path

    # Load ARIS file and extract frames
    aris_data, first_frame = DataImport("sonar.aris")
    frames = extract_frames_as_numpy_arrays(
        aris_data, start_frame=0, end_frame=100
    )
    aris_frames_to_mp4v_video(
        frames, filepath_save=Path("output.mp4"), fps=24
    )

    # Video processing utilities
    from aris import get_average_frame, encode_video_with_h264_codec

    avg_frame = get_average_frame(Path("input.mp4"))
    encode_video_with_h264_codec(Path("input.mp4"), Path("output.mp4"))

Target Users:
    - Oceanographers analyzing underwater imagery
    - Marine researchers studying fish populations
    - Robotics engineers working with sonar systems

For command-line usage, install the package and use CLI commands:
    - aris-convert: Convert ARIS files to MP4
    - aris-chunk: Split videos into chunks
    - aris-extract-frame: Extract average frame
    - aris-encode: Re-encode videos with specific codec
    - aris-stabilize: Apply temporal stabilization
    - aris-preprocess: Apply preprocessing pipeline
    - aris-stabilize-preprocess: Combined stabilization and preprocessing
"""

__version__ = "0.1.0"

# Core ARIS file format handling
from aris.pyARIS.pyARIS import ARIS_File as ARISFile
from aris.pyARIS.pyARIS import ARIS_Frame as ARISFrame
from aris.pyARIS.pyARIS import DataImport
from aris.pyARIS.pyARIS import FrameRead

# Frame extraction and video generation
from aris.frame import (
    aris_frames_to_mp4v_video,
    extract_frame_as_numpy_array,
    extract_frames_as_numpy_arrays,
    get_recorded_at_datetime,
    grayscale_to_rgb,
    is_grayscale,
)

# Video utilities
from aris.video.utils import (
    encode_video_with_h264_codec,
    get_all_frames,
    get_average_frame,
    get_fps,
    get_video_duration,
    save_frames_to_video,
)

# Preprocessing functions
from aris.preprocessing import (
    create_dual_channel_visualization,
    create_gaussian_kernel,
    preprocess_frame,
    smooth_frames_temporal,
)

__all__ = [
    # Version
    "__version__",
    # Core ARIS classes and functions
    "ARISFile",
    "ARISFrame",
    "DataImport",
    "FrameRead",
    # Frame processing
    "aris_frames_to_mp4v_video",
    "extract_frame_as_numpy_array",
    "extract_frames_as_numpy_arrays",
    "get_recorded_at_datetime",
    "grayscale_to_rgb",
    "is_grayscale",
    # Video utilities
    "encode_video_with_h264_codec",
    "get_all_frames",
    "get_average_frame",
    "get_fps",
    "get_video_duration",
    "save_frames_to_video",
    # Preprocessing
    "create_dual_channel_visualization",
    "create_gaussian_kernel",
    "preprocess_frame",
    "smooth_frames_temporal",
]
