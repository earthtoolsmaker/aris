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
    from aris.pyARIS import pyARIS
    import aris.frame as aris_frame

    aris_data, first_frame = pyARIS.DataImport("sonar.aris")
    frames = aris_frame.extract_frames_as_numpy_arrays(
        aris_data, start_frame=0, end_frame=100
    )
    aris_frame.aris_frames_to_mp4v_video(
        frames, filepath_save=Path("output.mp4"), fps=24
    )

Target Users:
    - Oceanographers analyzing underwater imagery
    - Marine researchers studying fish populations
    - Robotics engineers working with sonar systems

For command-line usage, see the scripts in the scripts/ directory.
"""


def hello() -> str:
    """
    Return a greeting message from the ARIS package.

    Returns:
        str: A simple greeting string.

    Note:
        This is a placeholder function for package testing.
    """
    return "Hello from aris!"
