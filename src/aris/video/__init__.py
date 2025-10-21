"""
Video processing utilities for ARIS sonar data.

This subpackage provides tools for video encoding, frame extraction, and metadata
handling for sonar videos. It supports various video codecs and offers utilities
for batch processing video files.

Modules:
    utils: Core video processing functions including:
        - H.264 and MPEG-4 encoding
        - Average frame extraction
        - Video metadata retrieval (FPS, duration, frame count)
        - Frame sequence to video conversion
        - Background subtraction for motion detection

Key Functions (available in aris.video.utils):
    - encode_video_with_h264_codec(): Re-encode videos with H.264 codec
    - get_average_frame(): Extract temporal average of video frames
    - get_all_frames(): Extract all frames from a video file
    - save_frames_to_video(): Create MP4 from frame list
    - get_fps(): Retrieve video frame rate
    - get_video_duration(): Get video duration in seconds

Example:
    from aris.video.utils import get_average_frame, encode_video_with_h264_codec

    # Extract average frame for thumbnail
    avg_frame = get_average_frame(Path("sonar.mp4"), max_frames=100)

    # Re-encode with H.264 for compatibility
    encode_video_with_h264_codec(
        filepath_input=Path("input.mp4"),
        filepath_output=Path("output_h264.mp4")
    )

See Also:
    - aris.frame: For ARIS-specific frame extraction and video generation
    - aris.preprocessing: For video stabilization and preprocessing
"""
