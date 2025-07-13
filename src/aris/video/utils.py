import logging
from pathlib import Path

import ffmpeg

def encode_video_with_h264_codec(filepath_input: Path, filepath_output: Path):
    """
    Encode the video with h264 codec.
    """

    assert filepath_input.exists(), "filepath_input does not exist!"

    logging.info(f"Encoding video from {filepath_input} to {filepath_output}")

    try:
        (
            ffmpeg
            .input(str(filepath_input))
            .output(str(filepath_output), vcodec='libx264', preset='medium')
            .run(capture_stdout=True, capture_stderr=True)
        )
        logging.info("Video encoded successfully.")

    except ffmpeg.Error as e:
        logging.error("An error occurred while encoding the video.")
        logging.error("Error message:", e.stderr.decode())
