import logging
import subprocess
from pathlib import Path

def encode_video_with_h264_codec(filepath_input: Path, filepath_output: Path):
    """
    Encode the video with h264 codec.
    """

    assert filepath_input.exists(), "filepath_input does not exist!"

    command = [
        "ffmpeg",
        "-i",
        str(filepath_input),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        str(filepath_output),
    ]

    logging.info(f"Command to run: {command}")

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logging.info("Command executed successfully.")

    except subprocess.CalledProcessError as e:
        logging.error("An error occurred while executing the command.")
        logging.error("Error message:", e.stderr.decode())
