"""
Integration tests for aris-extract-frame CLI command.

Tests the CLI command that extracts average frames from video files.
"""

import subprocess
from pathlib import Path

import cv2


class TestExtractAverageFrameScript:
    """Tests for aris-extract-frame CLI command."""

    def test_script_exists(self):
        """Test that the script module exists in the package."""
        script_path = Path("src/aris/scripts/extract_average_video_frame.py")
        assert script_path.exists()

    def test_extract_average_frame_basic(self, sample_reference_mp4, tmp_path):
        """Test basic average frame extraction from reference MP4."""
        output_path = tmp_path / "average_frame.jpg"

        # Run the CLI command
        result = subprocess.run(
            [
                "uv",
                "run",
                "aris-extract-frame",
                "--filepath-video",
                str(sample_reference_mp4),
                "--filepath-save",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )

        # Check command succeeded
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_extract_average_frame_with_max_frames(
        self, sample_reference_mp4, tmp_path
    ):
        """Test average frame extraction with max_frames limit."""
        output_path = tmp_path / "average_frame_limited.jpg"

        # Run with max_frames=10
        result = subprocess.run(
            [
                "uv",
                "run",
                "aris-extract-frame",
                "--filepath-video",
                str(sample_reference_mp4),
                "--filepath-save",
                str(output_path),
                "--max-frames",
                "10",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert output_path.exists()

    def test_output_is_valid_image(self, sample_reference_mp4, tmp_path):
        """Test that output file is a valid image that can be loaded."""
        output_path = tmp_path / "average_frame.jpg"

        # Run the CLI command
        subprocess.run(
            [
                "uv",
                "run",
                "aris-extract-frame",
                "--filepath-video",
                str(sample_reference_mp4),
                "--filepath-save",
                str(output_path),
                "--max-frames",
                "5",  # Use few frames to keep test fast
            ],
            check=True,
        )

        # Load the image and verify it's valid
        img = cv2.imread(str(output_path))
        assert img is not None
        assert img.shape[0] > 0  # Has height
        assert img.shape[1] > 0  # Has width
        assert img.shape[2] == 3  # RGB

    def test_creates_parent_directories(self, sample_reference_mp4, tmp_path):
        """Test that command creates parent directories if they don't exist."""
        output_path = tmp_path / "nested" / "dir" / "average_frame.jpg"

        result = subprocess.run(
            [
                "uv",
                "run",
                "aris-extract-frame",
                "--filepath-video",
                str(sample_reference_mp4),
                "--filepath-save",
                str(output_path),
                "--max-frames",
                "5",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert output_path.exists()
        assert output_path.parent.exists()

    def test_nonexistent_video_fails(self, tmp_path):
        """Test that command fails gracefully with non-existent video."""
        fake_video = tmp_path / "nonexistent.mp4"
        output_path = tmp_path / "output.jpg"

        result = subprocess.run(
            [
                "uv",
                "run",
                "aris-extract-frame",
                "--filepath-video",
                str(fake_video),
                "--filepath-save",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0
        assert not output_path.exists()

    def test_different_output_formats(self, sample_reference_mp4, tmp_path):
        """Test that command works with different image output formats."""
        formats = ["jpg", "png", "bmp"]

        for fmt in formats:
            output_path = tmp_path / f"average_frame.{fmt}"

            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "aris-extract-frame",
                    "--filepath-video",
                    str(sample_reference_mp4),
                    "--filepath-save",
                    str(output_path),
                    "--max-frames",
                    "5",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Failed for format {fmt}: {result.stderr}"
            assert output_path.exists(), f"Output not created for format {fmt}"

    def test_output_dimensions_match_video(self, sample_reference_mp4, tmp_path):
        """Test that extracted frame dimensions match source video."""
        output_path = tmp_path / "average_frame.jpg"

        # Get video dimensions
        cap = cv2.VideoCapture(str(sample_reference_mp4))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Extract average frame
        subprocess.run(
            [
                "uv",
                "run",
                "aris-extract-frame",
                "--filepath-video",
                str(sample_reference_mp4),
                "--filepath-save",
                str(output_path),
                "--max-frames",
                "5",
            ],
            check=True,
        )

        # Check dimensions
        img = cv2.imread(str(output_path))
        assert img.shape[1] == video_width  # Width
        assert img.shape[0] == video_height  # Height
