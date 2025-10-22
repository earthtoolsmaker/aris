"""
Unit tests for aris.frame module.

Tests the frame manipulation utilities including grayscale/RGB conversion,
timestamp extraction, and frame type detection.

Note: Tests for extract_frame_as_numpy_array() and extract_frames_as_numpy_arrays()
are excluded as they depend on pyARIS functionality.
"""

import datetime

import numpy as np
import pytz

from aris.frame import (
    aris_frames_to_mp4v_video,
    get_recorded_at_datetime,
    grayscale_to_rgb,
    is_grayscale,
)


class TestGrayscaleToRgb:
    """Tests for grayscale_to_rgb() function."""

    def test_converts_grayscale_to_rgb_shape(self, sample_grayscale_frame):
        """Test that grayscale conversion produces correct RGB shape (H, W, 3)."""
        rgb = grayscale_to_rgb(sample_grayscale_frame)
        assert rgb.shape == (100, 100, 3)

    def test_all_channels_identical(self, sample_grayscale_frame):
        """Test that all three RGB channels contain identical grayscale values."""
        rgb = grayscale_to_rgb(sample_grayscale_frame)
        assert np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
        assert np.array_equal(rgb[:, :, 1], rgb[:, :, 2])

    def test_preserves_original_values(self, sample_grayscale_frame):
        """Test that RGB channels match the original grayscale values."""
        rgb = grayscale_to_rgb(sample_grayscale_frame)
        assert np.array_equal(rgb[:, :, 0], sample_grayscale_frame)

    def test_handles_black_frame(self, empty_frame):
        """Test conversion of completely black frame."""
        rgb = grayscale_to_rgb(empty_frame)
        assert rgb.shape == (100, 100, 3)
        assert np.all(rgb == 0)

    def test_handles_white_frame(self, white_frame):
        """Test conversion of completely white frame."""
        rgb = grayscale_to_rgb(white_frame)
        assert rgb.shape == (100, 100, 3)
        assert np.all(rgb == 255)

    def test_preserves_dtype(self, sample_grayscale_frame):
        """Test that output dtype remains uint8."""
        rgb = grayscale_to_rgb(sample_grayscale_frame)
        assert rgb.dtype == np.uint8


class TestIsGrayscale:
    """Tests for is_grayscale() function."""

    def test_detects_grayscale_frame(self, sample_grayscale_frame):
        """Test that 2D arrays are detected as grayscale."""
        assert is_grayscale(sample_grayscale_frame) is True

    def test_detects_rgb_frame(self, sample_rgb_frame):
        """Test that 3D arrays are detected as RGB."""
        assert is_grayscale(sample_rgb_frame) is False

    def test_detects_converted_rgb_frame(self, sample_grayscale_frame):
        """Test that converted frames are detected as RGB."""
        rgb = grayscale_to_rgb(sample_grayscale_frame)
        assert is_grayscale(rgb) is False

    def test_handles_different_dimensions(self):
        """Test detection with various frame dimensions."""
        small_gray = np.zeros((10, 10), dtype=np.uint8)
        large_gray = np.zeros((1000, 1000), dtype=np.uint8)
        small_rgb = np.zeros((10, 10, 3), dtype=np.uint8)

        assert is_grayscale(small_gray) is True
        assert is_grayscale(large_gray) is True
        assert is_grayscale(small_rgb) is False


class TestGetRecordedAtDatetime:
    """Tests for get_recorded_at_datetime() function."""

    def test_returns_datetime_object(self, mock_aris_frame):
        """Test that function returns a datetime object."""
        dt = get_recorded_at_datetime(mock_aris_frame)
        assert isinstance(dt, datetime.datetime)

    def test_datetime_is_utc(self, mock_aris_frame):
        """Test that returned datetime is timezone-aware and in UTC."""
        dt = get_recorded_at_datetime(mock_aris_frame)
        assert dt.tzinfo is not None
        assert dt.tzinfo == pytz.UTC

    def test_known_timestamp_conversion(self, known_timestamp_frame):
        """Test conversion of a known timestamp to verify correctness."""
        dt = get_recorded_at_datetime(known_timestamp_frame)
        assert dt.year == 2024
        assert dt.month == 6
        assert dt.day == 15
        assert dt.hour == 14
        assert dt.minute == 30
        assert dt.second == 45

    def test_epoch_timestamp(self):
        """Test conversion of Unix epoch (1970-01-01 00:00:00 UTC)."""

        class EpochFrame:
            sonartimestamp = 0  # Unix epoch

        dt = get_recorded_at_datetime(EpochFrame())
        assert dt.year == 1970
        assert dt.month == 1
        assert dt.day == 1


class TestArisFramesToMp4vVideo:
    """Tests for aris_frames_to_mp4v_video() function."""

    def test_creates_video_file(self, sample_frames_list, tmp_path):
        """Test that video file is created at specified path."""
        output_path = tmp_path / "output.mp4"
        aris_frames_to_mp4v_video(sample_frames_list, output_path, fps=24)
        assert output_path.exists()

    def test_creates_parent_directories(self, sample_frames_list, tmp_path):
        """Test that parent directories are created if they don't exist."""
        output_path = tmp_path / "subdir" / "nested" / "output.mp4"
        aris_frames_to_mp4v_video(sample_frames_list, output_path, fps=24)
        assert output_path.exists()
        assert output_path.parent.exists()

    def test_handles_grayscale_frames(self, tmp_path):
        """Test that grayscale frames are converted and encoded correctly."""
        grayscale_frames = [
            np.random.randint(0, 256, (100, 100), dtype=np.uint8) for _ in range(3)
        ]
        output_path = tmp_path / "grayscale_video.mp4"
        aris_frames_to_mp4v_video(grayscale_frames, output_path, fps=24)
        assert output_path.exists()

    def test_handles_rgb_frames(self, sample_frames_list, tmp_path):
        """Test that RGB frames are encoded correctly."""
        output_path = tmp_path / "rgb_video.mp4"
        aris_frames_to_mp4v_video(sample_frames_list, output_path, fps=24)
        assert output_path.exists()

    def test_custom_fps(self, sample_frames_list, tmp_path):
        """Test that video is created with custom FPS."""
        output_path = tmp_path / "custom_fps.mp4"
        aris_frames_to_mp4v_video(sample_frames_list, output_path, fps=30)
        assert output_path.exists()

    def test_single_frame_video(self, sample_rgb_frame, tmp_path):
        """Test creating a video from a single frame."""
        output_path = tmp_path / "single_frame.mp4"
        aris_frames_to_mp4v_video([sample_rgb_frame], output_path, fps=1)
        assert output_path.exists()

    def test_mixed_grayscale_rgb_frames(self, tmp_path):
        """Test handling of mixed grayscale and RGB frames in same sequence."""
        mixed_frames = [
            np.random.randint(0, 256, (100, 100), dtype=np.uint8),  # Grayscale
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),  # RGB
            np.random.randint(0, 256, (100, 100), dtype=np.uint8),  # Grayscale
        ]
        output_path = tmp_path / "mixed_frames.mp4"
        # Note: This should work but all frames will be converted to RGB internally
        aris_frames_to_mp4v_video(mixed_frames, output_path, fps=24)
        assert output_path.exists()
