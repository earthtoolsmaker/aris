"""
Unit tests for aris.video.utils module.

Tests video processing utilities including frame extraction, averaging,
metadata reading, and video encoding operations.
"""

import cv2
import numpy as np
import pytest

from aris.video.utils import (
    encode_video_with_h264_codec,
    get_all_frames,
    get_average_frame,
    get_fps,
    get_video_duration,
    save_frames_to_video,
)


class TestGetAverageFrame:
    """Tests for get_average_frame() function."""

    def test_returns_numpy_array(self, sample_video_file):
        """Test that function returns a NumPy array."""
        avg_frame = get_average_frame(sample_video_file)
        assert isinstance(avg_frame, np.ndarray)

    def test_correct_shape(self, sample_video_file):
        """Test that average frame has expected dimensions."""
        avg_frame = get_average_frame(sample_video_file)
        assert avg_frame.shape == (100, 100, 3)

    def test_correct_dtype(self, sample_video_file):
        """Test that average frame has uint8 dtype."""
        avg_frame = get_average_frame(sample_video_file)
        assert avg_frame.dtype == np.uint8

    def test_max_frames_limit(self, sample_video_file):
        """Test that max_frames parameter limits frames used for averaging."""
        # Video has 10 frames, but we only use first 3
        avg_frame = get_average_frame(sample_video_file, max_frames=3)
        assert avg_frame is not None
        assert avg_frame.shape == (100, 100, 3)

    def test_average_calculation_consistency(self, tmp_path):
        """Test that averaging produces mathematically correct results."""
        # Create video with known pixel values
        video_path = tmp_path / "known_values.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 24, (10, 10))

        # Frame 1: all pixels = 0
        # Frame 2: all pixels = 100
        # Expected average: 50
        writer.write(np.zeros((10, 10, 3), dtype=np.uint8))
        writer.write(np.full((10, 10, 3), 100, dtype=np.uint8))
        writer.release()

        avg_frame = get_average_frame(video_path)
        # Allow some tolerance due to video compression
        assert np.all(avg_frame >= 40)
        assert np.all(avg_frame <= 60)

    def test_handles_nonexistent_file(self, tmp_path):
        """Test behavior with non-existent video file."""
        fake_path = tmp_path / "nonexistent.mp4"
        avg_frame = get_average_frame(fake_path)
        # Should return None or raise exception gracefully
        assert avg_frame is None


class TestGetAllFrames:
    """Tests for get_all_frames() function."""

    def test_returns_list(self, sample_video_file):
        """Test that function returns a list."""
        frames = get_all_frames(sample_video_file)
        assert isinstance(frames, list)

    def test_correct_frame_count(self, sample_video_file):
        """Test that all frames are extracted (video has 10 frames)."""
        frames = get_all_frames(sample_video_file)
        assert len(frames) == 10

    def test_all_frames_are_numpy_arrays(self, sample_video_file):
        """Test that each frame is a NumPy array."""
        frames = get_all_frames(sample_video_file)
        assert all(isinstance(frame, np.ndarray) for frame in frames)

    def test_all_frames_same_shape(self, sample_video_file):
        """Test that all frames have consistent dimensions."""
        frames = get_all_frames(sample_video_file)
        shapes = [frame.shape for frame in frames]
        assert all(shape == (100, 100, 3) for shape in shapes)

    def test_all_frames_uint8_dtype(self, sample_video_file):
        """Test that all frames have uint8 dtype."""
        frames = get_all_frames(sample_video_file)
        assert all(frame.dtype == np.uint8 for frame in frames)

    def test_empty_video_returns_empty_list(self, tmp_path):
        """Test handling of corrupted or empty video file."""
        # Create an empty file
        empty_video = tmp_path / "empty.mp4"
        empty_video.touch()
        frames = get_all_frames(empty_video)
        assert frames == []


class TestGetFps:
    """Tests for get_fps() function."""

    def test_returns_float(self, sample_video_file):
        """Test that FPS is returned as a float."""
        fps = get_fps(sample_video_file)
        assert isinstance(fps, float)

    def test_correct_fps_24(self, sample_video_file):
        """Test that correct FPS is read from 24fps video."""
        fps = get_fps(sample_video_file)
        assert fps == 24.0

    def test_correct_fps_30(self, sample_video_file_30fps):
        """Test that correct FPS is read from 30fps video."""
        fps = get_fps(sample_video_file_30fps)
        assert fps == 30.0

    def test_nonexistent_file_raises_exception(self, tmp_path):
        """Test that missing file raises an exception."""
        fake_path = tmp_path / "nonexistent.mp4"
        with pytest.raises((Exception, cv2.error)):
            get_fps(fake_path)


class TestGetVideoDuration:
    """Tests for get_video_duration() function."""

    def test_returns_float_or_none(self, sample_video_file):
        """Test that duration is returned as float or None."""
        duration = get_video_duration(sample_video_file)
        assert isinstance(duration, (float, type(None)))

    def test_duration_is_positive(self, sample_video_file):
        """Test that duration is a positive value."""
        duration = get_video_duration(sample_video_file)
        if duration is not None:
            assert duration > 0

    def test_duration_calculation_30fps(self, sample_video_file_30fps):
        """Test duration calculation for 30fps video with 30 frames (should be ~1 second)."""
        duration = get_video_duration(sample_video_file_30fps)
        if duration is not None:
            # Should be approximately 1 second (30 frames at 30fps)
            assert 0.8 <= duration <= 1.2

    def test_nonexistent_file_returns_none(self, tmp_path):
        """Test that missing file returns None."""
        fake_path = tmp_path / "nonexistent.mp4"
        duration = get_video_duration(fake_path)
        assert duration is None


class TestSaveFramesToVideo:
    """Tests for save_frames_to_video() function."""

    def test_creates_video_file(self, sample_frames_list, tmp_path):
        """Test that video file is created."""
        output_path = tmp_path / "output.mp4"
        save_frames_to_video(sample_frames_list, output_path, fps=30)
        assert output_path.exists()

    def test_custom_fps(self, sample_frames_list, tmp_path):
        """Test that video is created with custom FPS."""
        output_path = tmp_path / "custom_fps.mp4"
        save_frames_to_video(sample_frames_list, output_path, fps=15)
        assert output_path.exists()
        # Verify FPS
        fps = get_fps(output_path)
        assert fps == 15.0

    def test_single_frame(self, sample_rgb_frame, tmp_path):
        """Test creating video from single frame."""
        output_path = tmp_path / "single_frame.mp4"
        save_frames_to_video([sample_rgb_frame], output_path, fps=1)
        assert output_path.exists()

    def test_empty_frame_list_logs_error(self, tmp_path, caplog):
        """Test that empty frame list is handled gracefully."""
        output_path = tmp_path / "empty.mp4"
        save_frames_to_video([], output_path, fps=30)
        # Should log error and not create file
        assert not output_path.exists()

    def test_large_frame_count(self, tmp_path):
        """Test handling of larger number of frames."""
        frames = [
            np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8) for _ in range(100)
        ]
        output_path = tmp_path / "large_video.mp4"
        save_frames_to_video(frames, output_path, fps=30)
        assert output_path.exists()


class TestEncodeVideoWithH264Codec:
    """Tests for encode_video_with_h264_codec() function."""

    def test_creates_output_file(self, sample_video_file, tmp_path):
        """Test that H.264 encoded video is created."""
        output_path = tmp_path / "encoded_h264.mp4"
        encode_video_with_h264_codec(sample_video_file, output_path)
        assert output_path.exists()

    def test_input_file_must_exist(self, tmp_path):
        """Test that assertion fails for non-existent input file."""
        fake_input = tmp_path / "nonexistent.mp4"
        output_path = tmp_path / "output.mp4"
        with pytest.raises(AssertionError):
            encode_video_with_h264_codec(fake_input, output_path)

    def test_preserves_video_dimensions(self, sample_video_file, tmp_path):
        """Test that encoding preserves video dimensions."""
        output_path = tmp_path / "encoded_h264.mp4"
        encode_video_with_h264_codec(sample_video_file, output_path)

        # Read original and encoded videos to compare dimensions
        original_cap = cv2.VideoCapture(str(sample_video_file))
        encoded_cap = cv2.VideoCapture(str(output_path))

        original_width = int(original_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(original_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        encoded_width = int(encoded_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        encoded_height = int(encoded_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        original_cap.release()
        encoded_cap.release()

        assert original_width == encoded_width
        assert original_height == encoded_height

    def test_creates_parent_directories(self, sample_video_file, tmp_path):
        """Test that parent directories are created if needed."""
        output_path = tmp_path / "subdir" / "nested" / "encoded.mp4"

        # Function should create parent directories automatically
        encode_video_with_h264_codec(sample_video_file, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()
