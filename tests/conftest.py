"""
Shared pytest fixtures for ARIS test suite.

This module provides reusable fixtures for testing ARIS sonar processing toolkit,
including synthetic frames, mock ARIS objects, and test video files.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_grayscale_frame():
    """
    Create a synthetic 100x100 grayscale frame.

    Returns:
        NDArray[np.uint8]: Random grayscale image for testing.
    """
    return np.random.randint(0, 256, (100, 100), dtype=np.uint8)


@pytest.fixture
def sample_rgb_frame():
    """
    Create a synthetic 100x100 RGB frame.

    Returns:
        NDArray[np.uint8]: Random RGB image for testing.
    """
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_frames_list():
    """
    Create a list of 5 synthetic RGB frames for video generation tests.

    Returns:
        list[NDArray[np.uint8]]: List of 5 random RGB frames.
    """
    return [np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) for _ in range(5)]


@pytest.fixture
def sample_video_file(tmp_path):
    """
    Create a small test video file with 10 frames at 24 FPS.

    Args:
        tmp_path: pytest tmp_path fixture providing temporary directory.

    Returns:
        Path: Path to the generated test video file.
    """
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 24, (100, 100))

    # Generate 10 frames with different patterns for testing
    for i in range(10):
        frame = np.full((100, 100, 3), i * 25, dtype=np.uint8)  # Gradient pattern
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def sample_video_file_30fps(tmp_path):
    """
    Create a test video file with 30 frames at 30 FPS (1 second duration).

    Useful for testing duration calculations and FPS detection.

    Args:
        tmp_path: pytest tmp_path fixture providing temporary directory.

    Returns:
        Path: Path to the generated test video file.
    """
    video_path = tmp_path / "test_video_30fps.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30, (100, 100))

    for _i in range(30):
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def mock_aris_frame():
    """
    Create a mock ARIS frame object with basic attributes.

    Simulates the pyARIS.ARIS_Frame structure without requiring actual ARIS files.
    Timestamp corresponds to 2021-01-01 00:00:00 UTC (1609459200 seconds).

    Returns:
        MockARISFrame: Mock object with sonartimestamp and remap attributes.
    """

    class MockARISFrame:
        sonartimestamp = 1609459200000000  # 2021-01-01 00:00:00 UTC in microseconds
        remap = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    return MockARISFrame()


@pytest.fixture
def known_timestamp_frame():
    """
    Create a mock ARIS frame with a known, specific timestamp for precise testing.

    Timestamp: 2024-06-15 14:30:45 UTC

    Returns:
        MockARISFrame: Mock frame with specific timestamp.
    """

    class MockARISFrame:
        # 2024-06-15 14:30:45 UTC = 1718461845 seconds = 1718461845000000 microseconds
        sonartimestamp = 1718461845000000
        remap = np.zeros((50, 50), dtype=np.uint8)

    return MockARISFrame()


@pytest.fixture
def empty_frame():
    """
    Create a completely black (all zeros) grayscale frame.

    Useful for testing edge cases and averaging algorithms.

    Returns:
        NDArray[np.uint8]: 100x100 black frame.
    """
    return np.zeros((100, 100), dtype=np.uint8)


@pytest.fixture
def white_frame():
    """
    Create a completely white (all 255s) grayscale frame.

    Useful for testing edge cases and value range handling.

    Returns:
        NDArray[np.uint8]: 100x100 white frame.
    """
    return np.full((100, 100), 255, dtype=np.uint8)


@pytest.fixture
def fixtures_dir():
    """
    Return path to test fixtures directory containing real ARIS and MP4 files.

    Returns:
        Path: Absolute path to tests/fixtures/ directory.
    """
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_aris_file(fixtures_dir):
    """
    Return path to sample ARIS sonar file for integration testing.

    This file is tracked with Git LFS and should be ~63MB.

    Returns:
        Path: Path to sample ARIS file.
    """
    return fixtures_dir / "aris" / "2025-05-13_092300.aris"


@pytest.fixture
def sample_reference_mp4(fixtures_dir):
    """
    Return path to reference MP4 video file for integration testing.

    This is the expected output from converting the sample ARIS file.
    Tracked with Git LFS, should be ~65MB.

    Returns:
        Path: Path to reference MP4 file.
    """
    return fixtures_dir / "mp4" / "2025-05-13_092300.mp4"
