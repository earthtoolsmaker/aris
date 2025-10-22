"""
Integration tests for ARIS file to MP4 video conversion pipeline.

Tests the end-to-end workflow of loading ARIS sonar files and converting them
to MP4 videos, validating against reference outputs.

Note: These tests require Git LFS files to be pulled (see tests/README.md).
"""

import pytest

from aris.frame import aris_frames_to_mp4v_video, extract_frames_as_numpy_arrays
from aris.pyARIS import pyARIS
from aris.video.utils import get_all_frames, get_fps


class TestArisFileLoading:
    """Tests for loading ARIS files."""

    def test_aris_file_exists(self, sample_aris_file):
        """Test that sample ARIS file exists and is accessible."""
        assert sample_aris_file.exists()
        assert sample_aris_file.stat().st_size > 1_000_000  # Should be > 1MB

    def test_aris_file_loads_successfully(self, sample_aris_file):
        """Test that ARIS file can be loaded by pyARIS parser."""
        aris_data, first_frame = pyARIS.DataImport(str(sample_aris_file))
        assert aris_data is not None
        assert first_frame is not None

    def test_aris_file_has_frames(self, sample_aris_file):
        """Test that loaded ARIS file contains frames."""
        aris_data, _ = pyARIS.DataImport(str(sample_aris_file))
        assert aris_data.FrameCount > 0

    def test_aris_file_metadata(self, sample_aris_file):
        """Test that ARIS file metadata is readable."""
        aris_data, _ = pyARIS.DataImport(str(sample_aris_file))
        # Check common metadata fields
        assert aris_data.NumRawBeams in [48, 64, 96, 128]  # Valid beam counts
        # Note: SampleRate and FrameRate may be 0 in file header (obsolete fields)
        # Modern ARIS files store this info in frame headers instead


class TestReferenceMP4:
    """Tests for reference MP4 file."""

    def test_reference_mp4_exists(self, sample_reference_mp4):
        """Test that reference MP4 file exists."""
        assert sample_reference_mp4.exists()
        assert sample_reference_mp4.stat().st_size > 1_000_000  # Should be > 1MB

    def test_reference_mp4_has_valid_fps(self, sample_reference_mp4):
        """Test that reference MP4 has valid frame rate."""
        fps = get_fps(sample_reference_mp4)
        assert fps > 0
        assert fps <= 60  # Reasonable upper bound

    def test_reference_mp4_has_frames(self, sample_reference_mp4):
        """Test that reference MP4 contains frames."""
        # Just check first few frames to keep test fast
        frames = get_all_frames(sample_reference_mp4)
        assert len(frames) > 0


class TestArisToMP4Conversion:
    """Tests for ARIS to MP4 conversion pipeline."""

    def test_extract_single_frame(self, sample_aris_file):
        """Test extracting a single frame from ARIS file."""
        aris_data, _ = pyARIS.DataImport(str(sample_aris_file))
        frames = extract_frames_as_numpy_arrays(aris_data, 0, 1, skip_frame=0)

        assert len(frames) == 1
        assert frames[0] is not None
        assert frames[0].shape[0] > 0  # Has height
        assert frames[0].shape[1] > 0  # Has width

    def test_extract_multiple_frames(self, sample_aris_file):
        """Test extracting multiple frames from ARIS file."""
        aris_data, _ = pyARIS.DataImport(str(sample_aris_file))
        num_frames = 10
        frames = extract_frames_as_numpy_arrays(aris_data, 0, num_frames, skip_frame=0)

        assert len(frames) == num_frames
        assert all(f is not None for f in frames)

    def test_all_frames_same_shape(self, sample_aris_file):
        """Test that all extracted frames have consistent dimensions."""
        aris_data, _ = pyARIS.DataImport(str(sample_aris_file))
        frames = extract_frames_as_numpy_arrays(aris_data, 0, 5, skip_frame=0)

        shapes = [f.shape for f in frames]
        assert all(shape == shapes[0] for shape in shapes)

    def test_convert_to_mp4(self, sample_aris_file, tmp_path):
        """Test full ARIS to MP4 conversion pipeline."""
        # Load ARIS file
        aris_data, _ = pyARIS.DataImport(str(sample_aris_file))

        # Extract first 10 frames (keep test fast)
        frames = extract_frames_as_numpy_arrays(aris_data, 0, 10, skip_frame=0)

        # Convert to MP4
        output_path = tmp_path / "test_output.mp4"
        aris_frames_to_mp4v_video(frames, output_path, fps=24)

        # Verify output
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_converted_video_has_correct_fps(self, sample_aris_file, tmp_path):
        """Test that converted video has requested FPS."""
        aris_data, _ = pyARIS.DataImport(str(sample_aris_file))
        frames = extract_frames_as_numpy_arrays(aris_data, 0, 5, skip_frame=0)

        output_path = tmp_path / "test_fps.mp4"
        target_fps = 30
        aris_frames_to_mp4v_video(frames, output_path, fps=target_fps)

        actual_fps = get_fps(output_path)
        assert actual_fps == target_fps

    def test_converted_video_frame_count(self, sample_aris_file, tmp_path):
        """Test that converted video has correct number of frames."""
        aris_data, _ = pyARIS.DataImport(str(sample_aris_file))
        num_frames = 5
        frames = extract_frames_as_numpy_arrays(aris_data, 0, num_frames, skip_frame=0)

        output_path = tmp_path / "test_count.mp4"
        aris_frames_to_mp4v_video(frames, output_path, fps=24)

        # Read back the frames
        output_frames = get_all_frames(output_path)
        assert len(output_frames) == num_frames


@pytest.mark.slow
class TestFullConversionWorkflow:
    """
    Tests for full-scale conversion workflows.

    These tests process more frames and may take longer to run.
    Mark as slow to allow skipping in quick test runs.
    """

    def test_convert_100_frames(self, sample_aris_file, tmp_path):
        """Test converting 100 frames from ARIS to MP4."""
        aris_data, _ = pyARIS.DataImport(str(sample_aris_file))

        # Extract 100 frames
        max_frames = min(100, aris_data.FrameCount)
        frames = extract_frames_as_numpy_arrays(aris_data, 0, max_frames, skip_frame=0)

        # Convert to MP4
        output_path = tmp_path / "test_100_frames.mp4"
        aris_frames_to_mp4v_video(frames, output_path, fps=24)

        # Verify
        assert output_path.exists()
        output_frames = get_all_frames(output_path)
        assert len(output_frames) == max_frames

    def test_compare_frame_dimensions_with_reference(
        self, sample_aris_file, sample_reference_mp4
    ):
        """Test that converted frames have similar dimensions to reference MP4."""
        # Get reference dimensions
        ref_frames = get_all_frames(sample_reference_mp4)
        ref_height, ref_width = ref_frames[0].shape[:2]

        # Convert ARIS file
        aris_data, _ = pyARIS.DataImport(str(sample_aris_file))
        frames = extract_frames_as_numpy_arrays(aris_data, 0, 5, skip_frame=0)

        # Check dimensions are similar (within 2 pixels tolerance)
        # Small differences can occur due to encoding/processing variations
        for frame in frames:
            assert abs(frame.shape[0] - ref_height) <= 2, (
                f"Height mismatch: {frame.shape[0]} vs {ref_height}"
            )
            assert abs(frame.shape[1] - ref_width) <= 2, (
                f"Width mismatch: {frame.shape[1]} vs {ref_width}"
            )
