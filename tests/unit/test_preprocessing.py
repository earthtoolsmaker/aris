"""
Unit tests for aris.preprocessing module.

Tests preprocessing functions including Gaussian kernel creation, temporal smoothing,
and triple-channel visualization for sonar video processing.
"""

import numpy as np

from aris.preprocessing import (
    create_gaussian_kernel,
    create_visualization,
    smooth_frames_temporal,
)


class TestCreateGaussianKernel:
    """Tests for create_gaussian_kernel() function."""

    def test_returns_numpy_array(self):
        """Test that function returns a NumPy array."""
        kernel = create_gaussian_kernel(5, 1.0)
        assert isinstance(kernel, np.ndarray)

    def test_correct_dtype(self):
        """Test that kernel has float32 dtype."""
        kernel = create_gaussian_kernel(5, 1.0)
        assert kernel.dtype == np.float32

    def test_correct_length(self):
        """Test that kernel length matches window size."""
        for window_size in [3, 5, 7, 9]:
            kernel = create_gaussian_kernel(window_size, 1.0)
            assert len(kernel) == window_size

    def test_normalized_sum(self):
        """Test that kernel weights sum to 1.0."""
        kernel = create_gaussian_kernel(5, 1.0)
        assert np.isclose(np.sum(kernel), 1.0)

    def test_symmetric(self):
        """Test that kernel is symmetric around center."""
        kernel = create_gaussian_kernel(7, 1.0)
        assert np.allclose(kernel, kernel[::-1])

    def test_center_has_max_weight(self):
        """Test that center of kernel has the highest weight."""
        kernel = create_gaussian_kernel(9, 1.0)
        center_idx = len(kernel) // 2
        assert kernel[center_idx] == np.max(kernel)

    def test_different_sigma_values(self):
        """Test that larger sigma produces flatter distribution."""
        kernel_narrow = create_gaussian_kernel(9, 1.0)
        kernel_wide = create_gaussian_kernel(9, 3.0)

        # With larger sigma, center weight should be smaller (more distributed)
        center = len(kernel_narrow) // 2
        assert kernel_narrow[center] > kernel_wide[center]

    def test_weights_decay_from_center(self):
        """Test that weights monotonically decrease from center."""
        kernel = create_gaussian_kernel(11, 1.5)
        center = len(kernel) // 2

        # Check left half
        for i in range(center):
            assert kernel[i] <= kernel[i + 1]

        # Check right half
        for i in range(center + 1, len(kernel) - 1):
            assert kernel[i] >= kernel[i + 1]

    def test_all_positive_weights(self):
        """Test that all weights are positive."""
        kernel = create_gaussian_kernel(7, 2.0)
        assert np.all(kernel > 0)


class TestSmoothFramesTemporal:
    """Tests for smooth_frames_temporal() function."""

    def test_returns_numpy_array(self):
        """Test that function returns a NumPy array."""
        frames = [np.zeros((100, 100), dtype=np.uint8) for _ in range(5)]
        weights = np.ones(5, dtype=np.float32) / 5
        result = smooth_frames_temporal(frames, weights)
        assert isinstance(result, np.ndarray)

    def test_correct_dtype(self):
        """Test that output has uint8 dtype."""
        frames = [np.zeros((100, 100), dtype=np.uint8) for _ in range(3)]
        weights = np.ones(3, dtype=np.float32) / 3
        result = smooth_frames_temporal(frames, weights)
        assert result.dtype == np.uint8

    def test_correct_shape(self):
        """Test that output has same shape as input frames."""
        frames = [np.zeros((50, 80), dtype=np.uint8) for _ in range(5)]
        weights = np.ones(5, dtype=np.float32) / 5
        result = smooth_frames_temporal(frames, weights)
        assert result.shape == (50, 80)

    def test_uniform_weights_produces_average(self):
        """Test that uniform weights produce a simple average."""
        # Create frames with known values
        frames = [
            np.full((10, 10), 0, dtype=np.uint8),
            np.full((10, 10), 100, dtype=np.uint8),
            np.full((10, 10), 200, dtype=np.uint8),
        ]
        weights = np.ones(3, dtype=np.float32) / 3  # Uniform weights

        result = smooth_frames_temporal(frames, weights)
        expected_avg = (0 + 100 + 200) / 3  # = 100
        assert np.allclose(result, expected_avg, atol=1)

    def test_single_weight_selects_frame(self):
        """Test that weight of 1.0 on one frame returns that frame."""
        frames = [
            np.full((10, 10), 50, dtype=np.uint8),
            np.full((10, 10), 100, dtype=np.uint8),
            np.full((10, 10), 150, dtype=np.uint8),
        ]
        # Weight only the middle frame
        weights = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        result = smooth_frames_temporal(frames, weights)
        assert np.array_equal(result, frames[1])

    def test_weighted_average(self):
        """Test weighted averaging with non-uniform weights."""
        frames = [
            np.full((10, 10), 0, dtype=np.uint8),
            np.full((10, 10), 100, dtype=np.uint8),
        ]
        # 80% weight on second frame, 20% on first
        weights = np.array([0.2, 0.8], dtype=np.float32)

        result = smooth_frames_temporal(frames, weights)
        expected = 0.2 * 0 + 0.8 * 100  # = 80
        assert np.allclose(result, expected, atol=1)

    def test_preserves_value_range(self):
        """Test that output values stay in uint8 range [0, 255]."""
        frames = [np.full((10, 10), 255, dtype=np.uint8) for _ in range(5)]
        weights = np.ones(5, dtype=np.float32) / 5

        result = smooth_frames_temporal(frames, weights)
        assert np.all(result >= 0)
        assert np.all(result <= 255)

    def test_black_frames_stay_black(self):
        """Test that averaging black frames produces black output."""
        frames = [np.zeros((10, 10), dtype=np.uint8) for _ in range(3)]
        weights = np.ones(3, dtype=np.float32) / 3

        result = smooth_frames_temporal(frames, weights)
        assert np.array_equal(result, np.zeros((10, 10), dtype=np.uint8))

    def test_white_frames_stay_white(self):
        """Test that averaging white frames produces white output."""
        frames = [np.full((10, 10), 255, dtype=np.uint8) for _ in range(3)]
        weights = np.ones(3, dtype=np.float32) / 3

        result = smooth_frames_temporal(frames, weights)
        assert np.array_equal(result, np.full((10, 10), 255, dtype=np.uint8))


class TestCreateVisualization:
    """Tests for create_visualization() function."""

    def test_returns_numpy_array(self):
        """Test that function returns a NumPy array."""
        blurred = np.zeros((100, 100), dtype=np.uint8)
        edges = np.zeros((100, 100), dtype=np.uint8)
        preprocessed = np.zeros((100, 100), dtype=np.uint8)
        result = create_visualization(blurred, edges, preprocessed)
        assert isinstance(result, np.ndarray)

    def test_correct_shape(self):
        """Test that output is RGB with shape (H, W, 3)."""
        blurred = np.zeros((50, 80), dtype=np.uint8)
        edges = np.zeros((50, 80), dtype=np.uint8)
        preprocessed = np.zeros((50, 80), dtype=np.uint8)
        result = create_visualization(blurred, edges, preprocessed)
        assert result.shape == (50, 80, 3)

    def test_correct_dtype(self):
        """Test that output has uint8 dtype."""
        blurred = np.zeros((10, 10), dtype=np.uint8)
        edges = np.zeros((10, 10), dtype=np.uint8)
        preprocessed = np.zeros((10, 10), dtype=np.uint8)
        result = create_visualization(blurred, edges, preprocessed)
        assert result.dtype == np.uint8

    def test_blue_channel_is_blurred(self):
        """Test that blue channel (index 0) contains blurred frame."""
        blurred = np.full((10, 10), 100, dtype=np.uint8)
        edges = np.full((10, 10), 50, dtype=np.uint8)
        preprocessed = np.full((10, 10), 200, dtype=np.uint8)
        result = create_visualization(blurred, edges, preprocessed)
        assert np.array_equal(result[:, :, 0], blurred)

    def test_green_channel_is_edges(self):
        """Test that green channel (index 1) contains edge intersection."""
        blurred = np.full((10, 10), 100, dtype=np.uint8)
        edges = np.full((10, 10), 150, dtype=np.uint8)
        preprocessed = np.full((10, 10), 200, dtype=np.uint8)
        result = create_visualization(blurred, edges, preprocessed)
        assert np.array_equal(result[:, :, 1], edges)

    def test_red_channel_is_preprocessed(self):
        """Test that red channel (index 2) contains preprocessed frame."""
        blurred = np.full((10, 10), 100, dtype=np.uint8)
        edges = np.full((10, 10), 50, dtype=np.uint8)
        preprocessed = np.full((10, 10), 200, dtype=np.uint8)
        result = create_visualization(blurred, edges, preprocessed)
        assert np.array_equal(result[:, :, 2], preprocessed)

    def test_black_inputs_produce_black_output(self):
        """Test that black inputs produce black RGB output."""
        blurred = np.zeros((10, 10), dtype=np.uint8)
        edges = np.zeros((10, 10), dtype=np.uint8)
        preprocessed = np.zeros((10, 10), dtype=np.uint8)
        result = create_visualization(blurred, edges, preprocessed)
        assert np.array_equal(result, np.zeros((10, 10, 3), dtype=np.uint8))

    def test_visual_color_interpretation(self):
        """Test color interpretation for visual validation."""
        # Pure blue: blurred=255, edges=0, preprocessed=0
        blurred = np.full((5, 5), 255, dtype=np.uint8)
        edges = np.zeros((5, 5), dtype=np.uint8)
        preprocessed = np.zeros((5, 5), dtype=np.uint8)
        result = create_visualization(blurred, edges, preprocessed)
        # Should be [255, 0, 0] = blue in BGR
        assert np.array_equal(result[0, 0], [255, 0, 0])

        # Pure green: blurred=0, edges=255, preprocessed=0
        blurred2 = np.zeros((5, 5), dtype=np.uint8)
        edges2 = np.full((5, 5), 255, dtype=np.uint8)
        preprocessed2 = np.zeros((5, 5), dtype=np.uint8)
        result2 = create_visualization(blurred2, edges2, preprocessed2)
        # Should be [0, 255, 0] = green in BGR
        assert np.array_equal(result2[0, 0], [0, 255, 0])

        # Pure red: blurred=0, edges=0, preprocessed=255
        blurred3 = np.zeros((5, 5), dtype=np.uint8)
        edges3 = np.zeros((5, 5), dtype=np.uint8)
        preprocessed3 = np.full((5, 5), 255, dtype=np.uint8)
        result3 = create_visualization(blurred3, edges3, preprocessed3)
        # Should be [0, 0, 255] = red in BGR
        assert np.array_equal(result3[0, 0], [0, 0, 255])

        # Yellow (green + red): blurred=0, edges=255, preprocessed=255
        blurred4 = np.zeros((5, 5), dtype=np.uint8)
        edges4 = np.full((5, 5), 255, dtype=np.uint8)
        preprocessed4 = np.full((5, 5), 255, dtype=np.uint8)
        result4 = create_visualization(blurred4, edges4, preprocessed4)
        # Should be [0, 255, 255] = yellow in BGR
        assert np.array_equal(result4[0, 0], [0, 255, 255])
