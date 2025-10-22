"""
Sonar video preprocessing and stabilization functions.

This module provides reusable functions for:
- Temporal stabilization (bidirectional Gaussian smoothing)
- Motion detection preprocessing (MOG2, guided filter)
- Dual-channel visualization

These functions are used by:
- stabilize_sonar_video.py
- preprocess_sonar_video.py
- stabilize_and_preprocess_sonar_video.py
"""

import cv2
import numpy as np
from numpy.typing import NDArray


def create_gaussian_kernel(window_size: int, sigma: float) -> NDArray[np.float32]:
    """
    Create a 1D Gaussian kernel for temporal weighting.

    The kernel is centered at the middle of the window, with weights decaying
    based on temporal distance according to a Gaussian function. Weights are
    normalized to sum to 1.0.

    Args:
        window_size: Size of the temporal window (must be odd)
        sigma: Standard deviation of the Gaussian function

    Returns:
        NDArray[np.float32]: 1D array of normalized Gaussian weights

    Example:
        >>> kernel = create_gaussian_kernel(5, 1.0)
        >>> # Returns weights for [t-2, t-1, t, t+1, t+2]
        >>> # with highest weight at center (t)
    """
    center = window_size // 2
    weights = np.array(
        [np.exp(-((i - center) ** 2) / (2 * sigma**2)) for i in range(window_size)],
        dtype=np.float32,
    )
    # Normalize so weights sum to 1
    return weights / np.sum(weights)


def smooth_frames_temporal(
    frames: list[NDArray[np.uint8]], weights: NDArray[np.float32]
) -> NDArray[np.uint8]:
    """
    Apply Gaussian weighted temporal smoothing to a window of frames.

    Computes a weighted average of frames using the provided Gaussian weights.
    This reduces temporal noise while preserving important features.

    Args:
        frames: List of frames in the temporal window
        weights: Gaussian weights for each frame (must have same length as frames)

    Returns:
        NDArray[np.uint8]: Smoothed frame

    Note:
        Weights should already be normalized to sum to 1.0
    """
    # Convert to float for accurate weighted averaging
    smoothed = np.zeros_like(frames[0], dtype=np.float32)

    for frame, weight in zip(frames, weights, strict=True):
        smoothed += weight * frame.astype(np.float32)

    # Convert back to uint8
    return smoothed.astype(np.uint8)


def preprocess_frame(
    frame: NDArray[np.uint8],
    bg_subtractor: cv2.BackgroundSubtractorMOG2,
    gaussian_kernel: int,
    gaussian_sigma: float,
    guided_radius: int,
    guided_eps: float,
    frame_history: NDArray[np.uint8] | None,
    frame_count: int,
    temporal_weight: float,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
    """
    Process a single frame with preprocessing pipeline.

    1. Apply Gaussian blur to reduce noise in sonar imagery
    2. Apply MOG2 background subtraction to isolate moving objects
    3. Apply guided filter to refine MOG2 mask using frame edge information
    4. Apply temporal smoothing to reduce frame-to-frame noise

    Args:
        frame: Input frame (grayscale)
        bg_subtractor: MOG2 background subtractor instance
        gaussian_kernel: Gaussian blur kernel size (must be odd)
        gaussian_sigma: Gaussian blur sigma value
        guided_radius: Guided filter radius
        guided_eps: Guided filter epsilon (regularization)
        frame_history: Previous processed frame for temporal smoothing (None for first frame)
        frame_count: Current frame number (0-indexed)
        temporal_weight: Weight for current frame in temporal blending (0-1)

    Returns:
        tuple: (blurred_frame, preprocessed_frame, preprocessed_frame_for_next_history)
            - blurred_frame: Output of step 1 (for blue channel visualization)
            - preprocessed_frame: Output of full pipeline (for red channel visualization)
            - preprocessed_frame_for_next_history: Same as preprocessed_frame, for history tracking
    """
    # Step 1: Apply Gaussian blur to reduce noise
    # This is critical for sonar imagery which has significant speckle noise
    frame_blurred = cv2.GaussianBlur(
        frame, (gaussian_kernel, gaussian_kernel), gaussian_sigma
    )

    # Step 2: Apply MOG2 background subtraction
    # This isolates moving objects (fish) from the static background
    mog_mask = bg_subtractor.apply(frame_blurred)

    # Step 3: Apply guided filter
    # Convert grayscale MOG mask to RGB for guided filter
    mog_mask_rgb = cv2.cvtColor(mog_mask, cv2.COLOR_GRAY2RGB)

    # Apply guided filter: use original frame as guide to refine MOG mask
    # This preserves edges from the original frame while smoothing the MOG mask
    # Requires opencv-contrib-python for cv2.ximgproc.guidedFilter
    guided_mog_rgb = cv2.ximgproc.guidedFilter(
        guide=frame_blurred,
        src=mog_mask_rgb,
        radius=guided_radius,
        eps=guided_eps,
    )

    # Convert back to grayscale
    guided_mog = cv2.cvtColor(guided_mog_rgb, cv2.COLOR_BGR2GRAY)

    # Step 4: Apply temporal smoothing
    # Blend current frame with history to reduce temporal noise
    if frame_count < 2 or frame_history is None:
        # For first couple frames, no history available
        processed_frame = guided_mog
    else:
        # Blend: temporal_weight * current + (1-temporal_weight) * history
        # Default: 0.8 * current + 0.2 * history
        processed_frame = (
            temporal_weight * guided_mog.astype(np.float32)
            + (1 - temporal_weight) * frame_history.astype(np.float32)
        ).astype(np.uint8)

    return frame_blurred, processed_frame, processed_frame


def create_dual_channel_visualization(
    frame_blurred: NDArray[np.uint8], frame_preprocessed: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    """
    Create RGB dual-channel visualization for sonar preprocessing.

    Combines the Gaussian-blurred input and preprocessed output into a single
    RGB frame for easy visual comparison.

    Channel assignment:
    - Blue channel: Gaussian-blurred input (shows sonar structure after noise reduction)
    - Green channel: Empty (zeros/black)
    - Red channel: Preprocessed output (shows detected motion)

    Visual interpretation:
    - Pure blue regions: Static sonar background
    - Pure red regions: Detected moving objects (e.g., fish)
    - Magenta/purple regions: Motion overlapping with sonar structures
    - Black regions: No signal in either channel

    Args:
        frame_blurred: Gaussian-blurred input frame (grayscale)
        frame_preprocessed: Preprocessed output frame (grayscale)

    Returns:
        NDArray[np.uint8]: RGB frame with shape (H, W, 3)
    """
    return np.dstack(
        [
            frame_blurred,  # Blue channel
            np.zeros_like(frame_blurred),  # Green channel (empty)
            frame_preprocessed,  # Red channel
        ]
    )
