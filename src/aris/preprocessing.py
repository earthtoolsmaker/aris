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

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class PreprocessingResult:
    """
    Result of preprocessing a single sonar frame.

    This dataclass encapsulates the outputs of the preprocessing pipeline,
    providing clear names for each component of the result.

    Attributes:
        blurred: Gaussian-blurred input frame (used for blue channel visualization)
        edges: Edge intersection mask showing high-confidence fish boundaries
               (used for green channel visualization)
        motion: Refined motion detection result after guided filtering and temporal
                smoothing (used for red channel visualization)
        history: Copy of motion result to be used as history for the next frame's
                 temporal smoothing
    """

    blurred: NDArray[np.uint8]
    edges: NDArray[np.uint8]
    motion: NDArray[np.uint8]
    history: NDArray[np.uint8]


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
    frame_history: NDArray[np.uint8] | None,
    frame_count: int,
    gaussian_kernel: int = 3,
    gaussian_sigma: float = 1.4,
    guided_radius: int = 10,
    guided_eps: float = 0.01,
    temporal_weight: float = 0.8,
    edge_canny_low: int = 200,
    edge_canny_high: int = 255,
    edge_dilation_size: int = 2,
) -> PreprocessingResult:
    """
    Process a single frame with preprocessing pipeline.

    1. Apply Gaussian blur to reduce noise in sonar imagery
    2. Apply MOG2 background subtraction to isolate moving objects
    3. Apply bidirectional guided filtering (frame-guided MOG and MOG-guided frame)
    4. Apply edge detection to both guided outputs (Canny)
    5. Compute edge intersection for high-confidence fish boundaries
    6. Apply temporal smoothing to reduce frame-to-frame noise

    Args:
        frame: Input frame (grayscale)
        bg_subtractor: MOG2 background subtractor instance
        frame_history: Previous processed frame for temporal smoothing (None for first frame)
        frame_count: Current frame number (0-indexed)
        gaussian_kernel: Gaussian blur kernel size (must be odd, default: 3)
        gaussian_sigma: Gaussian blur sigma value (default: 1.4)
        guided_radius: Guided filter radius (default: 10)
        guided_eps: Guided filter epsilon/regularization (default: 0.01)
        temporal_weight: Weight for current frame in temporal blending, 0-1 (default: 0.8)
        edge_canny_low: Canny edge detection lower threshold (default: 200)
        edge_canny_high: Canny edge detection upper threshold (default: 255)
        edge_dilation_size: Dilation size for edge tolerance in pixels (default: 2)

    Returns:
        PreprocessingResult: Dataclass containing:
            - blurred: Gaussian-blurred input (for blue channel)
            - edges: High-confidence fish edges (for green channel)
            - motion: Refined motion detection (for red channel)
            - history: Motion result for next frame's temporal smoothing
    """
    # Step 1: Apply Gaussian blur to reduce noise
    # This is critical for sonar imagery which has significant speckle noise
    frame_blurred = cv2.GaussianBlur(
        frame, (gaussian_kernel, gaussian_kernel), gaussian_sigma
    )

    # Step 2: Apply MOG2 background subtraction
    # This isolates moving objects (fish) from the static background
    mog_mask = bg_subtractor.apply(frame_blurred)

    # Step 3: Apply bidirectional guided filtering
    # Convert grayscale inputs to RGB for guided filter
    mog_mask_rgb = cv2.cvtColor(mog_mask, cv2.COLOR_GRAY2RGB)
    frame_blurred_rgb = cv2.cvtColor(frame_blurred, cv2.COLOR_GRAY2RGB)

    # Direction 1: Frame-guided MOG (original direction)
    # Use original frame as guide to refine MOG mask
    # This preserves edges from the original frame while smoothing the MOG mask
    guided_mog_rgb = cv2.ximgproc.guidedFilter(
        guide=frame_blurred,
        src=mog_mask_rgb,
        radius=guided_radius,
        eps=guided_eps,
    )

    # Direction 2: MOG-guided frame (reverse direction)
    # Use MOG mask as guide to extract frame structure in motion regions
    # This highlights frame details where motion was detected
    guided_img_rgb = cv2.ximgproc.guidedFilter(
        guide=mog_mask,
        src=frame_blurred_rgb,
        radius=guided_radius,
        eps=guided_eps,
    )

    # Convert back to grayscale
    guided_mog = cv2.cvtColor(guided_mog_rgb, cv2.COLOR_BGR2GRAY)
    guided_img = cv2.cvtColor(guided_img_rgb, cv2.COLOR_BGR2GRAY)

    # Step 4: Edge detection on both guided outputs
    # Detect edges in the MOG-guided frame
    edge_img = cv2.Canny(guided_img, edge_canny_low, edge_canny_high)

    # Detect edges in the frame-guided MOG
    edge_mog = cv2.Canny(guided_mog, edge_canny_low, edge_canny_high)

    # Step 5: Compute edge intersection
    # Dilate one edge map to allow slight misalignment tolerance
    kernel_size = edge_dilation_size * 2 + 1  # Convert radius to kernel size
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edge_mog_dilated = cv2.dilate(edge_mog, kernel, iterations=1)

    # Intersection: only keep edges that appear in BOTH filtered versions
    # This eliminates spurious edges and keeps only high-confidence fish boundaries
    edge_intersection = cv2.bitwise_and(edge_img, edge_mog_dilated)

    # Step 6: Apply temporal smoothing to motion channel
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

    return PreprocessingResult(
        blurred=frame_blurred,
        edges=edge_intersection,
        motion=processed_frame,
        history=processed_frame,
    )


def create_visualization(
    frame_blurred: NDArray[np.uint8],
    edge_intersection: NDArray[np.uint8],
    frame_preprocessed: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """
    Create RGB visualization for sonar preprocessing with edge detection.

    Combines the Gaussian-blurred input, edge intersection, and preprocessed output
    into a single RGB frame for comprehensive visual analysis.

    Channel assignment:
    - Blue channel: Gaussian-blurred input (shows sonar structure after noise reduction)
    - Green channel: Edge intersection (high-confidence fish boundaries from bidirectional filtering)
    - Red channel: Preprocessed output (shows detected motion after guided filter + temporal smoothing)

    Visual interpretation (colors are additive):
    - Pure blue: Static sonar background structure
    - Pure green: High-confidence object edges (fish boundaries detected by both filters)
    - Pure red: Detected moving regions (motion blobs)
    - Cyan (blue + green): Static edges in the scene
    - Yellow (green + red): Moving objects with clear boundaries → FISH (highest confidence)
    - Magenta (blue + red): Motion over background structure
    - White (all channels): Bright moving object with edges over structure
    - Black: No signal in any channel

    Args:
        frame_blurred: Gaussian-blurred input frame (grayscale)
        edge_intersection: Edge intersection mask (grayscale, binary edges)
        frame_preprocessed: Preprocessed output frame (grayscale)

    Returns:
        NDArray[np.uint8]: RGB frame with shape (H, W, 3)
    """
    return np.dstack(
        [
            frame_blurred,  # Blue channel
            edge_intersection,  # Green channel (fish edges)
            frame_preprocessed,  # Red channel
        ]
    )
