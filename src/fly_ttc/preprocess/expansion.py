"""Image-space expansion components; these are not biological neuron models."""

from __future__ import annotations

import cv2
import numpy as np


def center_weights(shape: tuple[int, int], edge_weight: float = 0.3) -> np.ndarray:
    """Elliptical Gaussian, normalized to one centrally and floored at the edge.

    Each image half-axis is one Gaussian radius with value ``edge_weight``.
    The floor keeps the entire boundary at that value, including the corners.
    """
    if not 0 < edge_weight <= 1:
        raise ValueError("center_edge_weight must be in (0, 1]")
    h, w = shape
    y, x = np.mgrid[:h, :w].astype(np.float32)
    x = (x - (w - 1) / 2) / max((w - 1) / 2, 1)
    y = (y - (h - 1) / 2) / max((h - 1) / 2, 1)
    weights = np.exp(np.log(edge_weight) * (x * x + y * y))
    weights /= weights.max()
    return np.maximum(weights, edge_weight).astype(np.float32)


def divergence(flow: np.ndarray, sobel_ksize: int = 3) -> np.ndarray:
    """Sobel du/dx + dv/dy with derivative scaling in inverse pixels."""
    if sobel_ksize not in (1, 3, 5, 7):
        raise ValueError("sobel_ksize must be 1, 3, 5, or 7")
    # A unit-ramp derivative must be one, independent of Sobel kernel size.
    derivative, smoothing = cv2.getDerivKernels(1, 0, sobel_ksize, normalize=False)
    coordinates = np.arange(len(derivative)) - (len(derivative) - 1) / 2
    ramp_gain = float(np.dot(derivative.ravel(), coordinates) * smoothing.sum())
    scale = 1.0 / ramp_gain
    u, v = flow[..., 0], flow[..., 1]
    return cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=sobel_ksize, scale=scale) + cv2.Sobel(
        v, cv2.CV_32F, 0, 1, ksize=sobel_ksize, scale=scale
    )


def radial_flow(flow: np.ndarray) -> np.ndarray:
    """Project displacement onto outward unit vectors from the image center."""
    h, w = flow.shape[:2]
    y, x = np.mgrid[:h, :w].astype(np.float32)
    x -= (w - 1) / 2
    y -= (h - 1) / 2
    radius = np.maximum(np.hypot(x, y), 1.0)
    return (flow[..., 0] * x + flow[..., 1] * y) / radius


def equivalent_blob_diameter(
    previous: np.ndarray,
    current: np.ndarray,
    diff_threshold: float = 0.04,
    min_area_px: float = 16.0,
) -> float:
    """Equivalent diameter of the largest external frame-difference contour.

    The outer contour includes the interior of a moving ring. It is a change
    region proxy, not an object segmentation. A uniform flash can create a
    full-image contour; it cannot produce a growth term without a prior blob.
    """
    mask = (np.abs(current - previous) > diff_threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = max((cv2.contourArea(contour) for contour in contours), default=0.0)
    return float(np.sqrt(4.0 * area / np.pi)) if area >= min_area_px else 0.0


def blob_growth(diameter: float, previous_diameter: float) -> float:
    """Positive fractional growth, zero on absent masks and first appearance."""
    if diameter <= 0 or previous_diameter <= 0:
        return 0.0
    return max(diameter - previous_diameter, 0.0) / max(previous_diameter, 1.0)
