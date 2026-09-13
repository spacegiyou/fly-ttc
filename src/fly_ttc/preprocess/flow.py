"""CPU dense optical flow with explicit intensity and displacement units."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np


FARNEBACK_DEFAULTS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 21,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}


def validate_gray(gray: np.ndarray) -> np.ndarray:
    """Validate the scorer contract, returning contiguous float32 in [0, 1]."""
    image = np.asarray(gray)
    if image.ndim != 2 or min(image.shape, default=0) < 3:
        raise ValueError("gray must be a nonempty HxW image at least 3x3")
    if not np.issubdtype(image.dtype, np.floating):
        raise TypeError("gray must be floating point in [0, 1]; divide uint8 by 255")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("gray must contain finite intensities in [0, 1]")
    return np.ascontiguousarray(image, dtype=np.float32)


def farneback_flow(
    previous: np.ndarray,
    current: np.ndarray,
    config: Mapping | None = None,
) -> np.ndarray:
    """Return previous-to-current (u, v) displacement in pixels per frame pair.

    Farneback's polynomial conditioning depends on intensity scale. Convert the
    public normalized images to uint8; passing 0..1 float images directly can
    silently produce near-zero flow. Positive u is right and positive v is down.
    """
    previous = validate_gray(previous)
    current = validate_gray(current)
    if previous.shape != current.shape:
        raise ValueError("Consecutive grayscale frames must have the same shape")
    supplied = dict(config or {})
    if "method" in supplied or "farneback" in supplied:
        method = supplied.pop("method", "farneback")
        if method != "farneback":
            raise NotImplementedError("v0 implements CPU Farneback only")
        supplied = dict(supplied.get("farneback", {}))
    unexpected = set(supplied) - FARNEBACK_DEFAULTS.keys()
    if unexpected:
        raise ValueError(f"Unknown Farneback parameters: {sorted(unexpected)}")
    parameters = {**FARNEBACK_DEFAULTS, **supplied}
    prev_u8 = np.rint(previous * 255).astype(np.uint8)
    curr_u8 = np.rint(current * 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(prev_u8, curr_u8, None, **parameters)
    if not np.isfinite(flow).all():
        raise ValueError("Farneback returned nonfinite displacement")
    return flow


def clip_flow_magnitude(flow: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Winsorize large displacement magnitudes while preserving directions."""
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[-1] != 2 or not np.isfinite(flow).all():
        raise ValueError("flow must be finite HxWx2")
    if not 0 < percentile <= 100:
        raise ValueError("flow clipping percentile must be in (0, 100]")
    magnitude = np.linalg.norm(flow, axis=-1)
    limit = float(np.percentile(magnitude, percentile))
    scale = np.ones_like(magnitude)
    np.divide(limit, magnitude, out=scale, where=magnitude > limit)
    return flow * scale[..., None]
