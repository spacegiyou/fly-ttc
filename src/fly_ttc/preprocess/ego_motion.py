"""Fixed, deterministic dominant-image-motion subtraction for an ablation.

Fit displacement (not transformed coordinates) in pixels per frame pair on a
regular grid outside a central rectangle. A robust affine fit can account for
image translation, in-plane rotation, shear and uniform zoom. It is NOT a
physical ego-motion estimate: forward translation has depth-dependent parallax,
and broad true looming is indistinguishable from global image zoom here. The
central exclusion protects only localized central motion, not every obstacle.

The affine model follows the coordinate map documented by OpenCV:
https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
The perspective projection there, x = f X / Z, explains the depth limitation.
We use deterministic Huber IRLS rather than a random RANSAC sampler. All fit
settings are fixed engineering choices; fitting uses no labels or time stamps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EgoMotionConfig:
    """Settings at the resized-image scale; errors are pixels per frame pair."""

    method: str = "affine"
    grid_step_px: int = 12
    border_fraction: float = 0.05
    center_exclusion_fraction: float = 0.5
    irls_iterations: int = 8
    huber_delta_px: float = 1.0
    inlier_threshold_px: float = 1.5
    min_inlier_fraction: float = 0.5
    min_samples: int = 24
    min_axis_span_fraction: float = 0.5
    min_quadrants: int = 3
    max_condition_number: float = 100.0
    min_motion_px: float = 0.02

    def __post_init__(self) -> None:
        if self.method not in {"affine", "translation"}:
            raise ValueError("ego-motion method must be affine or translation")
        for name in ("grid_step_px", "irls_iterations", "min_samples", "min_quadrants"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_samples < 3 or self.min_quadrants > 4:
            raise ValueError("min_samples must be >=3 and min_quadrants <=4")
        if not 0 <= self.border_fraction < 0.5:
            raise ValueError("border_fraction must be in [0, 0.5)")
        if not 0 <= self.center_exclusion_fraction < 1:
            raise ValueError("center_exclusion_fraction must be in [0, 1)")
        for name in ("min_inlier_fraction", "min_axis_span_fraction"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("huber_delta_px", "inlier_threshold_px", "max_condition_number"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.min_motion_px) or self.min_motion_px < 0:
            raise ValueError("min_motion_px must be finite and nonnegative")


def _sample_coordinates(shape: tuple[int, int], config: EgoMotionConfig) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    margin_y = max(1, int(np.ceil(config.border_fraction * (height - 1))))
    margin_x = max(1, int(np.ceil(config.border_fraction * (width - 1))))
    yy, xx = np.meshgrid(
        np.arange(margin_y, height - margin_y, config.grid_step_px),
        np.arange(margin_x, width - margin_x, config.grid_step_px),
        indexing="ij",
    )
    x, y = xx.ravel(), yy.ravel()
    cx, cy = (width - 1) / 2, (height - 1) / 2
    inside = (
        (np.abs(x - cx) < config.center_exclusion_fraction * width / 2)
        & (np.abs(y - cy) < config.center_exclusion_fraction * height / 2)
    )
    return x[~inside], y[~inside]


def _geometry_valid(x: np.ndarray, y: np.ndarray, shape: tuple[int, int], config: EgoMotionConfig) -> bool:
    if len(x) < config.min_samples:
        return False
    height, width = shape
    if np.ptp(x) / (width - 1) < config.min_axis_span_fraction:
        return False
    if np.ptp(y) / (height - 1) < config.min_axis_span_fraction:
        return False
    quadrants = (x >= (width - 1) / 2).astype(int) + 2 * (y >= (height - 1) / 2)
    if len(np.unique(quadrants)) < config.min_quadrants:
        return False
    # Axis spans alone would accept a thin diagonal stripe. Conditioning the
    # normalized spatial design rejects collinear or nearly collinear support.
    design = np.column_stack((2 * x / (width - 1) - 1, 2 * y / (height - 1) - 1, np.ones(len(x))))
    return bool(np.linalg.matrix_rank(design) == 3 and np.linalg.cond(design) <= config.max_condition_number)


def estimate_background_flow(flow: np.ndarray, config: EgoMotionConfig = EgoMotionConfig()) -> dict:
    """Return background/residual fields and transparent fit diagnostics.

    Input is finite HxWx2 previous-to-current displacement BEFORE magnitude
    clipping, in resized pixels per frame pair. The output uses float32 and the
    same pixel coordinate system. No warping or temporal normalization occurs.

    On insufficient, spatially narrow, low-motion or inconsistent support, the
    background is zero and the residual is an unchanged copy of raw flow. A
    false ``fit_valid`` means abstention, not a verified stationary camera.
    ``fit_error_px`` is median sampled residual norm; ``inlier_fraction`` uses
    all samples, excluding the fixed central rectangle and image border.
    The flow-only API cannot establish image texture/photometric confidence.
    """
    raw = np.asarray(flow)
    if raw.ndim != 3 or raw.shape[-1] != 2 or min(raw.shape[:2]) < 3:
        raise ValueError("flow must have shape HxWx2 with H,W >= 3")
    if not np.issubdtype(raw.dtype, np.floating):
        raise TypeError("flow must be floating point displacement")
    if not np.isfinite(raw).all():
        raise ValueError("flow must contain finite displacement")
    raw = np.ascontiguousarray(raw, dtype=np.float32)
    height, width = raw.shape[:2]
    x, y = _sample_coordinates((height, width), config)
    diagnostic = dict(
        fit_valid=False,
        inlier_fraction=0.0,
        fit_error_px=float("nan"),
        method=config.method,
        fit_reason="insufficient_samples",
        n_samples=int(len(x)),
        n_inliers=0,
    )

    def abstain(reason: str) -> dict:
        return dict(
            diagnostic,
            fit_reason=reason,
            background_flow=np.zeros_like(raw),
            residual_flow=raw.copy(),
        )

    if len(x) < config.min_samples:
        return abstain("insufficient_samples")
    if not _geometry_valid(x, y, (height, width), config):
        return abstain("insufficient_spatial_support")
    observed = raw[y, x].astype(np.float64)
    magnitude = np.linalg.norm(observed, axis=1)
    if float(np.median(magnitude)) <= config.min_motion_px:
        diagnostic.update(fit_error_px=float(np.median(magnitude)))
        return abstain("low_motion")
    design = np.column_stack((2 * x / (width - 1) - 1, 2 * y / (height - 1) - 1, np.ones(len(x))))
    # The translation-only control deliberately uses a componentwise median.
    # The affine variant initializes there and minimizes vector Huber loss.
    coefficients = np.zeros((3, 2), dtype=np.float64)
    coefficients[2] = np.median(observed, axis=0)
    if config.method == "affine":
        for _ in range(config.irls_iterations):
            errors = np.linalg.norm(observed - design @ coefficients, axis=1)
            weights = np.sqrt(np.minimum(1.0, config.huber_delta_px / np.maximum(errors, 1e-12)))
            weighted_design = design * weights[:, None]
            if np.linalg.cond(weighted_design) > config.max_condition_number:
                return abstain("ill_conditioned_fit")
            try:
                coefficients, _, rank, _ = np.linalg.lstsq(weighted_design, observed * weights[:, None], rcond=None)
            except np.linalg.LinAlgError:
                return abstain("linear_algebra_failure")
            if rank != 3 or not np.isfinite(coefficients).all():
                return abstain("ill_conditioned_fit")
    errors = np.linalg.norm(observed - design @ coefficients, axis=1)
    inliers = errors <= config.inlier_threshold_px
    diagnostic.update(
        fit_error_px=float(np.median(errors)),
        inlier_fraction=float(np.mean(inliers)),
        n_inliers=int(np.sum(inliers)),
    )
    if diagnostic["inlier_fraction"] < config.min_inlier_fraction:
        return abstain("insufficient_inliers")
    if not _geometry_valid(x[inliers], y[inliers], (height, width), config):
        return abstain("insufficient_inlier_spatial_support")
    xx = (2 * np.arange(width, dtype=np.float64) / (width - 1) - 1)[None, :, None]
    yy = (2 * np.arange(height, dtype=np.float64) / (height - 1) - 1)[:, None, None]
    background = (xx * coefficients[0] + yy * coefficients[1] + coefficients[2]).astype(np.float32)
    return dict(
        diagnostic,
        fit_valid=True,
        fit_reason="accepted",
        background_flow=background,
        residual_flow=raw - background,
    )
