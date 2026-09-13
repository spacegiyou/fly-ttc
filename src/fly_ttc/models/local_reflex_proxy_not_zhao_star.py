"""Timestamp-aware local radial opponency, explicitly not a Zhao STAR port.

The EMD recurrence follows the authors' public ``emd.m`` and ``loom.m``;
local opponent pooling follows Zhao 2023 STAR Eqs. 1--5. RF tiling, arm
means, fourth-root output and optional output EMA are engineering choices.
See docs/LOCAL_REFLEX_MODEL_SOURCES.md for exact upstream paths and deviations.
No optical flow, GF spikes, angular calibration, or accident probability is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

MODEL_NAME = "proxy_not_zhao_star"
DIRECTION_NAMES = ("right", "left", "down", "up")
OPPOSITE = (1, 0, 3, 2)
UPSTREAM_COMMIT = "50c7c8f3f74052b2562ccba967c9f1032f37405b"


@dataclass(frozen=True)
class LocalReflexConfig:
    """Times are seconds, lengths are input pixels, thresholds are EMD AU.

    ``sample_spacing_px`` is neighbor separation, not a resize operation.
    The permutation lists the input channel assigned to each canonical channel.
    Opposite channels are then paired in that permuted coordinate system.
    """

    highpass_tau_s: float = 0.250
    lowpass_tau_s: float = 0.050
    sample_spacing_px: int = 1
    rf_size_px: int = 32
    grid: int = 5
    arm_width_fraction: float = 1 / 3
    arm_threshold: float = 0.0
    output_tau_s: float = 0.050
    direction_permutation: tuple[int, int, int, int] = (0, 1, 2, 3)
    aggregation: str = "product"
    pathways: str = "on_off"

    def __post_init__(self) -> None:
        for name in ("highpass_tau_s", "lowpass_tau_s"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("sample_spacing_px", "rf_size_px", "grid"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.arm_width_fraction) or not 0 < self.arm_width_fraction <= 1:
            raise ValueError("arm_width_fraction must be in (0, 1]")
        if not np.isfinite(self.arm_threshold) or self.arm_threshold < 0:
            raise ValueError("arm_threshold must be finite and nonnegative")
        if not np.isfinite(self.output_tau_s) or self.output_tau_s < 0:
            raise ValueError("output_tau_s must be finite and nonnegative")
        permutation = tuple(self.direction_permutation)
        if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in permutation):
            raise ValueError("direction_permutation must permute 0, 1, 2, 3")
        if sorted(permutation) != [0, 1, 2, 3]:
            raise ValueError("direction_permutation must permute 0, 1, 2, 3")
        object.__setattr__(self, "direction_permutation", permutation)
        if self.aggregation not in ("product", "mean"):
            raise ValueError("aggregation must be product or mean")
        if self.pathways not in ("on_off", "off"):
            raise ValueError("pathways must be on_off or off")


def lowpass_alpha(dt_s: float, tau_s: float) -> float:
    """Backward-Euler coefficient, matching upstream dt/(tau+dt)."""
    if not np.isfinite(dt_s) or dt_s <= 0 or not np.isfinite(tau_s) or tau_s < 0:
        raise ValueError("dt_s must be positive and tau_s nonnegative, both finite")
    return float(dt_s / (tau_s + dt_s))


def _validate_gray(gray: np.ndarray) -> np.ndarray:
    image = np.asarray(gray)
    if image.ndim != 2 or min(image.shape) < 2:
        raise ValueError("gray must be a two-dimensional image")
    if not np.issubdtype(image.dtype, np.number) or np.iscomplexobj(image):
        raise ValueError("gray must contain real numbers in [0, 1]")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("gray must contain finite numbers in [0, 1]")
    return np.asarray(image, dtype=np.float64)


class TimestampEMD:
    """ON/OFF neighbor correlators with causal state and explicit timestamps.

    Output channels are nonnegative half-correlator products R,L,D,U, not
    velocities. Opponency is performed by the readout. The common output
    lattice has shape (H-spacing, W-spacing); no boundary wrapping/padding.
    """

    def __init__(self, config: LocalReflexConfig | None = None):
        self.config = config or LocalReflexConfig()
        self.reset()

    def reset(self) -> None:
        self.previous_gray: np.ndarray | None = None
        self.previous_t_s: float | None = None
        self.highpass: np.ndarray | None = None
        self.delayed_on: np.ndarray | None = None
        self.delayed_off: np.ndarray | None = None

    def update(self, gray: np.ndarray, t_s: float) -> dict[str, Any]:
        current = _validate_gray(gray)
        if not np.isfinite(t_s):
            raise ValueError("t_s must be finite")
        step = self.config.sample_spacing_px
        if min(current.shape) <= step:
            raise ValueError("image must be larger than sample_spacing_px")
        if self.previous_t_s is not None and t_s <= self.previous_t_s:
            raise ValueError("timestamps must be strictly increasing; reset for a new clip")
        if self.previous_gray is not None and current.shape != self.previous_gray.shape:
            raise ValueError("image shape changed; reset for a new clip")
        if self.previous_gray is None:
            self.previous_gray = current.copy()
            self.previous_t_s = float(t_s)
            self.highpass = np.zeros_like(current)
            self.delayed_on = np.zeros_like(current)
            self.delayed_off = np.zeros_like(current)
            return {"directional_motion": np.zeros((*np.subtract(current.shape, step), 4)),
                    "dt_s": 0.0, "t_s": float(t_s), "input_shape": current.shape}

        dt_s = float(t_s - self.previous_t_s)
        # Upstream emd.m/loom.m: beta_H=tau_H/(tau_H+dt), alpha_L=dt/(tau_L+dt).
        beta = 1.0 - lowpass_alpha(dt_s, self.config.highpass_tau_s)
        hp = beta * (current - self.previous_gray + self.highpass)
        on = np.maximum(hp, 0.0)
        # The authors' OffRect(x,0.05) is max(0.05-x,0), not max(-x-0.05,0).
        off = np.maximum(0.05 - hp, 0.0)
        alpha = lowpass_alpha(dt_s, self.config.lowpass_tau_s)
        delayed_on = alpha * on + (1.0 - alpha) * self.delayed_on
        delayed_off = alpha * off + (1.0 - alpha) * self.delayed_off
        channels = np.zeros((current.shape[0] - step, current.shape[1] - step, 4))
        paths = [(off, delayed_off)]
        if self.config.pathways == "on_off":
            paths.append((on, delayed_on))
        for fast, delayed in paths:
            origin = fast[:-step, :-step]
            delay_origin = delayed[:-step, :-step]
            # Delay the upstream position: a feature moving right makes R>L.
            channels[..., 0] += delay_origin * fast[:-step, step:]
            channels[..., 1] += origin * delayed[:-step, step:]
            channels[..., 2] += delay_origin * fast[step:, :-step]
            channels[..., 3] += origin * delayed[step:, :-step]
        self.previous_gray = current.copy()
        self.previous_t_s = float(t_s)
        self.highpass = hp
        self.delayed_on = delayed_on
        self.delayed_off = delayed_off
        return {"directional_motion": channels, "dt_s": dt_s,
                "t_s": float(t_s), "input_shape": current.shape}


@lru_cache(maxsize=64)
def _rf_layout(motion_shape: tuple[int, int], rf_size_px: int,
               grid: int, width_fraction: float, spacing: int) -> tuple[np.ndarray, np.ndarray]:
    """Return centers and half-open rectangles (x0,y0,x1,y1), with full RF support."""
    height, width = motion_shape
    if min(height, width) < rf_size_px:
        raise ValueError("image has insufficient complete RF support; reduce rf_size_px")
    # All channels share the EMD cell center convention (x+s/2,y+s/2).
    offset = spacing / 2
    x_min = offset - 0.5 + rf_size_px / 2
    x_max = offset - 0.5 + width - rf_size_px / 2
    y_min = offset - 0.5 + rf_size_px / 2
    y_max = offset - 0.5 + height - rf_size_px / 2
    xs = np.linspace(x_min, x_max, grid) if grid > 1 else np.array([(x_min + x_max) / 2])
    ys = np.linspace(y_min, y_max, grid) if grid > 1 else np.array([(y_min + y_max) / 2])
    xx, yy = np.meshgrid(xs, ys)
    centers = np.stack([xx, yy], axis=-1)
    half = rf_size_px / 2
    narrow = rf_size_px * width_fraction / 2
    rectangles = np.zeros((grid, grid, 4, 4), dtype=int)
    for row in range(grid):
        for col in range(grid):
            cx, cy = centers[row, col]
            boxes = ((cx, cy - narrow, cx + half, cy + narrow),
                     (cx - half, cy - narrow, cx, cy + narrow),
                     (cx - narrow, cy, cx + narrow, cy + half),
                     (cx - narrow, cy - half, cx + narrow, cy))
            for arm, bounds in enumerate(boxes):
                rectangle = np.ceil(np.asarray(bounds) - offset).astype(int)
                rectangle[[0, 2]] = np.clip(rectangle[[0, 2]], 0, width)
                rectangle[[1, 3]] = np.clip(rectangle[[1, 3]], 0, height)
                if rectangle[0] >= rectangle[2] or rectangle[1] >= rectangle[3]:
                    raise ValueError("RF arm has no EMD samples; increase RF size or arm width")
                rectangles[row, col, arm] = rectangle
    centers.setflags(write=False)
    rectangles.setflags(write=False)
    return centers, rectangles


def _rect_sum(integral: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = np.moveaxis(boxes, -1, 0)
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def pool_directional_motion(motion: np.ndarray, config: LocalReflexConfig | None = None,
                            input_shape: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
    """Pure RF readout for fair reuse of one EMD state across channel permutations.

    ``motion`` contains raw half-correlators R,L,D,U. ``activation_map`` is
    unsmoothed. ``simple_energy_map`` averages unsigned horizontal+vertical
    opponency across the SAME four arm supports (overlaps counted per arm).
    It ignores direction assignment and is invariant under the readout control.
    No state, fitted coefficients, thresholds, or random draws are hidden here.
    """
    cfg = config or LocalReflexConfig()
    field = np.asarray(motion, dtype=np.float64)
    if field.ndim != 3 or field.shape[-1] != 4 or min(field.shape[:2]) == 0:
        raise ValueError("motion must have shape (H-spacing, W-spacing, 4)")
    if not np.isfinite(field).all() or (field < 0).any():
        raise ValueError("half-correlator fields must be finite and nonnegative")
    if input_shape is not None and tuple(np.add(field.shape[:2], cfg.sample_spacing_px)) != tuple(input_shape):
        raise ValueError("input_shape disagrees with the EMD field and spacing")
    centers, rectangles = _rf_layout(field.shape[:2], cfg.rf_size_px, cfg.grid,
                                     cfg.arm_width_fraction, cfg.sample_spacing_px)
    permuted = field[..., cfg.direction_permutation]
    opponent = permuted - permuted[..., OPPOSITE]
    energy = np.abs(field[..., 0] - field[..., 1]) + np.abs(field[..., 2] - field[..., 3])
    combined = np.concatenate([opponent, energy[..., None]], axis=-1)
    integral = np.pad(combined.cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0)))
    sums = _rect_sum(integral, rectangles)
    areas = ((rectangles[..., 2] - rectangles[..., 0]) *
             (rectangles[..., 3] - rectangles[..., 1]))
    means = sums / areas[..., None]
    arm_opponent = np.stack([means[..., arm, arm] for arm in range(4)], axis=-1)
    # Zhao Eq.5 subtracts the threshold; the public MATLAB code instead hard-gates.
    positive = np.maximum(arm_opponent - cfg.arm_threshold, 0.0)
    if cfg.aggregation == "product":
        # Fourth root preserves the all-four-arms gate, on an interpretable AU scale.
        activation = np.prod(positive, axis=-1) ** 0.25
    else:
        activation = positive.mean(axis=-1)
    simple_energy = means[..., 4].mean(axis=-1)
    return {"activation_map": activation, "unit_centers_xy": centers.copy(),
            "arm_opponent": arm_opponent, "simple_energy_map": simple_energy}


class LocalReflexScorer:
    """Small CPU visual-feature module; ``S`` is maximum local activation AU."""

    model_name = MODEL_NAME

    def __init__(self, config: LocalReflexConfig | None = None):
        self.config = config or LocalReflexConfig()
        self.emd = TimestampEMD(self.config)
        self._activation: np.ndarray | None = None

    def reset(self) -> None:
        self.emd.reset()
        self._activation = None

    def update(self, gray: np.ndarray, t_s: float) -> dict[str, Any]:
        # Validate readout geometry before updating EMD state.
        shape = np.asarray(gray).shape
        if len(shape) == 2:
            _rf_layout(tuple(x - self.config.sample_spacing_px for x in shape),
                       self.config.rf_size_px, self.config.grid,
                       self.config.arm_width_fraction, self.config.sample_spacing_px)
        frame = self.emd.update(gray, t_s)
        pooled = pool_directional_motion(frame["directional_motion"], self.config, frame["input_shape"])
        raw = pooled["activation_map"]
        if self._activation is None:
            self._activation = raw.copy()
        else:
            # Exact exponential smoothing under a piecewise-constant raw readout.
            alpha = 1.0 if self.config.output_tau_s == 0 else -np.expm1(-frame["dt_s"] / self.config.output_tau_s)
            self._activation += alpha * (raw - self._activation)
        return {**frame, **pooled, "activation_raw_map": raw,
                "activation_map": self._activation.copy(),
                "S": float(self._activation.max()), "model_name": self.model_name}
