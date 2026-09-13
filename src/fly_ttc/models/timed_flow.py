"""Timestamp-aware flow control for synthetic research, separate from frozen v0."""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from fly_ttc.preprocess.expansion import center_weights, divergence, radial_flow
from fly_ttc.preprocess.flow import clip_flow_magnitude, farneback_flow, validate_gray


def continuous_ema_alpha(dt_s: float, tau_s: float) -> float:
    """Exact decay for a held scalar input, with physical time in seconds."""
    if not np.isfinite([dt_s, tau_s]).all() or dt_s <= 0 or tau_s <= 0:
        raise ValueError("dt_s and tau_s must be finite and positive")
    return -math.expm1(-dt_s / tau_s)


def reference_displacement(flow, dt_s: float, reference_dt_s: float):
    """Convert pair displacement to pixels per declared reference interval."""
    if not np.isfinite([dt_s, reference_dt_s]).all() or min(dt_s, reference_dt_s) <= 0:
        raise ValueError("Intervals must be finite and positive")
    value = np.asarray(flow, dtype=np.float32)
    if value.ndim != 3 or value.shape[-1] != 2 or not np.isfinite(value).all():
        raise ValueError("flow must be a finite HxWx2 displacement field")
    return value * (reference_dt_s / dt_s)


@dataclass(frozen=True)
class TimedFlowConfig:
    reference_dt_s: float = 1 / 15
    output_tau_s: float = 0.05
    blur_ksize: int = 5
    clip_percentile: float = 99.0
    center_edge_weight: float = 0.3

    def __post_init__(self):
        continuous_ema_alpha(self.reference_dt_s, self.output_tau_s)
        if self.blur_ksize <= 0 or self.blur_ksize % 2 != 1:
            raise ValueError("blur_ksize must be positive and odd")
        if not 0 < self.clip_percentile <= 100 or not 0 < self.center_edge_weight <= 1:
            raise ValueError("Invalid clipping percentile or center weight")


class TimedFlowScorer:
    """Uncalibrated divergence+radial control, no blob or affine subtraction.

    Components use a fixed reference-interval displacement; their sum is an
    arbitrary engineering score, not TTC or an estimate in physical seconds.
    Timestamp normalization does not remove sampling/optical-flow estimation
    error. The original v0 extractor and results are never modified.
    """

    def __init__(self, config=TimedFlowConfig(), flow_config=None):
        self.config, self.flow_config = config, flow_config
        self.reset()

    def reset(self):
        self._gray, self._t, self._weights = None, None, None
        self._score = 0.0

    def update(self, gray, t_s):
        gray = validate_gray(gray)
        if not np.isfinite(t_s) or (self._t is not None and t_s <= self._t):
            raise ValueError("Timestamps must be finite and strictly increasing")
        if self._gray is not None and gray.shape != self._gray.shape:
            raise ValueError("Frame shape changed; reset before a new stimulus")
        gray = cv2.GaussianBlur(gray, (self.config.blur_ksize,) * 2, 0)
        if self._gray is None:
            self._gray, self._t = gray, float(t_s)
            self._weights = center_weights(gray.shape, self.config.center_edge_weight)
            return dict(S=0.0, S_div_ref=0.0, S_rad_ref=0.0, dt_s=0.0, valid=False)
        dt = float(t_s - self._t)
        flow = reference_displacement(farneback_flow(self._gray, gray, self.flow_config), dt, self.config.reference_dt_s)
        flow = clip_flow_magnitude(flow, self.config.clip_percentile)
        div = float(np.mean(np.maximum(divergence(flow), 0) * self._weights))
        rad = float(np.mean(np.maximum(radial_flow(flow), 0) * self._weights))
        alpha = continuous_ema_alpha(dt, self.config.output_tau_s)
        self._score += alpha * (div + rad - self._score)
        self._gray, self._t = gray, float(t_s)
        return dict(S=self._score, S_div_ref=div, S_rad_ref=rad, dt_s=dt, valid=True)
