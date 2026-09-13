"""Fixed Farneback expansion proxy inspired by LPLC2/Giant Fiber."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from fly_ttc.preprocess.expansion import (
    blob_growth,
    center_weights,
    divergence,
    equivalent_blob_diameter,
    radial_flow,
)
from fly_ttc.preprocess.flow import clip_flow_magnitude, farneback_flow, validate_gray


COMPONENTS = ("S_div", "S_rad", "S_blob")


@dataclass(frozen=True)
class V0Config:
    ema: float = 0.3
    center_weight: bool = True
    blur_ksize: int = 5
    score: str = "combined"
    flow_clip_percentile: float = 99.0
    center_edge_weight: float = 0.3
    sobel_ksize: int = 3
    blob_diff_threshold: float = 0.04
    blob_min_area_px: float = 16.0
    blob_weight: float = 0.5
    normalization_std_floor: float = 1e-6

    def __post_init__(self) -> None:
        if not 0 < self.ema <= 1:
            raise ValueError("ema must be in (0, 1]")
        if self.blur_ksize < 1 or self.blur_ksize % 2 != 1:
            raise ValueError("blur_ksize must be a positive odd integer")
        if self.score not in {"divergence", "radial", "blob", "combined"}:
            raise ValueError("score must be divergence, radial, blob, or combined")
        if self.sobel_ksize not in (1, 3, 5, 7):
            raise ValueError("sobel_ksize must be 1, 3, 5, or 7")
        if not 0 < self.flow_clip_percentile <= 100:
            raise ValueError("flow_clip_percentile must be in (0, 100]")
        if not 0 < self.center_edge_weight <= 1:
            raise ValueError("center_edge_weight must be in (0, 1]")
        if not 0 <= self.blob_diff_threshold <= 1 or self.blob_min_area_px < 0:
            raise ValueError("Invalid blob threshold or minimum area")
        if self.blob_weight < 0 or self.normalization_std_floor <= 0:
            raise ValueError("blob_weight must be nonnegative and std floor positive")


def fit_normalization(
    negative_validation_frames: Iterable[pd.DataFrame],
    config: V0Config = V0Config(),
) -> dict:
    """Fit population statistics to frame components from negative validation.

    The caller supplies only held-out negative clips, never evaluation or
    positive clips. When label/split columns are supplied, enforce that contract.
    Initial frames without a flow pair are omitted. The caller records clip ids.
    """
    arrays = []
    for frames in negative_validation_frames:
        if "label" in frames and not frames["label"].eq(0).all():
            raise ValueError("Normalization requires negative validation clips only")
        if "split" in frames and not frames["split"].isin(["val", "validation"]).all():
            raise ValueError("Normalization requires validation clips only")
        valid = frames.loc[frames["flow_valid"].astype(bool)] if "flow_valid" in frames else frames
        values = valid.loc[:, COMPONENTS].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Normalization components must be finite")
        if len(values):
            arrays.append(values)
    if not arrays:
        raise ValueError("No valid flow pairs from negative validation clips")
    values = np.concatenate(arrays, axis=0)
    result = {
        name: {
            "mean": float(values[:, index].mean()),
            "std": max(float(values[:, index].std(ddof=0)), config.normalization_std_floor),
        }
        for index, name in enumerate(COMPONENTS)
    }
    result.update(source="negative_validation", n_frames=int(len(values)))
    return result


def _raw_score(values: np.ndarray, normalization: Mapping | None, config: V0Config) -> np.ndarray:
    transformed = np.asarray(values, dtype=float).copy()
    if normalization is not None:
        for index, name in enumerate(COMPONENTS):
            mean = float(normalization[name]["mean"])
            std = float(normalization[name]["std"])
            if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
                raise ValueError(f"Invalid normalization statistics for {name}")
            transformed[..., index] = (transformed[..., index] - mean) / max(
                std, config.normalization_std_floor
            )
    if config.score == "combined":
        return transformed @ np.array([1.0, 1.0, config.blob_weight])
    index = {"divergence": 0, "radial": 1, "blob": 2}[config.score]
    return transformed[..., index]


def score_components(
    frames: pd.DataFrame,
    normalization: Mapping | None,
    config: V0Config = V0Config(),
) -> np.ndarray:
    """Reapply fixed normalization and causal EMA to one clip's raw components.

    ``None`` means uncalibrated unit statistics for synthetic/debug use only.
    Each call resets temporal state. Invalid initial rows remain zero and never
    enter the EMA; the first genuine flow pair initializes the EMA directly.
    """
    values = frames.loc[:, COMPONENTS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Score components must be finite")
    raw = _raw_score(values, normalization, config)
    valid = frames["flow_valid"].to_numpy(dtype=bool) if "flow_valid" in frames else np.ones(len(frames), bool)
    result = np.zeros(len(frames), dtype=float)
    previous = None
    for index, value in enumerate(raw):
        if valid[index]:
            previous = float(value) if previous is None else config.ema * float(value) + (1 - config.ema) * previous
            result[index] = previous
    return result


class ExpansionScorer:
    """Stateful frame-pair extraction with optional fixed score calibration.

    ``normalization=None`` exposes uncalibrated scores. Production evaluation
    extracts components first and then uses ``score_components`` with negative
    validation statistics. ``last_flow`` and signed ``last_radial`` support
    visual diagnostics; neither contains the input video image.
    """

    def __init__(
        self,
        config: V0Config = V0Config(),
        flow_config: Mapping | None = None,
        normalization: Mapping | None = None,
    ) -> None:
        self.config = config
        self.flow_config = dict(flow_config or {})
        self.normalization = normalization
        self.reset()

    def reset(self) -> None:
        self._previous: np.ndarray | None = None
        self._previous_diameter = 0.0
        self._ema: float | None = None
        self._weights: np.ndarray | None = None
        self.last_flow: np.ndarray | None = None
        self.last_unclipped_flow: np.ndarray | None = None
        self.last_radial: np.ndarray | None = None

    def update(self, gray_t: np.ndarray) -> dict:
        """Accept HxW float32 in [0,1]; return S and three raw components."""
        gray = validate_gray(gray_t)
        gray = cv2.GaussianBlur(gray, (self.config.blur_ksize,) * 2, 0)
        if self._previous is None:
            self._previous = gray.copy()
            self._weights = center_weights(gray.shape, self.config.center_edge_weight) if self.config.center_weight else np.ones_like(gray)
            self.last_flow = np.zeros((*gray.shape, 2), dtype=np.float32)
            self.last_unclipped_flow = self.last_flow.copy()
            self.last_radial = np.zeros_like(gray)
            return dict(S=0.0, S_div=0.0, S_rad=0.0, S_blob=0.0, flow_median=0.0, flow_horizontal=0.0, frame_diff=0.0, flow_valid=False)
        self.last_unclipped_flow = farneback_flow(self._previous, gray, self.flow_config)
        flow = clip_flow_magnitude(self.last_unclipped_flow, self.config.flow_clip_percentile)
        div = divergence(flow, self.config.sobel_ksize)
        rad = radial_flow(flow)
        diameter = equivalent_blob_diameter(self._previous, gray, self.config.blob_diff_threshold, self.config.blob_min_area_px)
        values = np.array([
            np.mean(np.maximum(div, 0) * self._weights),
            np.mean(np.maximum(rad, 0) * self._weights),
            blob_growth(diameter, self._previous_diameter),
        ], dtype=float)
        raw = float(_raw_score(values, self.normalization, self.config))
        self._ema = raw if self._ema is None else self.config.ema * raw + (1 - self.config.ema) * self._ema
        result = dict(zip(COMPONENTS, map(float, values)))
        result.update(S=self._ema, flow_median=float(np.median(np.linalg.norm(flow, axis=-1))), flow_horizontal=float(np.mean(np.abs(flow[..., 0]))), frame_diff=float(np.mean(np.abs(gray - self._previous))), flow_valid=True)
        self._previous = gray.copy()
        self._previous_diameter = diameter
        self.last_flow, self.last_radial = flow, rad
        return result
