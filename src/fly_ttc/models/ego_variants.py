"""One shared flow extraction for a prespecified, non-neural ablation."""

from dataclasses import replace

import numpy as np
import pandas as pd

from fly_ttc.models.v0_expansion import ExpansionScorer, V0Config
from fly_ttc.preprocess.ego_motion import EgoMotionConfig, estimate_background_flow
from fly_ttc.preprocess.expansion import center_weights, divergence, radial_flow
from fly_ttc.preprocess.flow import clip_flow_magnitude

MODELS = ("v0_original", "raw_flow_no_blob", "affine_residual_no_blob", "frame_difference")


class EgoVariantScorer:
    """Keep original v0 components exact; derive residual flow without labels.

    Both flow-only arms exclude blob, so that their contrast isolates global
    affine subtraction. This is not a reconstruction of physical ego-motion.
    Failed coherent-background fits pass through raw flow, with validity flags.
    """

    def __init__(self, config: V0Config, flow_config: dict, ego_config: EgoMotionConfig):
        self.config = config
        self.ego_config = ego_config
        self.base = ExpansionScorer(config, flow_config)
        self.reset()

    def reset(self):
        self.base.reset()
        self.last_flow = self.last_radial = None
        self.snapshot_fields = {}
        self._weights = None

    def update(self, gray):
        result = self.base.update(gray)
        self.last_flow, self.last_radial = self.base.last_flow, self.base.last_radial
        if not result["flow_valid"]:
            return {**result, "residual_div": 0., "residual_rad": 0.,
                    "fit_valid": False, "fit_reason": "initial_frame", "inlier_fraction": 0.,
                    "fit_error_px": np.nan, "background_median": 0.}
        fit = estimate_background_flow(self.base.last_unclipped_flow, self.ego_config)
        residual = clip_flow_magnitude(fit["residual_flow"], self.config.flow_clip_percentile)
        self.last_flow, self.last_radial = residual, radial_flow(residual)
        if self._weights is None:
            self._weights = (center_weights(gray.shape, self.config.center_edge_weight)
                             if self.config.center_weight else np.ones_like(gray))
        result.update(
            residual_div=float(np.mean(np.maximum(divergence(residual, self.config.sobel_ksize), 0) * self._weights)),
            residual_rad=float(np.mean(np.maximum(self.last_radial, 0) * self._weights)),
            fit_valid=bool(fit["fit_valid"]), fit_reason=fit["fit_reason"],
            inlier_fraction=float(fit["inlier_fraction"]), fit_error_px=float(fit["fit_error_px"]),
            background_median=float(np.median(np.linalg.norm(fit["background_flow"], axis=-1))),
        )
        # Show the exact unclipped decomposition; the scorer clips residuals after subtraction.
        self.snapshot_fields = {"raw_flow": self.base.last_unclipped_flow.copy(),
                                "background_flow": fit["background_flow"].copy(),
                                "residual_flow": fit["residual_flow"].copy()}
        return result


def model_components(frames: pd.DataFrame, model: str) -> pd.DataFrame:
    """Map shared frame rows onto the existing negative-calibration interface."""
    out = frames.copy()
    if model == "v0_original":
        return out
    out["S_blob"] = 0.0
    if model == "raw_flow_no_blob":
        return out
    if model == "affine_residual_no_blob":
        out["S_div"], out["S_rad"] = frames["residual_div"], frames["residual_rad"]
    elif model == "frame_difference":
        out["S_div"], out["S_rad"] = frames["frame_diff"], 0.0
    else:
        raise ValueError(f"Unknown ablation model: {model}")
    return out


def model_config(base: V0Config, model: str) -> V0Config:
    if model == "v0_original":
        return base
    if model not in MODELS:
        raise ValueError(f"Unknown ablation model: {model}")
    return replace(base, score="divergence" if model == "frame_difference" else "combined", blob_weight=0.)
