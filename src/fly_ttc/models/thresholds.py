"""Clip-level validation calibration with the strict ``score > theta`` rule."""

import math

import numpy as np


def choose_threshold(negative_clip_maxima, fpr_target=0.10) -> dict:
    """Return the lowest observed cutoff meeting the empirical negative FPR.

    A clip is a false positive when *any* frame is strictly greater than theta.
    Selecting an order statistic (without interpolating quantiles) handles small
    validation sets and tied maxima exactly. Calibration inputs must contain
    validation negatives only; the caller records their IDs in threshold.yaml.
    """
    values = np.asarray(negative_clip_maxima, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("At least one finite validation-negative clip maximum is required")
    if not 0 <= fpr_target < 1:
        raise ValueError("fpr_target must satisfy 0 <= fpr_target < 1")
    allowed = math.floor(float(fpr_target) * len(values) + 1e-12)
    theta = float(np.sort(values)[len(values) - allowed - 1])
    count = int(np.count_nonzero(values > theta))
    return {
        "theta": theta, "fpr_target": float(fpr_target),
        "val_fpr": count / len(values), "n_val_negatives": int(len(values)),
        "n_val_false_positives": count, "decision_rule": "S > theta",
        "calibration_unit": "clip maximum",
    }
