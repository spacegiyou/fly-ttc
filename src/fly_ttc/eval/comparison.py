"""Frozen-model comparison with clip-paired, class-stratified uncertainty."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .metrics import _aggregate, bool_values, valid_clips


MODELS = ("v0_original", "raw_flow_no_blob", "affine_residual_no_blob", "frame_difference")
SPLITS = ("validation", "exploratory", "confirmation")


def wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    """Two-sided 95% Wilson interval; unavailable when no trials exist."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Require 0 <= successes <= total")
    if not total:
        return None, None
    z = NormalDist().inv_cdf(0.975)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return float(max(0, center - half)), float(min(1, center + half))


def _predictions(frame: pd.DataFrame) -> np.ndarray:
    if "predicted_positive" in frame:
        return bool_values(frame.predicted_positive).to_numpy(dtype=float)
    if "theta_used" in frame:
        theta = pd.to_numeric(frame.theta_used, errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(theta), frame.s_peak.to_numpy(dtype=float) > theta, np.nan)
    return np.full(len(frame), np.nan)


def _auc(negative: np.ndarray, positive: np.ndarray) -> np.ndarray:
    """Rank-sum AUC, including half credit for tied scores, along the last axis."""
    n_neg, n_pos = negative.shape[-1], positive.shape[-1]
    ranks = rankdata(np.concatenate((negative, positive), axis=-1), axis=-1)
    return (ranks[..., n_neg:].sum(axis=-1) - n_pos * (n_pos + 1) / 2) / (n_neg * n_pos)


def _interval(samples: np.ndarray) -> tuple[float, float]:
    return tuple(float(v) for v in np.quantile(samples, [0.025, 0.975]))


def _auc_interval(frame: pd.DataFrame, iterations: int, seed: int):
    negative = frame.loc[frame.label.eq(0), "s_peak"].to_numpy(dtype=float)
    positive = frame.loc[frame.label.eq(1), "s_peak"].to_numpy(dtype=float)
    if not len(negative) or not len(positive):
        return None, None
    rng = np.random.default_rng(seed)
    neg_indices = rng.integers(len(negative), size=(iterations, len(negative)))
    pos_indices = rng.integers(len(positive), size=(iterations, len(positive)))
    return _interval(_auc(negative[neg_indices], positive[pos_indices]))


def _paired_comparison(clips: pd.DataFrame, iterations: int, seed: int) -> dict:
    candidate, comparator = "affine_residual_no_blob", "raw_flow_no_blob"
    requested = clips[clips.split.eq("confirmation") & clips.model.isin([candidate, comparator])]
    valid = valid_clips(requested)
    left = valid[valid.model.eq(candidate)].set_index("video_id").sort_index()
    right = valid[valid.model.eq(comparator)].set_index("video_id").sort_index()
    ids = left.index.intersection(right.index).sort_values()
    left, right = left.loc[ids], right.loc[ids]
    all_ids = set(requested.video_id.astype(str))
    labels = left.label.to_numpy(dtype=int)
    result = {
        "split": "confirmation", "candidate": candidate, "comparator": comparator,
        "direction": "candidate minus comparator; positive delta AUROC/TPR and negative delta FPR favor candidate",
        "n_requested_union": len(all_ids), "n_match": len(ids), "n_excluded": len(all_ids - set(ids)),
        "n_positive": int(np.sum(labels == 1)), "n_negative": int(np.sum(labels == 0)),
        "matched_ids": list(map(str, ids)), "excluded_ids": sorted(all_ids - set(ids)),
        "bootstrap_iterations": iterations, "bootstrap_seed": seed,
        "method": "paired within-class clip resampling, percentile 95% intervals; calibration fixed",
        "outcomes": {},
    }
    rng = np.random.default_rng(seed)
    scores_left, scores_right = left.s_peak.to_numpy(dtype=float), right.s_peak.to_numpy(dtype=float)
    neg, pos = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    outcome = {"estimate": None, "ci_low": None, "ci_high": None, "n": len(ids)}
    if len(neg) and len(pos):
        sampled_neg = neg[rng.integers(len(neg), size=(iterations, len(neg)))]
        sampled_pos = pos[rng.integers(len(pos), size=(iterations, len(pos)))]
        delta = _auc(scores_left[sampled_neg], scores_left[sampled_pos]) - _auc(
            scores_right[sampled_neg], scores_right[sampled_pos]
        )
        low, high = _interval(delta)
        outcome.update(estimate=float(_auc(scores_left[neg], scores_left[pos]) - _auc(
            scores_right[neg], scores_right[pos])), ci_low=low, ci_high=high)
    result["outcomes"]["delta_auroc"] = outcome
    predictions_left, predictions_right = _predictions(left), _predictions(right)
    decision_valid = np.isfinite(predictions_left) & np.isfinite(predictions_right)
    result["n_decision_match"] = int(decision_valid.sum())
    result["n_decision_excluded"] = int(len(ids) - decision_valid.sum())
    for label, name in ((0, "delta_fpr"), (1, "delta_tpr")):
        difference = (predictions_left - predictions_right)[(labels == label) & decision_valid]
        outcome = {"estimate": None, "ci_low": None, "ci_high": None, "n": len(difference)}
        if len(difference):
            indices = rng.integers(len(difference), size=(iterations, len(difference)))
            low, high = _interval(difference[indices].mean(axis=1))
            outcome.update(estimate=float(difference.mean()), ci_low=low, ci_high=high)
        result["outcomes"][name] = outcome
    return result


def summarize_comparison(clips: pd.DataFrame, config: dict) -> dict:
    """Report every frozen model and use only confirmation for the primary contrast.

    Selected calibration FPR/TPR receive no inferential interval. Exploratory
    intervals describe sampling variability and do not undo adaptive reuse.
    No scores, thresholds, or model settings are modified by this function.
    """
    required = {"video_id", "model", "split", "label", "s_peak"}
    if not required.issubset(clips):
        raise ValueError(f"Missing comparison columns: {sorted(required - set(clips))}")
    clips = clips.copy()
    clips["video_id"] = clips.video_id.astype(str)
    if not clips.model.isin(MODELS).all() or not clips.split.isin(SPLITS).all():
        raise ValueError("Unknown frozen model or comparison split")
    if clips.duplicated(["model", "video_id"]).any():
        raise ValueError("Duplicate model/video_id rows would overweight clips")
    if (clips.groupby("video_id").label.nunique(dropna=False) > 1).any():
        raise ValueError("Mismatched labels across models")
    if (clips.groupby("video_id").split.nunique(dropna=False) > 1).any():
        raise ValueError("A video_id occurs in multiple comparison splits")
    bootstrap = config.get("followup", {})
    iterations, seed = int(bootstrap.get("bootstrap_iterations", 2000)), int(bootstrap.get("bootstrap_seed", 0))
    if iterations < 1:
        raise ValueError("bootstrap_iterations must be positive")
    results = []
    for split in SPLITS:
        for model in MODELS:
            subset = clips[clips.split.eq(split) & clips.model.eq(model)].sort_values("video_id")
            valid = valid_clips(subset)
            summary = _aggregate(subset)
            summary = {key: value for key, value in summary.items() if not isinstance(value, dict)}
            summary.update(model=model, split=split, auroc_ci_low=None, auroc_ci_high=None)
            if split != "validation":
                summary["auroc_ci_low"], summary["auroc_ci_high"] = _auc_interval(valid, iterations, seed)
            for label, metric, prefix in ((0, "fpr", "fpr"), (1, "true_positive_rate", "tpr")):
                values = _predictions(valid[valid.label.eq(label)])
                values = values[np.isfinite(values)]
                n, k = len(values), int(values.sum())
                summary[f"{metric}_n"] = n
                summary[f"{prefix}_count"] = k
                summary[f"{prefix}_ci_low"], summary[f"{prefix}_ci_high"] = (
                    wilson_interval(k, n) if split != "validation" else (None, None)
                )
            results.append(summary)
    return {
        "schema_version": 1, "models": list(MODELS), "splits": list(SPLITS), "results": results,
        "primary_comparison": _paired_comparison(clips, iterations, seed),
        "metric_definitions": {
            "auprc": "average precision of pre-event clip maximum; prevalence dependent",
            "decision": "strict S > frozen theta at any allowed frame; all allowed frames for negatives",
            "lead": "event or alert time minus pre-event peak time, not first threshold crossing",
            "validation": "selected calibration performance, no inferential FPR/TPR interval",
            "exploratory": "previously inspected 80 clips; intervals do not undo adaptive design",
            "confirmation": "new clip-disjoint sample; trip/location independence unverified",
            "intervals": "Wilson FPR/TPR; within-class percentile bootstrap AUROC; 95%, fixed calibration",
            "primary": "one prespecified AUROC contrast; FPR/TPR deltas are secondary, no multiplicity claim",
        },
    }
