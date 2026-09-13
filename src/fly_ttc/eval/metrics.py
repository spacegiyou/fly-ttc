"""Clip-level metrics with validation separated from held-out evaluation."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def bool_values(series: pd.Series) -> pd.Series:
    """Parse nullable booleans without treating the string 'False' as true."""
    def convert(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, str):
            lowered = value.casefold().strip()
            if lowered in {"true", "1", "1.0"}:
                return 1.0
            if lowered in {"false", "0", "0.0"}:
                return 0.0
            return np.nan
        return float(bool(value))
    return series.map(convert)


def valid_clips(clips: pd.DataFrame) -> pd.DataFrame:
    if clips.empty or "s_peak" not in clips:
        return clips.iloc[:0].copy()
    keep = pd.to_numeric(clips.s_peak, errors="coerce").map(np.isfinite)
    keep &= clips.label.isin([0, 1])
    if "status" in clips:
        keep &= clips.status.eq("ok")
    if "time_of_event" in clips:
        keep &= clips.label.eq(0) | pd.to_numeric(clips.time_of_event, errors="coerce").notna()
    return clips[keep].copy()


def tag_values(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None or pd.isna(value) or not str(value).strip():
        return []
    text = str(value).strip()
    if text.startswith("["):
        try:
            return list(json.loads(text))
        except (ValueError, TypeError):
            pass
    return [part.strip() for part in text.replace(",", ";").split(";") if part.strip()]


def _aggregate(clips: pd.DataFrame) -> dict:
    valid = valid_clips(clips)
    pos = valid[valid.label.eq(1)] if "label" in valid else valid
    neg = valid[valid.label.eq(0)] if "label" in valid else valid
    result = {
        "n_requested": int(len(clips)), "n_scored": int(len(valid)),
        "n_excluded": int(len(clips) - len(valid)),
        "n_positive": int(len(pos)), "n_negative": int(len(neg)),
        "auroc": None, "auprc": None, "fpr": None,
        "true_positive_rate": None,
        "status_counts": {str(k): int(v) for k, v in clips.get("status", pd.Series(dtype=str)).value_counts().items()},
    }
    if len(pos) and len(neg):
        result["auroc"] = float(roc_auc_score(valid.label, valid.s_peak))
        result["auprc"] = float(average_precision_score(valid.label, valid.s_peak))
    for subset, metric in [(neg, "fpr"), (pos, "true_positive_rate")]:
        if len(subset):
            if "predicted_positive" in subset:
                predictions = bool_values(subset.predicted_positive).dropna()
            elif "theta_used" in subset:
                usable = subset[pd.to_numeric(subset.theta_used, errors="coerce").notna()]
                predictions = (usable.s_peak > usable.theta_used).astype(float)
            else:
                predictions = pd.Series(dtype=float)
            result[metric] = float(predictions.mean()) if len(predictions) else None
            result[f"{metric}_n"] = int(len(predictions))
    for column in ("lead_to_alert_s", "lead_to_event_s"):
        values = pd.to_numeric(pos.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
        base = column.removesuffix("_s")
        result[f"{base}_n"] = int(len(values))
        result[f"{base}_median_s"] = float(values.median()) if len(values) else None
        result[f"{base}_q25_s"] = float(values.quantile(0.25)) if len(values) else None
        result[f"{base}_q75_s"] = float(values.quantile(0.75)) if len(values) else None
        result[f"{base}_iqr_s"] = float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else None
    for column in ("hit_alert", "hit_event", *sorted(c for c in clips if c.startswith("hit_early_"))):
        values = bool_values(pos.get(column, pd.Series(dtype=float))).dropna()
        result[f"{column}_rate"] = float(values.mean()) if len(values) else None
        result[f"{column}_n"] = int(len(values))
    counts = Counter(tag for value in valid.get("failure_tags", []) for tag in set(tag_values(value)))
    result["failure_tag_counts"] = {tag: int(n) for tag, n in sorted(counts.items())}
    result["failure_tag_rates"] = {tag: n / len(valid) for tag, n in sorted(counts.items())} if len(valid) else {}
    return result


def summarize_metrics(clip_scores: pd.DataFrame) -> dict:
    """Keep calibration clips out of the headline, including on small runs.

    With no split column, only overall metrics can be reported. Validation and
    overall values remain descriptive, never masquerade as held-out estimates.
    """
    split = clip_scores.get("split", pd.Series("unspecified", index=clip_scores.index))
    evaluation = clip_scores[split.isin(["evaluation", "heldout", "train"])]
    validation = clip_scores[split.eq("validation")]
    heldout_metrics = _aggregate(evaluation)
    return {
        "headline_split": "evaluation", "headline": heldout_metrics,
        "evaluation": heldout_metrics, "validation": _aggregate(validation),
        "all": _aggregate(clip_scores),
        "metric_definitions": {
            "auprc": "average precision on clip maxima",
            "decision": "any pre-event frame S > theta (all analyzed frames for negatives)",
            "early_hit": "at least one observed crossing by event minus lead window",
            "lead": "reference time minus pre-event peak time; not first crossing time",
            "validation": "calibration set; not an unbiased performance estimate",
        },
    }


def stratified_metrics(clip_scores: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return scene/light strata for each split and overall, with denominators."""
    records = []
    if column not in clip_scores:
        return pd.DataFrame(columns=["split", column, "n_scored", "auroc", "auprc", "fpr"])
    split = clip_scores.get("split", pd.Series("unspecified", index=clip_scores.index))
    for split_name, subset in [
        ("all", clip_scores), ("evaluation", clip_scores[split.isin(["evaluation", "heldout", "train"])]),
        ("validation", clip_scores[split.eq("validation")]),
    ]:
        for value, group in subset.groupby(column, dropna=False, sort=True):
            summary = _aggregate(group)
            records.append({
                "split": split_name, column: "Unknown" if pd.isna(value) else value,
                **{key: summary[key] for key in (
                    "n_scored", "n_positive", "n_negative", "auroc", "auprc", "fpr",
                    "lead_to_alert_median_s", "hit_alert_rate",
                )},
            })
    return pd.DataFrame(records)
