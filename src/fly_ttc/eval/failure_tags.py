"""Transparent descriptive heuristics, not ground-truth collision classes."""

from __future__ import annotations

import numpy as np
import pandas as pd


FAILURE_TAGS = (
    "sideswipe_suspect", "rear_or_nonexpanding", "ego_shake",
    "low_visibility", "good_loom",
)


def failure_tags(frame_df: pd.DataFrame, row, reference: dict, config: dict) -> list[str]:
    """Tag clips using raw components and validation-negative p95 references.

    Rules: sideswipe = final 1.5 s radial <= 0.25 * negative radial p95,
    and horizontal flow > negative horizontal p95; nonexpanding = final 1.5 s
    divergence <= epsilon; shake = median flow > negative flow-median p95;
    visibility = Dark/Twilight/Rain with no pre-event threshold crossing;
    good loom = final 1 s mean radial > negative radial p95. Missing diagnostic
    columns/reference statistics suppress dependent tags, never imply zero.
    All event rules exclude the event timestamp itself and subsequent frames.
    """
    if frame_df.empty:
        return []
    cfg = config.get("failure_tags", {})
    event = row.get("time_of_event")
    positive = int(row.get("label", 0)) == 1 and pd.notna(event)
    pre = frame_df[frame_df.t < float(event)] if positive else frame_df
    if pre.empty:
        return []
    near = pre[pre.t >= float(event) - cfg.get("pre_window_s", 1.5)] if positive else pre
    tags = []

    def mean(df, column):
        if column not in df or df.empty:
            return None
        value = float(pd.to_numeric(df[column], errors="coerce").mean())
        return value if np.isfinite(value) else None

    def reference_value(key):
        value = reference.get(key)
        return float(value) if value is not None and np.isfinite(value) else None

    rad, horiz = mean(near, "S_rad"), mean(near, "flow_horizontal")
    rad95, horiz95 = reference_value("S_rad_p95"), reference_value("flow_horizontal_p95")
    if (positive and rad is not None and horiz is not None and rad95 is not None
            and horiz95 is not None and rad <= cfg.get("radial_low_factor", 0.25) * rad95
            and horiz > horiz95):
        tags.append("sideswipe_suspect")
    divergence = mean(near, "S_div")
    if positive and divergence is not None and divergence <= cfg.get("divergence_epsilon", 1e-4):
        tags.append("rear_or_nonexpanding")
    shake = mean(pre, "flow_median")
    shake95 = reference_value("flow_median_p95")
    if shake is not None and shake95 is not None and shake > shake95:
        tags.append("ego_shake")
    low_visibility = (
        str(row.get("light", row.get("light_conditions", ""))).casefold() in {"dark", "twilight"}
        or str(row.get("weather", "")).casefold() == "rain"
    )
    theta = row.get("theta_used")
    if low_visibility and theta is not None and pd.notna(theta) and "S" in pre:
        finite = pd.to_numeric(pre.S, errors="coerce").dropna()
        if len(finite) and not (finite > float(theta)).any():
            tags.append("low_visibility")
    if positive and rad95 is not None:
        last = pre[pre.t >= float(event) - cfg.get("good_loom_window_s", 1.0)]
        final_rad = mean(last, "S_rad")
        if final_rad is not None and final_rad > rad95:
            tags.append("good_loom")
    return tags
