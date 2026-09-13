"""Causal, pre-event clip summaries shared by all detector versions."""

from __future__ import annotations

import numpy as np


def _optional_time(value):
    return None if value is None or not np.isfinite(float(value)) else float(value)


def lead_times(
    t, S, theta, t_alert, t_event, lead_windows_ms=(500, 1000, 1500),
    alert_window_pre_s=1.5,
) -> dict:
    """Summarize scores, excluding ``t >= t_event`` from every event metric.

    ``s_at_alert`` is the latest available sample at/before alert; ``s_at_event``
    is the latest sample strictly before event. No future interpolation is used.
    Early hit means any observed threshold crossing by event minus the window.
    A window preceding every analyzed frame is unavailable, rather than a miss.
    A missing event denotes a negative clip; callers must exclude unlabeled
    positive clips before calling this function.
    """
    times, scores = np.asarray(t, dtype=float), np.asarray(S, dtype=float)
    if times.ndim != 1 or scores.ndim != 1 or times.shape != scores.shape:
        raise ValueError("t and S must be equally sized one-dimensional arrays")
    if not np.isfinite(theta):
        raise ValueError("theta must be finite")
    if alert_window_pre_s < 0:
        raise ValueError("alert_window_pre_s must be nonnegative")
    event, alert = _optional_time(t_event), _optional_time(t_alert)
    valid = np.isfinite(times) & np.isfinite(scores)
    if event is not None:
        valid &= times < event
    times, scores = times[valid], scores[valid]
    order = np.argsort(times, kind="stable")
    times, scores = times[order], scores[order]
    above = scores > theta
    out = {
        "t_peak": None, "s_peak": None, "s_at_alert": None,
        "s_at_event": None, "lead_to_alert_s": None, "lead_to_event_s": None,
        "hit_alert": None, "hit_event": None, "false_positive": None,
        "predicted_positive": bool(above.any()) if len(scores) else None,
        "n_eval_frames": int(len(scores)),
    }
    for window in lead_windows_ms:
        if float(window) < 0:
            raise ValueError("lead windows must be nonnegative")
        out[f"hit_early_{int(window)}ms"] = None
    if not len(scores):
        return out
    peak = int(np.argmax(scores))
    out.update(t_peak=float(times[peak]), s_peak=float(scores[peak]))
    if event is None:
        out["false_positive"] = bool(above.any())
        return out
    out.update(
        lead_to_event_s=float(event - times[peak]),
        s_at_event=float(scores[-1]), hit_event=bool(above.any()),
    )
    if alert is not None:
        out["lead_to_alert_s"] = float(alert - times[peak])
        before_alert = times <= alert
        if before_alert.any():
            out["s_at_alert"] = float(scores[before_alert][-1])
        alert_window = times >= alert - alert_window_pre_s
        out["hit_alert"] = bool(above[alert_window].any()) if alert_window.any() else None
    for window in lead_windows_ms:
        observable = times <= event - float(window) / 1000.0
        if observable.any():
            out[f"hit_early_{int(window)}ms"] = bool(above[observable].any())
    return out
