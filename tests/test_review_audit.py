"""Audit safeguards against event leakage and outcome/state conflation."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "review_audit", Path(__file__).parents[1] / "scripts/audit_review_20260913.py"
)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def test_peak_state_excludes_event_and_post_event_without_changing_decision():
    row = pd.Series(dict(video_id="001", split="confirmation", label=1,
                         time_of_event=2.0, predicted_positive=False))
    frames = pd.DataFrame(dict(t=[0.5, 1.0, 2.0, 2.5], S=[1.0, 2.0, 100.0, 1000.0],
                               fit_valid=[True, False, True, True],
                               fit_reason=["accepted", "insufficient_inliers", "accepted", "accepted"],
                               above_theta=[False, False, True, True]))
    result = audit.peak_fit_record(frames, row)
    assert result["t_peak"] == 1.0
    assert result["peak_fit_valid"] is False
    assert result["predicted_positive"] is False
    assert result["n_frames"] == 2
    assert result["rejected_frames"] == 1
    assert result["above_theta_accepted_frames"] == 0
    assert result["above_theta_rejected_frames"] == 0


def test_negative_window_does_not_require_an_event_and_empty_positive_is_error():
    row = pd.Series(dict(video_id="002", label=0, time_of_event=float("nan")))
    frame = pd.DataFrame(dict(t=[1.0, 2.0]))
    pd.testing.assert_frame_equal(audit.eligible_frames(frame, row), frame)
    row["label"], row["time_of_event"] = 1, 0.0
    with pytest.raises(ValueError, match="No eligible frames"):
        audit.eligible_frames(frame, row)


def test_metric_recalculation_preserves_recorded_decisions_separately_from_ranks():
    frame = pd.DataFrame(dict(label=[0, 0, 1, 1], s_peak=[1, 2, 3, 4],
                               predicted_positive=[True, False, False, True]))
    result = audit.metrics(frame)
    assert result["AUROC"] == 1.0
    assert result["AP"] == 1.0
    assert result["FPR"] == 0.5
    assert result["TPR"] == 0.5
