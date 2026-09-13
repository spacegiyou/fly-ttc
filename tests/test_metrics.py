"""Boundary and leakage tests for clip-level calibration and reporting."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from fly_ttc.eval.failure_tags import failure_tags
from fly_ttc.eval.lead_time import lead_times
from fly_ttc.eval.metrics import summarize_metrics
from fly_ttc.models.thresholds import choose_threshold


def test_post_event_score_cannot_create_success_or_peak():
    result = lead_times([7, 8, 9, 10, 11], [0, 1, 2, 100, 200], 3, 8.5, 10)
    assert result["s_peak"] == 2
    assert result["t_peak"] == 9
    assert result["lead_to_event_s"] == 1
    assert result["lead_to_alert_s"] == -0.5
    assert result["hit_event"] is False
    assert result["hit_alert"] is False
    assert result["s_at_event"] == 2
    assert result["s_at_alert"] == 1  # Never interpolate from a future frame.
    assert result["n_eval_frames"] == 3


def test_strict_threshold_and_early_deadline_boundaries():
    result = lead_times([7, 8.5, 9, 9.5, 10], [0, 1, 2, 3, 100], 1, 9, 10)
    assert result["hit_early_1500ms"] is False  # Equality S==theta is not a hit.
    assert result["hit_early_1000ms"] is True   # Equality t==deadline is observable.
    assert result["hit_early_500ms"] is True
    assert result["hit_alert"] is True


def test_unobserved_deadline_is_unavailable():
    result = lead_times([9.6, 9.8], [3, 4], 1, 9, 10)
    assert result["hit_early_500ms"] is None
    assert result["s_at_alert"] is None
    assert result["s_at_event"] == 4


def test_negative_uses_entire_window_and_no_event_leads():
    result = lead_times([1, 2, 3], [1, 2, 5], 2, None, None)
    assert result["false_positive"] is True
    assert result["s_peak"] == 5
    assert result["hit_event"] is None
    assert result["lead_to_event_s"] is None


def test_missing_finite_pre_event_frames_are_not_false_negatives():
    result = lead_times([10, 11], [100, 200], 1, 9, 10)
    assert result["n_eval_frames"] == 0
    assert result["predicted_positive"] is None
    assert result["s_peak"] is None


@pytest.mark.parametrize("values,target,theta,expected", [
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 0.1, 8, 0.1),
    ([0, 0, 0, 1, 1], 0.2, 1, 0.0),
    ([1, 2, 3], 0.1, 3, 0.0),
    ([2, 2, 2], 0.0, 2, 0.0),
])
def test_threshold_respects_clip_fpr_and_ties(values, target, theta, expected):
    result = choose_threshold(values, target)
    assert result["theta"] == theta
    assert result["val_fpr"] == pytest.approx(expected)
    assert result["val_fpr"] <= target
    # Moving below the selected distinct cutoff must violate the target.
    assert np.mean(np.asarray(values) > np.nextafter(theta, -np.inf)) > target


@pytest.mark.parametrize("values", [[], [1, np.nan], [np.inf]])
def test_invalid_threshold_input_is_explicit(values):
    with pytest.raises(ValueError, match="validation-negative"):
        choose_threshold(values)


def test_metrics_do_not_report_calibration_as_heldout_performance():
    frame = pd.DataFrame([
        dict(video_id="v0", label=0, s_peak=0, theta_used=1, split="validation", status="ok"),
        dict(video_id="v1", label=1, s_peak=10, theta_used=1, split="validation", status="ok"),
        dict(video_id="e0", label=0, s_peak=3, theta_used=1, split="evaluation", status="ok"),
        dict(video_id="e1", label=1, s_peak=2, theta_used=1, split="evaluation", status="ok"),
        dict(video_id="bad", label=1, s_peak=99, theta_used=1, split="evaluation", status="decode_error"),
    ])
    result = summarize_metrics(frame)
    assert result["headline"]["auroc"] == 0
    assert result["validation"]["auroc"] == 1
    assert result["headline"]["fpr"] == 1
    assert result["validation"]["fpr"] == 0
    assert result["all"]["n_scored"] == 4
    assert result["headline"]["n_excluded"] == 1


def test_no_split_or_one_class_does_not_invent_metrics():
    result = summarize_metrics(pd.DataFrame([dict(label=0, status="ok", s_peak=1, theta_used=2)]))
    assert result["headline"]["n_scored"] == 0
    assert result["headline"]["auroc"] is None
    assert result["all"]["auprc"] is None
    assert result["all"]["fpr"] == 0


def test_failure_tags_ignore_postevent_and_missing_diagnostics():
    frames = pd.DataFrame(dict(t=[8.5, 9.5, 10.0], S=[0, 0, 100], S_rad=[0, 0, 100], S_div=[0, 0, 100]))
    row = dict(label=1, time_of_event=10, light="Dark", weather="Clear", theta_used=1)
    tags = failure_tags(frames, row, {"S_rad_p95": 2}, {})
    assert tags == ["rear_or_nonexpanding", "low_visibility"]
    assert "ego_shake" not in tags
    assert "sideswipe_suspect" not in tags


def test_report_html_renders_tables_and_rejects_active_metadata():
    from fly_ttc.viz.report import _html_report

    source = """## Metrics

**Bold** and `code`

| Metric | Value |
| --- | --- |
| n | 2 |

```bash
pytest -q
```

<script>alert('unsafe')</script>
[bad](javascript:alert(1))
![remote](https://example.com/tracker.png)
![safe](figures/positive_traces.png)
"""
    rendered = _html_report(source, ["positive_traces.png"])
    assert "<strong>Bold</strong>" in rendered
    assert "<code>code</code>" in rendered
    assert "<table>" in rendered
    assert "<pre><code>pytest -q" in rendered
    assert "<script>" not in rendered
    assert 'href="javascript:' not in rendered
    assert 'src="https://' not in rendered
    assert 'src="figures/positive_traces.png"' in rendered


def test_report_produces_six_figures_and_preserves_negative_panel(tmp_path):
    from fly_ttc.viz.report import make_report

    run = tmp_path / "synthetic"
    (run / "frames").mkdir(parents=True)
    (run / "_report_frames").mkdir()
    records = []
    for label in [0, 1]:
        for index in range(2):
            video_id = f"{label}{index:04d}"
            row = dict(video_id=video_id, label=label, status="ok", split="evaluation",
                       scene="Urban", light="Normal", theta_used=1, time_of_alert=2 if label else None,
                       time_of_event=3 if label else None, failure_tags="good_loom" if label else "")
            values = [0, 2, 100] if label else [0, 0.2, 0.4]
            row.update(lead_times([1, 2, 3], values, 1, row["time_of_alert"], row["time_of_event"]))
            records.append(row)
            pd.DataFrame(dict(t=[1, 2, 3], S=values)).to_parquet(
                run / ("frames" if label else "_report_frames") / f"{video_id}.parquet", index=False)
    pd.DataFrame(records).to_csv(run / "clip_scores.csv", index=False)
    (run / "config.yaml").write_text(yaml.safe_dump(dict(seed=0, report=dict(n_traces=2, dpi=45))))
    (run / "threshold.yaml").write_text(yaml.safe_dump(dict(theta=1, val_fpr=0, fpr_target=0.1)))
    (run / "validation_checks.json").write_text(json.dumps(dict(
        pytest=dict(passed=1, failed=0), sha256_match_count=4, real_video_count=4,
        positive_pre_event_peak_hit_and_clock_checks=2, audit_errors=[],
        validation_threshold_recomputed=dict(theta=1, val_fpr=0),
    )))
    output = make_report(run)
    assert output.exists()
    assert "전체 초파리 뇌를 시뮬레이션한 것이 아니다" in output.read_text()
    assert "[전체 검증 기록](validation_checks.json)" in output.read_text()
    assert "| 검증 오류 | 0개 |" in output.read_text()
    assert "```json" not in output.read_text()
    assert len(list((run / "figures").glob("*.png"))) == 6
    assert (run / "REPORT.html").exists()
    assert not (run / "_report_frames").exists()
    original = (run / "figures" / "negative_traces.png").read_bytes()
    make_report(run)
    assert (run / "figures" / "negative_traces.png").read_bytes() == original
    assert json.loads((run / "report_selection.json").read_text())["negative"] == ["00000", "00001"]
