"""Statistical and frozen-data safeguards for post-hoc subgroup reporting."""

import hashlib
import json

import pandas as pd
import pytest
import yaml

from fly_ttc.eval.subgroups import analyze_subgroups, bootstrap_auroc, wilson_interval


def test_wilson_handles_zero_counts_without_claiming_certainty():
    assert wilson_interval(0, 0) == (None, None)
    low, high = wilson_interval(0, 4)
    assert low == pytest.approx(0)
    assert high == pytest.approx(0.4898908365)
    low, high = wilson_interval(1, 4)
    assert low == pytest.approx(0.0455872608)
    assert high == pytest.approx(0.6993581574)
    with pytest.raises(ValueError):
        wilson_interval(5, 4)


def test_bootstrap_ties_absent_class_and_seed():
    assert bootstrap_auroc([0, 0, 1, 1], [3, 3, 3, 3]) == (0.5, 0.5)
    assert bootstrap_auroc([0, 0, 1, 1], [1, 2, 3, 4]) == (1.0, 1.0)
    assert bootstrap_auroc([0, 0], [1, 2]) == (None, None)
    labels, scores = [0, 0, 0, 1, 1], [0, 1, 4, 2, 3]
    assert bootstrap_auroc(labels, scores, 7, 200) == bootstrap_auroc(labels, scores, 7, 200)
    with pytest.raises(ValueError, match="positive"):
        bootstrap_auroc(labels, scores, iterations=0)


@pytest.fixture
def frozen_run(tmp_path):
    run = tmp_path / "v0"
    run.mkdir()
    # Evaluation has one true/false positive and one true/false negative.
    # Calibration performance deliberately differs to expose accidental mixing.
    frame = pd.DataFrame([
        dict(video_id="00001", label=1, s_peak=2, split="evaluation", light="Normal", failure_tags=""),
        dict(video_id="00002", label=1, s_peak=1, split="evaluation", light="Dark", failure_tags="low_visibility"),
        dict(video_id="00003", label=0, s_peak=3, split="evaluation", light="Normal", failure_tags=""),
        dict(video_id="00004", label=0, s_peak=0, split="evaluation", light="Dark", failure_tags="low_visibility"),
        dict(video_id="00005", label=1, s_peak=100, split="validation", light="Normal", failure_tags=""),
        dict(video_id="00006", label=0, s_peak=-100, split="validation", light="Normal", failure_tags=""),
    ])
    frame["theta_used"], frame["scene"], frame["weather"], frame["status"] = 1.0, "Highway", "Clear", "ok"
    frame["predicted_positive"] = frame.s_peak > 1
    frame["hit_alert"] = frame.predicted_positive.where(frame.label.eq(1), None)
    frame.to_csv(run / "clip_scores.csv", index=False)
    (run / "threshold.yaml").write_text(yaml.safe_dump({
        "theta": 1.0, "evaluation_ids": ["00001", "00002", "00003", "00004"],
        "validation_ids": ["00005", "00006"],
    }))
    return run


def test_audit_preserves_source_split_threshold_and_negative_tags(frozen_run, tmp_path):
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in frozen_run.iterdir()}
    result = analyze_subgroups(frozen_run, tmp_path / "audit", bootstrap_iterations=100)
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in frozen_run.iterdir()}
    assert before == after == result["source_sha256"]
    evaluation = next(row for row in result["subgroups"] if row["split"] == "evaluation" and row["subgroup"] == "All clips")
    assert [evaluation[key] for key in ["tp", "fp", "tn", "fn"]] == [1, 1, 1, 1]
    assert evaluation["auroc"] == pytest.approx(0.5)
    assert evaluation["theta"] == 1
    audit = result["tag_audit"]["evaluation"]
    assert audit["tag_counts_by_outcome"]["low_visibility"]["fn"] == 1
    assert audit["tag_counts_by_outcome"]["low_visibility"]["tn"] == 1
    assert audit["low_visibility_rule_mismatch_ids"] == []
    assert json.loads((tmp_path / "audit" / "analysis.json").read_text())["source_unchanged"]
    assert (tmp_path / "audit" / "REPORT.md").is_file()
    assert (tmp_path / "audit" / "subgroup_intervals.png").is_file()


@pytest.mark.parametrize("column,value,error", [
    ("theta_used", 2, "fixed theta"),
    ("split", "validation", "IDs disagree"),
    ("predicted_positive", False, "Source predictions"),
])
def test_audit_rejects_mismatched_frozen_inputs(frozen_run, tmp_path, column, value, error):
    path = frozen_run / "clip_scores.csv"
    frame = pd.read_csv(path, dtype={"video_id": str})
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=error):
        analyze_subgroups(frozen_run, tmp_path / "audit", bootstrap_iterations=10)
    assert not (tmp_path / "audit").exists()


def test_audit_cannot_write_inside_original_run(frozen_run):
    with pytest.raises(ValueError, match="outside"):
        analyze_subgroups(frozen_run, frozen_run / "review")
