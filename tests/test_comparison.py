"""Scientific-contract tests for matched frozen-model comparisons."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import roc_auc_score

from fly_ttc.eval.comparison import MODELS, SPLITS, _auc, summarize_comparison, wilson_interval


CONFIG = {"followup": {"bootstrap_iterations": 200, "bootstrap_seed": 42}, "report": {"dpi": 45}}


def example_clips():
    rows = []
    for split_index, split in enumerate(SPLITS):
        for index, (label, score) in enumerate([(0, 0.5), (0, 2.0), (1, 1.0), (1, 3.0)]):
            for model in MODELS:
                multiplier = 2 if model == "affine_residual_no_blob" else 1
                rows.append(dict(
                    video_id=f"{split_index}{index:04d}", label=label, model=model, split=split,
                    status="ok", s_peak=score * multiplier, theta_used=1.5 * multiplier,
                    predicted_positive=str(score > 1.5), time_of_event=10 if label else None,
                    hit_alert=str(score > 1.5) if label else None,
                    hit_early_1000ms=str(score > 1.5) if label else None,
                    lead_to_alert_s=0.5 if label else None, lead_to_event_s=1.5 if label else None,
                ))
    return pd.DataFrame(rows)


def test_paired_bootstrap_preserves_identical_rankings_and_decisions():
    clips = example_clips()
    original = clips.copy(deep=True)
    result = summarize_comparison(clips, CONFIG)
    primary = result["primary_comparison"]
    assert primary["n_match"] == 4
    assert primary["n_excluded"] == 0
    # A separate random bootstrap for the two models would give a nonzero CI.
    for outcome in primary["outcomes"].values():
        assert outcome["estimate"] == outcome["ci_low"] == outcome["ci_high"] == 0
    pd.testing.assert_frame_equal(clips, original)
    assert summarize_comparison(clips.sample(frac=1, random_state=7), CONFIG) == result


def test_only_confirmation_enters_primary_contrast():
    clips = example_clips()
    before = summarize_comparison(clips, CONFIG)["primary_comparison"]
    mask = clips.split.ne("confirmation") & clips.model.eq("affine_residual_no_blob")
    clips.loc[mask, "s_peak"] = 100 * (1 - clips.loc[mask, "label"])
    assert summarize_comparison(clips, CONFIG)["primary_comparison"] == before


def test_selected_calibration_has_no_wilson_interval_and_reports_all_arms():
    result = summarize_comparison(example_clips(), CONFIG)
    assert len(result["results"]) == len(MODELS) * len(SPLITS)
    for row in result["results"]:
        assert row["fpr"] == row["true_positive_rate"] == 0.5
        assert row["fpr_count"] == row["tpr_count"] == 1
        assert row["auroc"] == 0.75
        assert (row["fpr_ci_low"] is None) == (row["split"] == "validation")
        assert (row["tpr_ci_high"] is None) == (row["split"] == "validation")


def test_unmatched_and_invalid_clips_are_explicitly_excluded():
    clips = example_clips()
    clips.loc[clips.video_id.eq("20000") & clips.model.eq("affine_residual_no_blob"), "status"] = "decode_error"
    clips = clips[~(clips.video_id.eq("20003") & clips.model.eq("raw_flow_no_blob"))]
    result = summarize_comparison(clips, CONFIG)["primary_comparison"]
    assert result["n_requested_union"] == 4
    assert result["n_match"] == 2
    assert result["n_excluded"] == 2
    assert result["excluded_ids"] == ["20000", "20003"]
    assert result["matched_ids"] == ["20001", "20002"]


@pytest.mark.parametrize("problem", ["duplicate", "label", "split"])
def test_pairing_rejects_silent_overweighting_or_leakage(problem):
    clips = example_clips()
    if problem == "duplicate":
        clips = pd.concat([clips, clips.iloc[:1]], ignore_index=True)
    elif problem == "label":
        clips.loc[0, "label"] = 1
    else:
        clips.loc[0, "split"] = "confirmation"
    with pytest.raises(ValueError):
        summarize_comparison(clips, CONFIG)


def test_missing_decisions_are_not_treated_as_misses():
    clips = example_clips()
    clips.loc[clips.video_id.eq("20000") & clips.model.eq("raw_flow_no_blob"), "predicted_positive"] = None
    primary = summarize_comparison(clips, CONFIG)["primary_comparison"]
    assert primary["n_match"] == 4
    assert primary["n_decision_match"] == 3
    assert primary["outcomes"]["delta_fpr"]["n"] == 1


def test_tied_auc_matches_sklearn_and_wilson_handles_boundaries():
    negative, positive = np.asarray([1, 2, 2, 5]), np.asarray([1, 2, 4])
    expected = roc_auc_score([0] * len(negative) + [1] * len(positive), np.r_[negative, positive])
    assert float(_auc(negative, positive)) == pytest.approx(expected)
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-15)
    assert high == pytest.approx(0.2775328)
    assert wilson_interval(0, 0) == (None, None)


def test_absent_confirmation_and_single_class_do_not_invent_auc():
    clips = example_clips()
    clips = clips[clips.split.ne("confirmation") | clips.label.eq(0)]
    result = summarize_comparison(clips, CONFIG)
    assert result["primary_comparison"]["outcomes"]["delta_auroc"]["estimate"] is None
    result = summarize_comparison(clips[clips.split.ne("confirmation")], CONFIG)
    assert result["primary_comparison"]["n_match"] == 0
    assert result["primary_comparison"]["outcomes"]["delta_tpr"]["estimate"] is None


def test_report_saves_four_safe_figures_and_honest_split_language(tmp_path):
    from fly_ttc.viz.ego_report import make_ego_report

    example_clips().to_csv(tmp_path / "clip_scores.csv", index=False)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(CONFIG))
    (tmp_path / "protocol.json").write_text(json.dumps({"frozen": True}))
    (tmp_path / "thresholds.yaml").write_text("{}\n")
    path = make_ego_report(tmp_path)
    source = path.read_text()
    assert "독립 평가가 아니다" in source
    assert "개선한다는 근거는 충분하지 않았다" in source
    assert "신뢰구간을 붙이지 않는다" in source
    assert "[고정 프로토콜·ID·소스 해시](protocol.json)" in source
    assert "```json" not in source
    assert (tmp_path / "comparison.csv").exists()
    assert len(list((tmp_path / "figures").glob("*.png"))) == 4
    rendered = (tmp_path / "REPORT.html").read_text()
    assert "<table>" in rendered
    assert 'src="figures/comparison_metrics.png"' in rendered
    assert "<title>Fly-TTC Ego-Motion Comparison</title>" in rendered
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert len(metrics["results"]) == 12
