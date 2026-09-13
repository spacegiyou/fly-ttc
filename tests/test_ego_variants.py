"""Checks for controls, unchanged v0, and validation-only frozen calibration."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fly_ttc.experiments.ego_motion import calibrate, config_digest, digest, manifest_semantics, select_confirmation, verify_frozen
from fly_ttc.models.ego_variants import MODELS, EgoVariantScorer, model_components, model_config
from fly_ttc.models.v0_expansion import ExpansionScorer, V0Config, score_components
from fly_ttc.preprocess.ego_motion import EgoMotionConfig
from synthetic_stimuli import stimulus_clips


def test_shared_scorer_preserves_original_v0_exactly_and_resets():
    frames = stimulus_clips(n_frames=15)["loom"]
    original = ExpansionScorer()
    ablation = EgoVariantScorer(V0Config(), {}, EgoMotionConfig())
    for gray in frames:
        expected, actual = original.update(gray), ablation.update(gray)
        for key in expected:
            assert actual[key] == expected[key]
    ablation.reset()
    assert ablation.update(frames[0])["flow_valid"] is False


def test_fallback_and_no_blob_controls_have_identical_scores():
    ablation = EgoVariantScorer(V0Config(), {}, EgoMotionConfig())
    gray = np.ones((192, 192), np.float32) * .5
    frames = pd.DataFrame([ablation.update(gray) for _ in range(3)])
    assert not frames.fit_valid.any()
    raw = model_components(frames, "raw_flow_no_blob")
    residual = model_components(frames, "affine_residual_no_blob")
    assert np.array_equal(raw.S_div, residual.S_div)
    assert np.array_equal(raw.S_rad, residual.S_rad)
    frames["S_blob"] = 100000.
    transformed = model_components(frames, "raw_flow_no_blob")
    assert transformed.S_blob.eq(0).all()
    assert model_config(V0Config(), "raw_flow_no_blob") == model_config(V0Config(), "affine_residual_no_blob")


def test_difference_control_uses_only_frame_difference():
    frame = pd.DataFrame(dict(S_div=[100, 200], S_rad=[500, 600], S_blob=[1000, 2000], frame_diff=[.1, .2]))
    transformed = model_components(frame, "frame_difference")
    scores = score_components(transformed, None, model_config(V0Config(ema=1), "frame_difference"))
    assert np.allclose(scores, [.1, .2])


def test_calibration_ignores_exploratory_confirmation_and_positive_extremes():
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/default.yaml").read_text())
    data = pd.DataFrame(dict(S_div=[1.,2.,3.], S_rad=[1.,2.,3.], S_blob=[0.,0.,0.],
                             residual_div=[.1,.2,.3], residual_rad=[.2,.4,.6], frame_diff=[.01,.02,.03]))
    cache = {"neg": data, "pos": data * 1e8, "confirmation": data * -1e8}
    protocol = {"negative_calibration_ids": ["neg"], "validation_ids": ["neg", "pos"]}
    original = pd.DataFrame(dict(video_id=["neg", "pos"], label=[0,1]))
    original_threshold = {"theta": 6.79, "normalization": {"frozen": True}}
    thresholds = calibrate(cache, original, protocol, config, original_threshold)
    assert thresholds["v0_original"] is original_threshold
    assert thresholds["raw_flow_no_blob"]["normalization"]["S_div"]["mean"] == 2.
    assert thresholds["affine_residual_no_blob"]["normalization"]["S_div"]["mean"] == pytest.approx(.2)
    assert all(thresholds[m]["val_fpr"] <= .1 for m in MODELS if m != "v0_original")
    del cache["neg"]
    with pytest.raises(ValueError, match="no replacement"):
        calibrate(cache, original, protocol, config, original_threshold)


def test_confirmation_selection_disjoint_and_reproducible():
    source = pd.DataFrame(dict(video_id=[f"{i:05d}" for i in range(20)], label=[0]*10+[1]*10,
                               time_of_event=[np.nan]*10+[4.]*10, scene="Urban", light_conditions="Normal"))
    excluded = ["00000", "00010"]
    first = select_confirmation(source, excluded, 3, 3, 1)
    second = select_confirmation(source.sample(frac=1, random_state=20), excluded, 3, 3, 1)
    assert first.video_id.tolist() == second.video_id.tolist()
    assert not set(first.video_id) & set(excluded)
    assert first.label.value_counts().to_dict() == {0:3,1:3}


def test_frozen_protocol_rejects_score_code_or_config_changes(tmp_path):
    code = tmp_path / "score.py"
    code.write_text("frozen")
    config = {"followup": {"baseline_run": str(tmp_path)}}
    protocol = {"config_sha256": config_digest(config), "scoring_file_sha256": {"score.py": digest(code)},
                "baseline_file_sha256": {}}
    verify_frozen(config, tmp_path, protocol)
    code.write_text("retuned")
    with pytest.raises(ValueError, match="code changed"):
        verify_frozen(config, tmp_path, protocol)
    with pytest.raises(ValueError, match="Configuration changed"):
        verify_frozen({**config,"new":1}, tmp_path, protocol)


def test_confirmation_semantics_freezes_labels_times_and_conditions_only():
    frame = pd.DataFrame([dict(video_id="00001", label=1, time_of_alert=2., time_of_event=3.,
                               scene="Urban", weather="Clear", light_conditions="Normal", source_repo="source",
                               source_revision="revision", source_path="train/positive/00001.mp4", source_sha256="abc",
                               mtime_ns=0, size_bytes=0)])
    frozen = manifest_semantics(frame)
    frame["mtime_ns"], frame["size_bytes"] = 123, 1000
    assert manifest_semantics(frame) == frozen
    for column, changed in (("label", 0), ("time_of_event", 3.1), ("scene", "Highway"), ("source_sha256", "def")):
        altered = frame.copy()
        altered[column] = changed
        assert manifest_semantics(altered) != frozen
