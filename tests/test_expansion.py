"""Actual Farneback stimulus checks plus mathematical and calibration contracts."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from fly_ttc.models.v0_expansion import ExpansionScorer, V0Config, fit_normalization, score_components
from fly_ttc.preprocess.expansion import blob_growth, center_weights, divergence, radial_flow
from fly_ttc.preprocess.flow import clip_flow_magnitude, farneback_flow
from synthetic_stimuli import stimulus_clips


@pytest.fixture(scope="module")
def synthetic_scores() -> dict[str, pd.DataFrame]:
    result = {}
    for name, frames in stimulus_clips().items():
        scorer = ExpansionScorer()
        result[name] = pd.DataFrame([scorer.update(frame) for frame in frames])
    return result


def test_farneback_loom_exceeds_controls(synthetic_scores):
    """One pixel/pair grid translation is nontrivial, not an almost-static foil."""
    late = slice(-15, None)
    means = {name: trace["S"].iloc[late].mean() for name, trace in synthetic_scores.items()}
    assert means["loom"] > 2 * means["recede"]
    assert means["loom"] > 1.2 * means["translate"]
    assert means["loom"] > 1.2 * means["edge"]
    assert synthetic_scores["translate"]["flow_horizontal"].iloc[5:].median() > 0.7
    assert synthetic_scores["flicker"]["S"].abs().max() < 1e-6


def test_farneback_loom_rises_over_time(synthetic_scores):
    scores = synthetic_scores["loom"]["S"].to_numpy()[1:]
    # Subpixel contours cause local ripples: each consecutive 10-frame block
    # must rise, with strong framewise rank correlation and a large late score.
    block_means = np.array([part.mean() for part in np.array_split(scores, 6)])
    assert np.all(np.diff(block_means) > 0)
    assert pd.Series(scores).corr(pd.Series(np.arange(len(scores))), method="spearman") > 0.95
    assert scores[-10:].mean() > 4 * scores[:10].mean()


@pytest.mark.parametrize("kernel", [1, 3, 5, 7])
def test_sobel_sign_and_scale(kernel):
    y, x = np.mgrid[:41, :45].astype(np.float32)
    flow = np.stack((0.2 * (x - 22), 0.3 * (y - 20)), axis=-1)
    np.testing.assert_allclose(divergence(flow, kernel)[4:-4, 4:-4], 0.5, atol=1e-6)
    np.testing.assert_allclose(divergence(-flow, kernel)[4:-4, 4:-4], -0.5, atol=1e-6)
    assert np.min(radial_flow(flow)) >= 0
    assert np.max(radial_flow(-flow)) <= 0


def test_flow_conversion_direction_and_percentile():
    rng = np.random.default_rng(0)
    previous = rng.uniform(size=(96, 96)).astype(np.float32)
    current = cv2.warpAffine(previous, np.float32([[1, 0, 2], [0, 1, 0]]), (96, 96), borderMode=cv2.BORDER_REFLECT)
    flow = farneback_flow(previous, current)
    assert np.median(flow[12:-12, 12:-12, 0]) == pytest.approx(2, abs=0.1)
    assert np.median(flow[12:-12, 12:-12, 1]) == pytest.approx(0, abs=0.1)
    flow[0, 0] = [300, 400]
    cap = np.percentile(np.linalg.norm(flow, axis=-1), 99)
    clipped = clip_flow_magnitude(flow)
    assert np.linalg.norm(clipped, axis=-1).max() <= cap + 1e-5
    assert clipped[0, 0, 0] / clipped[0, 0, 1] == pytest.approx(0.75)


def test_weights_and_blob_absence():
    weights = center_weights((101, 151))
    assert weights[50, 75] == pytest.approx(1.0)
    np.testing.assert_allclose(weights[[0, -1], :], 0.3, atol=1e-7)
    np.testing.assert_allclose(weights[:, [0, -1]], 0.3, atol=1e-7)
    assert blob_growth(10, 0) == 0
    assert blob_growth(0, 10) == 0
    assert blob_growth(12, 10) == pytest.approx(0.2)
    assert blob_growth(8, 10) == 0


def test_negative_validation_normalization_and_online_offline_parity():
    frames = stimulus_clips(size=128, n_frames=15)["loom"]
    extractor = ExpansionScorer()
    components = pd.DataFrame([extractor.update(frame) for frame in frames])
    negatives = pd.DataFrame({"S_div": [0, 1, 3], "S_rad": [0, 2, 4], "S_blob": [0, 0, 0], "flow_valid": [False, True, True], "label": [0, 0, 0], "split": ["validation"] * 3})
    stats = fit_normalization([negatives])
    assert stats["S_div"] == {"mean": 2.0, "std": 1.0}
    assert stats["S_blob"]["std"] == V0Config().normalization_std_floor
    assert stats["n_frames"] == 2
    offline = score_components(components, stats)
    scorer = ExpansionScorer(normalization=stats)
    online = [scorer.update(frame)["S"] for frame in frames]
    np.testing.assert_allclose(online, offline)
    scorer.reset()
    np.testing.assert_allclose([scorer.update(frame)["S"] for frame in frames], offline)
    with pytest.raises(ValueError, match="negative validation"):
        fit_normalization([negatives.assign(label=1)])
    with pytest.raises(ValueError, match="validation"):
        fit_normalization([negatives.assign(split="test")])
    with pytest.raises(ValueError, match="No valid flow"):
        fit_normalization([negatives.iloc[:1]])


def test_scorer_rejects_invalid_inputs():
    scorer = ExpansionScorer()
    with pytest.raises(TypeError, match="floating point"):
        scorer.update(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match="finite intensities"):
        scorer.update(np.full((10, 10), np.nan, dtype=np.float32))
    scorer.update(np.zeros((10, 10), dtype=np.float32))
    with pytest.raises(ValueError, match="same shape"):
        scorer.update(np.zeros((12, 12), dtype=np.float32))
