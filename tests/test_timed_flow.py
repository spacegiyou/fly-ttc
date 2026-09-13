"""Physical-time contracts, not claims of flow estimator invariance."""

import numpy as np
import pytest

from fly_ttc.models.timed_flow import TimedFlowScorer, continuous_ema_alpha, reference_displacement


def test_same_velocity_has_same_reference_displacement():
    velocity = np.ones((9, 11, 2), dtype=np.float32) * [3, -7]
    for dt in [1/60, 1/30, 1/15, 0.1]:
        np.testing.assert_allclose(reference_displacement(velocity * dt, dt, 1/15), velocity / 15, atol=1e-7)


def test_exponential_smoothing_composes_across_irregular_intervals():
    def integrate(intervals):
        score = 0.0
        for dt in intervals:
            score += continuous_ema_alpha(dt, 0.2) * (7.0 - score)
        return score
    assert integrate([0.1, 0.2, 0.4, 0.3]) == pytest.approx(integrate([0.01]*100), abs=1e-12)
    assert continuous_ema_alpha(1/15, -(1/15)/np.log(0.7)) == pytest.approx(0.3)


@pytest.mark.parametrize("dt", [0, -1, np.inf, np.nan])
def test_invalid_time_is_rejected(dt):
    with pytest.raises(ValueError):
        reference_displacement(np.zeros((4,4,2)), dt, 1/15)


def test_invalid_timestamp_does_not_change_state_and_reset_restarts():
    scorer = TimedFlowScorer()
    gray = np.ones((32,32), dtype=np.float32) * 0.5
    assert not scorer.update(gray, 2.0)["valid"]
    with pytest.raises(ValueError):
        scorer.update(gray, 2.0)
    assert scorer.update(gray, 2.1)["dt_s"] == pytest.approx(0.1)
    scorer.reset()
    assert not scorer.update(gray, 0)["valid"]
