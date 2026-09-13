"""Independent sign, source recurrence, temporal contract, and local RF checks."""
from dataclasses import replace

import numpy as np
import pytest

from fly_ttc.models.local_reflex_proxy_not_zhao_star import (
    LocalReflexConfig, LocalReflexScorer, TimestampEMD,
    lowpass_alpha, pool_directional_motion,
)


def _scalar_author_step(previous, current, hp_old, delayed_on_old, delayed_off_old):
    """Independent scalar translation of upstream emd.m at loom.m's 10 ms step.

    Origin: upstream commit 50c7c8f..., task 2/emd.m. No MATLAB runtime
    execution is implied by this arithmetic parity reference.
    """
    rows, cols = current.shape
    hp = np.zeros_like(current)
    on, off, don, doff = [np.zeros_like(current) for _ in range(4)]
    alpha = 1 / (0.050 / 0.010 + 1)
    beta = (0.250 / 0.010) / (0.250 / 0.010 + 1)
    for i in range(rows):
        for j in range(cols):
            hp[i, j] = beta * (current[i, j] - previous[i, j]) + beta * hp_old[i, j]
            on[i, j] = hp[i, j] if hp[i, j] > 0 else 0
            off[i, j] = abs(hp[i, j] - 0.05) if hp[i, j] < 0.05 else 0
            don[i, j] = alpha * on[i, j] + (1 - alpha) * delayed_on_old[i, j]
            doff[i, j] = alpha * off[i, j] + (1 - alpha) * delayed_off_old[i, j]
    motion = np.zeros((rows - 1, cols - 1, 4))
    for i in range(rows - 1):
        for j in range(cols - 1):
            for fast, delayed in ((on, don), (off, doff)):
                motion[i, j, 0] += delayed[i, j] * fast[i, j + 1]
                motion[i, j, 1] += fast[i, j] * delayed[i, j + 1]
                motion[i, j, 2] += delayed[i, j] * fast[i + 1, j]
                motion[i, j, 3] += fast[i, j] * delayed[i + 1, j]
    return hp, don, doff, motion


def test_emd_matches_author_scalar_recurrence_at_10_ms():
    rng = np.random.default_rng(240913)
    sequence = rng.random((9, 5, 6))
    emd = TimestampEMD()
    first = emd.update(sequence[0], 0.0)
    assert not first["directional_motion"].any()
    hp, don, doff = [np.zeros_like(sequence[0]) for _ in range(3)]
    for i in range(1, len(sequence)):
        hp, don, doff, expected = _scalar_author_step(sequence[i - 1], sequence[i], hp, don, doff)
        got = emd.update(sequence[i], i * 0.010)
        np.testing.assert_allclose(got["directional_motion"], expected, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(emd.highpass, hp, atol=1e-14)
        np.testing.assert_allclose(emd.delayed_on, don, atol=1e-14)
        np.testing.assert_allclose(emd.delayed_off, doff, atol=1e-14)


@pytest.mark.parametrize("axis,sign,preferred,opposite", [(1, 1, 0, 1), (1, -1, 1, 0), (0, 1, 2, 3), (0, -1, 3, 2)])
def test_actual_moving_grating_has_correct_direction_sign(axis, sign, preferred, opposite):
    coordinate = np.indices((48, 48))[axis]
    emd = TimestampEMD()
    observed = []
    for t in np.arange(0, 1.5, 1 / 60):
        gray = .5 + .45 * np.sin((coordinate - sign * 12 * t) * np.pi / 8)
        result = emd.update(gray, t)
        if t > .5:
            observed.append(result["directional_motion"].mean(axis=(0, 1)))
    channels = np.mean(observed, axis=0)
    assert channels[preferred] > channels[opposite] + .005


def test_physical_time_constants_change_with_actual_irregular_dt():
    emd = TimestampEMD()
    zeros, ones = np.zeros((4, 4)), np.ones((4, 4))
    emd.update(zeros, 8.0)
    emd.update(ones, 8.1)
    np.testing.assert_allclose(emd.highpass, .25 / (.25 + .1), atol=1e-14)
    np.testing.assert_allclose(emd.delayed_on, (.1 / (.05 + .1)) * (.25 / (.25 + .1)), atol=1e-14)
    old_hp = emd.highpass.copy()
    result = emd.update(ones, 8.13)
    assert result["dt_s"] == pytest.approx(.03)
    np.testing.assert_allclose(emd.highpass, old_hp * .25 / (.25 + .03), atol=1e-14)
    assert lowpass_alpha(.01, .05) == pytest.approx(1 / 6)


def test_no_wrap_between_opposite_image_borders():
    control, changed = TimestampEMD(), TimestampEMD()
    blank = np.zeros((12, 16))
    changed_first = blank.copy()
    changed_first[:, -1] = 1.0
    control.update(blank, 0)
    changed.update(changed_first, 0)
    second = blank.copy()
    second[:, 0] = 1.0
    a = control.update(second, .02)["directional_motion"]
    b = changed.update(second, .02)["directional_motion"]
    np.testing.assert_array_equal(a[:, :2], b[:, :2])
    assert a.shape == (11, 15, 4)


def test_reject_invalid_timestamp_or_shape_without_corrupting_state():
    emd = TimestampEMD()
    gray = np.full((8, 8), .5)
    emd.update(gray, 3.0)
    for timestamp in (3.0, 2.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            emd.update(gray, timestamp)
    with pytest.raises(ValueError, match="shape"):
        emd.update(np.zeros((9, 8)), 3.1)
    assert emd.previous_t_s == 3.0
    np.testing.assert_array_equal(emd.previous_gray, gray)
    emd.reset()
    assert emd.update(gray, 0)["dt_s"] == 0


@pytest.mark.parametrize("gray", [np.full((8, 8), np.nan), np.full((8, 8), 255), np.zeros((8, 8, 3))])
def test_require_finite_normalized_gray(gray):
    with pytest.raises(ValueError):
        TimestampEMD().update(gray, 0)


def _outward_motion(side=64):
    y, x = np.indices((side - 1, side - 1)) + .5
    center = (side - 1) / 2
    return np.stack([x >= center, x < center, y >= center, y < center], axis=-1).astype(float)


def test_pure_pooling_local_direction_assignment_and_matched_energy():
    motion = _outward_motion()
    cfg = LocalReflexConfig(grid=1)
    identity = pool_directional_motion(motion, cfg, input_shape=(64, 64))
    inverted = pool_directional_motion(motion, replace(cfg, direction_permutation=(1, 0, 3, 2)))
    np.testing.assert_allclose(identity["arm_opponent"], 1)
    np.testing.assert_allclose(identity["activation_map"], 1)
    np.testing.assert_allclose(inverted["arm_opponent"], -1)
    np.testing.assert_allclose(inverted["activation_map"], 0)
    np.testing.assert_array_equal(identity["simple_energy_map"], inverted["simple_energy_map"])
    np.testing.assert_allclose(identity["simple_energy_map"], 2)
    np.testing.assert_array_equal(identity["unit_centers_xy"], [[[31.5, 31.5]]])


def test_threshold_subtracts_per_star_equation_not_author_hard_gate():
    motion = _outward_motion()
    cfg = LocalReflexConfig(grid=1, arm_threshold=.25)
    got = pool_directional_motion(motion, cfg)
    np.testing.assert_allclose(got["activation_map"], .75)


def test_translation_cannot_activate_four_direction_product():
    field = np.zeros((63, 63, 4))
    field[..., 0] = 2
    cfg = LocalReflexConfig()
    response = pool_directional_motion(field, cfg)
    np.testing.assert_allclose(response["activation_map"], 0)
    np.testing.assert_allclose(response["simple_energy_map"], 2)
    mean = pool_directional_motion(field, replace(cfg, aggregation="mean"))
    np.testing.assert_allclose(mean["activation_map"], .5)


def test_geometry_has_complete_support_and_never_silently_resizes():
    with pytest.raises(ValueError, match="RF support"):
        LocalReflexScorer().update(np.zeros((32, 32)), 0)
    with pytest.raises(ValueError, match="input_shape"):
        pool_directional_motion(np.zeros((63, 63, 4)), input_shape=(63, 63))
    result = pool_directional_motion(np.zeros((63, 63, 4)))
    centers = result["unit_centers_xy"]
    assert centers.shape == (5, 5, 2)
    assert centers[..., 0].min() >= 16
    assert centers[..., 0].max() <= 47


def test_localized_loom_and_recede_with_explicit_time():
    y, x = np.indices((96, 96))
    center = (63.25, 47.5)
    maxima = {}
    for kind in ("loom", "recede"):
        scorer = LocalReflexScorer()
        best = None
        for t in np.arange(0, 1.2, 1 / 60):
            radius = 4 + 10 * t if kind == "loom" else 16 - 10 * t
            gray = np.clip(np.hypot(x - center[0], y - center[1]) - radius + .5, 0, 1)
            result = scorer.update(gray, t)
            if best is None or result["S"] > best["S"]:
                best = result
        maxima[kind] = best["S"]
        if kind == "loom":
            peak_row, peak_col = np.unravel_index(best["activation_map"].argmax(), (5, 5))
            assert peak_col > 2
            assert peak_row == 2
    assert maxima["loom"] > .005
    assert maxima["recede"] < maxima["loom"] * .01


def test_uniform_flicker_has_no_opponent_response():
    scorer = LocalReflexScorer()
    for t in np.arange(0, 1.0, 1 / 30):
        out = scorer.update(np.full((64, 64), .5 + .4 * np.sin(15 * t)), t)
        assert out["S"] == 0
        assert np.all(out["arm_opponent"] == 0)


def test_returned_maps_and_inputs_do_not_alias_scorer_state():
    scorer = LocalReflexScorer()
    frame = np.zeros((64, 64))
    out = scorer.update(frame, 0)
    frame[:] = 1
    out["activation_map"][:] = 200
    out["unit_centers_xy"][:] = 200
    next_out = scorer.update(np.zeros((64, 64)), .1)
    assert next_out["S"] == 0
    assert next_out["unit_centers_xy"].max() < 64


@pytest.mark.parametrize("kwargs", [
    {"direction_permutation": (0, 0, 2, 3)}, {"highpass_tau_s": 0},
    {"lowpass_tau_s": np.nan}, {"output_tau_s": -1}, {"rf_size_px": 0},
    {"grid": 2.5}, {"sample_spacing_px": True}, {"arm_width_fraction": 0},
    {"arm_threshold": -1}, {"aggregation": "unknown"}, {"pathways": "unknown"},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        LocalReflexConfig(**kwargs)
