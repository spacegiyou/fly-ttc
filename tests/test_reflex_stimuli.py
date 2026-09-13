"""Geometry and time-contract tests for analytical visual stimuli."""

from dataclasses import replace

import numpy as np
import pytest

from fly_ttc.synthetic.reflex_stimuli import KINDS, StimulusSpec, render_frame, timestamps, truth


def test_physical_disk_obeys_pinhole_radius_and_contact_clock():
    spec = StimulusSpec("loom")
    first, middle, last = [truth(spec, t) for t in (0.0, 1.0, 2.0)]
    assert [row["radius_px"] for row in (first, middle, last)] == [8.0, 12.0, 24.0]
    assert [row["true_ttc_s"] for row in (first, middle, last)] == [3.0, 2.0, 1.0]
    assert middle["radial_speed_px_s"] == 6.0
    assert middle["angular_diameter_rad"] == pytest.approx(2 * np.arctan(1 / 8))
    # At the optical axis, exact angular span and on-axis equivalent agree.
    assert middle["horizontal_angular_span_rad"] == middle["angular_diameter_rad"]
    assert middle["vertical_angular_span_rad"] == middle["angular_diameter_rad"]


def test_antialiased_area_measures_projected_disk_radius():
    spec = StimulusSpec("loom", contrast=1.0, supersample=4)
    for t in (0.0, 1.0, 2.0):
        coverage = (render_frame(spec, t) - 0.5) / 0.5
        measured_radius = np.sqrt(coverage.sum() / np.pi)
        assert measured_radius == pytest.approx(truth(spec, t)["radius_px"], abs=0.04)


def test_recede_is_time_reverse_of_same_physical_approach():
    loom = StimulusSpec("loom", center_fraction=(0.31, 0.62), polarity=-1)
    recede = replace(loom, kind="recede")
    for t in (0.0, 0.25, 0.75, 1.0, 2.0):
        np.testing.assert_array_equal(render_frame(recede, t), render_frame(loom, loom.duration_s - t))
        reverse = truth(recede, t)
        forward = truth(loom, loom.duration_s - t)
        assert reverse["radius_px"] == forward["radius_px"]
        assert reverse["radial_speed_px_s"] == -forward["radial_speed_px_s"]
        assert reverse["true_ttc_s"] is None
        assert reverse["depth_velocity_per_s"] > 0


def test_linear_enlargement_is_distinct_from_approach_with_reference_match():
    loom = StimulusSpec("loom")
    linear = replace(loom, kind="linear_expand")
    at_ref = truth(linear, 1.0)
    assert at_ref["radius_px"] == truth(loom, 1.0)["radius_px"]
    assert at_ref["radial_speed_px_s"] == truth(loom, 1.0)["radial_speed_px_s"]
    assert at_ref["true_ttc_s"] is None
    assert at_ref["motion_class"] == "arbitrary_linear_enlargement"
    assert truth(linear, 0.0)["radius_px"] != truth(loom, 0.0)["radius_px"]
    assert truth(linear, 2.0)["radius_px"] != truth(loom, 2.0)["radius_px"]


def test_translation_has_constant_radius_and_matched_pixels_per_second():
    spec = StimulusSpec("translate", motion_scale=2.0, direction_rad=np.pi / 2)
    first, middle, last = [truth(spec, t) for t in (0.0, 1.0, 2.0)]
    assert first["radius_px"] == middle["radius_px"] == last["radius_px"] == 12.0
    assert first["radial_speed_px_s"] == 0.0
    np.testing.assert_allclose(first["target_velocity_px_s"], [0, 12], atol=1e-12)
    np.testing.assert_allclose(np.subtract(last["center_xy_px"], first["center_xy_px"]), [0, 24], atol=1e-12)
    assert middle["true_ttc_s"] is None


def test_translating_target_exits_image_without_wrapping():
    spec = StimulusSpec("translate", motion_scale=30.0)
    np.testing.assert_array_equal(render_frame(spec, 0.0), np.full(spec.image_size, 0.5, dtype=np.float32))
    assert np.max(render_frame(spec, 1.0)) > 0.8
    np.testing.assert_array_equal(render_frame(spec, 2.0), np.full(spec.image_size, 0.5, dtype=np.float32))
    assert truth(spec, 0.0)["disk_fully_visible"] is False
    assert truth(spec, 1.0)["disk_fully_visible"] is True


def test_background_speed_and_rotation_match_at_declared_reference_radius():
    spec = StimulusSpec("background_translation", motion_scale=4.0)
    translation = truth(spec, 1.0)
    rotation = truth(replace(spec, kind="rotation"), 1.0)
    assert translation["background_velocity_px_s"] == [24.0, 0.0]
    assert rotation["rotation_rad_s"] * rotation["reference_radius_px"] == 24.0
    assert rotation["radius_px"] is None
    # Increasing a nuisance multiplier leaves the physical reference unchanged.
    assert rotation["reference_boundary_speed_px_s"] == 6.0


def test_analytical_grating_translation_matches_integer_pixel_shift():
    spec = StimulusSpec("background_translation", motion_scale=1.0, texture_kind="grating")
    first = render_frame(spec, 0.0)
    # 6 px/s * 0.5 s = 3 px, compared in the shared interior only.
    shifted = render_frame(spec, 0.5)
    np.testing.assert_allclose(shifted[:, 3:], first[:, :-3], atol=1e-7)


def test_off_axis_location_has_explicit_angular_geometry():
    center = StimulusSpec("loom")
    offcenter = replace(center, center_fraction=(0.75, 0.25))
    center_truth, off_truth = truth(center, 1.0), truth(offcenter, 1.0)
    assert off_truth["center_xy_px"] == [95.25, 31.75]
    assert off_truth["radius_px"] == center_truth["radius_px"]
    assert off_truth["angular_diameter_rad"] == center_truth["angular_diameter_rad"]
    assert off_truth["horizontal_angular_span_rad"] < off_truth["angular_diameter_rad"]
    frame = render_frame(offcenter, 1.0)
    yy, xx = np.indices(frame.shape)
    mass = frame - 0.5
    assert (mass * xx).sum() / mass.sum() == pytest.approx(95.25, abs=0.03)
    assert (mass * yy).sum() / mass.sum() == pytest.approx(31.75, abs=0.03)


def test_static_texture_and_luminance_flicker_do_not_move_pattern():
    spec = StimulusSpec("static", texture_contrast=0.4)
    np.testing.assert_array_equal(render_frame(spec, 0.0), render_frame(spec, 1.123))
    flicker = replace(spec, kind="flicker", flicker_amplitude=0.1)
    delta = render_frame(flicker, 0.25) - render_frame(flicker, 0.0)
    np.testing.assert_allclose(delta, 0.1, atol=1e-7)
    assert truth(flicker, 0.25)["target_velocity_px_s"] == [0.0, 0.0]
    # Deliberately saturating stimulus remains bounded, with clipping explicit.
    saturated = render_frame(replace(flicker, flicker_amplitude=2.0), 0.25)
    np.testing.assert_array_equal(saturated, np.ones(spec.image_size, dtype=np.float32))


def test_polarity_reversal_and_zero_contrast_have_expected_luminance():
    spec = StimulusSpec("loom", contrast=0.4)
    positive = render_frame(spec, 0.5)
    negative = render_frame(replace(spec, polarity=-1), 0.5)
    np.testing.assert_allclose(positive + negative, 1.0, atol=1e-7)
    np.testing.assert_array_equal(render_frame(replace(spec, contrast=0.0), 0.5), np.full(spec.image_size, 0.5, dtype=np.float32))


def test_loom_over_moving_texture_preserves_geometric_truth():
    clean = StimulusSpec("loom")
    textured = replace(clean, kind="loom_on_background", background_speed_px_s=12.0)
    for t in (0.0, 1.0, 2.0):
        assert truth(textured, t)["radius_px"] == truth(clean, t)["radius_px"]
        assert truth(textured, t)["true_ttc_s"] == truth(clean, t)["true_ttc_s"]
    frame = render_frame(textured, 1.0)
    assert frame[63, 63] == pytest.approx(0.9)
    assert np.std(frame[:10]) > 0.01
    assert truth(textured, 1.0)["background_velocity_px_s"] == [12.0, 0.0]


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_all_renderers_are_finite_bounded_and_deterministic(kind):
    spec = StimulusSpec(kind, image_size=(64, 80))
    frame = render_frame(spec, 0.413)
    assert frame.shape == (64, 80)
    assert frame.dtype == np.float32
    assert np.isfinite(frame).all()
    assert frame.min() >= 0.0 and frame.max() <= 1.0
    np.testing.assert_array_equal(frame, render_frame(spec, 0.413))


@pytest.mark.parametrize("schedule,n", [("15hz", 31), ("30hz", 61), ("60hz", 121), ("irregular", 31)])
def test_timestamp_schedules_preserve_duration_and_strict_order(schedule, n):
    t = timestamps(2.0, schedule)
    assert len(t) == n
    assert t[0] == 0.0 and t[-1] == 2.0
    assert np.all(np.diff(t) > 0)
    np.testing.assert_array_equal(t, timestamps(2.0, schedule))


def test_irregular_intervals_are_declared_not_frame_rounding_artifacts():
    t = timestamps(2.0, "irregular")
    np.testing.assert_allclose(np.diff(t), np.tile([1 / 30, 3 / 30], 15), rtol=0, atol=3e-16)
    np.testing.assert_array_equal(t, timestamps(2.0, [1 / 30, 3 / 30]))
    partial = timestamps(0.15, "irregular")
    np.testing.assert_allclose(partial, [0.0, 1 / 30, 4 / 30, 0.15])
    assert partial[-1] == 0.15
    tiny = timestamps(1e-11, [1e-12])
    assert len(tiny) == 11 and np.all(np.diff(tiny) > 0)


def test_continuous_time_frame_does_not_depend_on_sampling_schedule():
    spec = StimulusSpec("loom")
    # These schedules all contain exactly 1.0 s, irrespective of frame index.
    frames = []
    for schedule in ("15hz", "30hz", "60hz"):
        times = timestamps(spec.duration_s, schedule)
        frames.append(render_frame(spec, times[np.where(times == 1.0)[0][0]]))
    np.testing.assert_array_equal(frames[0], frames[1])
    np.testing.assert_array_equal(frames[1], frames[2])


@pytest.mark.parametrize("changes", [
    {"kind": "unknown"}, {"duration_s": 3.0}, {"approach_speed": 0.0},
    {"center_fraction": (1.1, 0.5)}, {"contrast": 1.1}, {"polarity": 0},
    {"reference_t_s": -1.0}, {"texture_kind": "noise"}, {"supersample": 0},
])
def test_invalid_stimulus_parameters_fail_before_rendering(changes):
    with pytest.raises(ValueError):
        StimulusSpec(**({"kind": "loom"} | changes))


def test_invalid_times_and_schedules_are_rejected():
    spec = StimulusSpec("loom")
    for t in (-0.01, 2.01, np.nan):
        with pytest.raises(ValueError):
            render_frame(spec, t)
    for schedule in ("random", [], [0], [-0.1], [np.nan], 0, np.inf):
        with pytest.raises(ValueError):
            timestamps(2.0, schedule)
    with pytest.raises(ValueError, match="allocation limit"):
        timestamps(2.0, 1e9)
