"""Analytic and actual-texture Farneback checks, independent of real labels."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from fly_ttc.preprocess.ego_motion import EgoMotionConfig, estimate_background_flow
from fly_ttc.preprocess.expansion import radial_flow
from fly_ttc.preprocess.flow import farneback_flow


def _affine_flow(size=192, zoom=0.012, rotation=0.018, dx=2.0, dy=-1.0):
    y, x = np.mgrid[:size, :size].astype(np.float32) - (size - 1) / 2
    return np.stack((dx + zoom * x - rotation * y, dy + rotation * x + zoom * y), axis=-1)


def _texture(size=192):
    rng = np.random.default_rng(730)
    texture = rng.uniform(size=(size, size)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (3, 3), 0.6)
    return ((texture - texture.min()) / np.ptp(texture)).astype(np.float32)


def _texture_pair(kind, size=192):
    previous = _texture(size)
    center = (size - 1) / 2
    if kind == "translation":
        matrix = np.float32([[1, 0, 2], [0, 1, -1]])
    else:
        matrix = cv2.getRotationMatrix2D((center, center), 0.8, 1.012)
        matrix[:, 2] += [1.5, -0.7]
    current = cv2.warpAffine(previous, matrix, (size, size), borderMode=cv2.BORDER_REFLECT)
    if kind in {"loom", "recede"}:
        # A central textured disk occupies < the excluded rectangle. It shares
        # the camera translation; only its size changes relative to background.
        y, x = np.mgrid[:size, :size].astype(np.float32)
        radii = (22.0, 26.0) if kind == "loom" else (26.0, 22.0)
        for frame, radius, shift in ((previous, radii[0], (0, 0)), (current, radii[1], (1.5, -0.7))):
            nx, ny = (x - center - shift[0]) / radius, (y - center - shift[1]) / radius
            mask = np.clip((1 - np.hypot(nx, ny)) * radius, 0, 1)
            disk = 0.25 + 0.22 * np.sin(15 * nx) * np.cos(13 * ny)
            frame[:] = frame * (1 - mask) + disk * mask
    return previous, current


@pytest.mark.parametrize("method", ["affine", "translation"])
def test_translation_is_removed_without_mutating_flow(method):
    flow = _affine_flow(zoom=0, rotation=0)
    original = flow.copy()
    result = estimate_background_flow(flow, EgoMotionConfig(method=method))
    assert result["fit_valid"]
    assert result["inlier_fraction"] == 1
    np.testing.assert_allclose(result["residual_flow"], 0, atol=1e-6)
    np.testing.assert_array_equal(flow, original)


def test_affine_removes_rotation_zoom_and_translation_deterministically():
    flow = _affine_flow()
    first, second = estimate_background_flow(flow), estimate_background_flow(flow)
    assert first["fit_valid"]
    assert first["fit_error_px"] < 1e-6
    np.testing.assert_allclose(first["residual_flow"], 0, atol=1e-6)
    np.testing.assert_array_equal(first["background_flow"], second["background_flow"])


def test_central_looming_survives_camera_flow_and_exceeds_recession():
    flow = _affine_flow()
    y, x = np.mgrid[:192, :192].astype(np.float32) - 95.5
    mask = np.hypot(x, y) < 30
    local = np.stack((0.1 * x, 0.1 * y), axis=-1) * mask[..., None]
    loom = estimate_background_flow(flow + local)
    recede = estimate_background_flow(flow - local)
    assert loom["fit_valid"] and recede["fit_valid"]
    np.testing.assert_allclose(loom["residual_flow"], local, atol=1e-6)
    assert np.mean(np.maximum(radial_flow(loom["residual_flow"])[mask], 0)) > 1.5
    assert np.max(np.maximum(radial_flow(recede["residual_flow"])[mask], 0)) < 1e-6


def test_broad_true_looming_is_removed_exposing_zoom_ambiguity():
    # This is a limitation regression, not a claim that removal is desirable.
    flow = _affine_flow(zoom=0.04, rotation=0, dx=0, dy=0)
    result = estimate_background_flow(flow)
    assert result["fit_valid"]
    assert np.mean(np.maximum(radial_flow(flow), 0)) > 2
    assert np.max(np.abs(result["residual_flow"])) < 1e-5


def test_huber_fit_tolerates_sparse_large_outliers():
    flow = _affine_flow()
    contaminated = flow.copy()
    rng = np.random.default_rng(13)
    mask = rng.random(flow.shape[:2]) < 0.15
    contaminated[mask] += rng.uniform(-30, 30, size=(int(mask.sum()), 2))
    result = estimate_background_flow(contaminated)
    assert result["fit_valid"]
    assert result["inlier_fraction"] > 0.75
    assert np.median(np.linalg.norm(result["background_flow"] - flow, axis=-1)) < 0.12


def test_incoherent_field_abstains_and_preserves_raw_flow():
    flow = np.random.default_rng(6).uniform(-12, 12, (192, 192, 2)).astype(np.float32)
    result = estimate_background_flow(flow)
    assert not result["fit_valid"]
    assert result["fit_reason"] == "insufficient_inliers"
    np.testing.assert_array_equal(result["residual_flow"], flow)
    np.testing.assert_array_equal(result["background_flow"], 0)


def test_spatially_narrow_inliers_are_rejected():
    flow = np.full((192, 192, 2), -1, dtype=np.float32)
    flow[:96, :, 0] = -30
    flow[96:, :, 0] = 30
    flow[82:110, :, :] = [2, -1]
    # The componentwise median finds [2,-1], but its agreeing samples form
    # only a thin horizontal strip: a plausible fit must still abstain.
    config = replace(EgoMotionConfig(), method="translation", min_inlier_fraction=0.05, min_samples=6)
    result = estimate_background_flow(flow, config)
    assert not result["fit_valid"]
    assert result["fit_reason"] == "insufficient_inlier_spatial_support"
    np.testing.assert_array_equal(result["residual_flow"], flow)


def test_depth_dependent_forward_flow_is_not_fully_removed():
    # Pinhole forward-translation flow is proportional to r/Z. A depth map
    # varying over the image produces parallax beyond one global affine field.
    y, x = np.mgrid[:192, :192].astype(np.float32) - 95.5
    inverse_depth = 0.015 + 0.012 * np.sin(x / 30) * np.sin(y / 30)
    flow = np.stack((inverse_depth * x, inverse_depth * y), axis=-1)
    result = estimate_background_flow(flow)
    assert np.mean(np.linalg.norm(result["residual_flow"], axis=-1)) > 0.2


def test_tiny_and_flat_fields_abstain_without_inventing_motion():
    for flow, reason in ((np.zeros((8, 8, 2), np.float32), "insufficient_samples"), (np.zeros((192, 192, 2), np.float32), "low_motion")):
        result = estimate_background_flow(flow)
        assert not result["fit_valid"]
        assert result["fit_reason"] == reason
        np.testing.assert_array_equal(result["residual_flow"], 0)
    previous = np.full((192, 192), 0.2, np.float32)
    current = np.full_like(previous, 0.8)
    result = estimate_background_flow(farneback_flow(previous, current))
    assert not result["fit_valid"]
    assert np.max(np.abs(result["residual_flow"])) < 1e-6


@pytest.mark.parametrize("kind", ["translation", "affine"])
def test_actual_farneback_textures_suppress_global_motion(kind):
    previous, current = _texture_pair(kind)
    flow = farneback_flow(previous, current)
    result = estimate_background_flow(flow)
    assert result["fit_valid"]
    interior = np.s_[15:-15, 15:-15]
    raw_motion = np.median(np.linalg.norm(flow[interior], axis=-1))
    residual_motion = np.median(np.linalg.norm(result["residual_flow"][interior], axis=-1))
    assert raw_motion > 1
    assert residual_motion < 0.15 * raw_motion


def test_actual_farneback_local_looming_survives_global_motion():
    values = {}
    for kind in ("affine", "loom", "recede"):
        previous, current = _texture_pair(kind)
        result = estimate_background_flow(farneback_flow(previous, current))
        assert result["fit_valid"]
        radial = radial_flow(result["residual_flow"])
        values[kind] = np.mean(np.maximum(radial[64:128, 64:128], 0))
    assert values["loom"] > 5 * values["affine"]
    assert values["loom"] > 3 * values["recede"]


def test_input_and_configuration_validation():
    with pytest.raises(ValueError, match="finite"):
        estimate_background_flow(np.full((12, 12, 2), np.nan))
    with pytest.raises(ValueError, match="shape"):
        estimate_background_flow(np.zeros((12, 12)))
    with pytest.raises(TypeError, match="floating"):
        estimate_background_flow(np.zeros((12, 12, 2), dtype=np.int32))
    for settings in ({"method": "homography"}, {"grid_step_px": 0}, {"min_quadrants": 5}, {"center_exclusion_fraction": 1}, {"huber_delta_px": float("nan")}):
        with pytest.raises(ValueError):
            EgoMotionConfig(**settings)
