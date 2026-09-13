"""Continuous-time image stimuli, independent of optical-flow estimation.

Coordinates use pixels (x right, y down); image_size is (height, width).
The approaching disk is a frontoparallel planar disk of physical radius R
with depth z(t) = z0 - v*t and projected radius f*R/z(t). Off-axis disks
retain a constant image bearing: their physical lateral location changes
with depth. This is an ideal pinhole construction, not a Nexar video.

All nuisance speeds are declared relative to the *physical loom's boundary
speed* at reference_t_s. Translation matches this speed, while rotation
matches it only on the circle of that reference radius. Equal speed is not
equal moving area, edge count, or motion energy. Arbitrary linear enlargement
is explicitly labelled and has no physical TTC claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from numbers import Real
from typing import Sequence

import numpy as np


KINDS = frozenset({
    "loom", "recede", "linear_expand", "translate", "background_translation",
    "rotation", "flicker", "static", "loom_on_background",
})
_DISK_KINDS = frozenset({"loom", "recede", "linear_expand", "translate", "loom_on_background"})


@dataclass(frozen=True)
class StimulusSpec:
    """One analytical stimulus, rendered at caller-supplied timestamps.

    Distances R, z0, and v use any one consistent physical length unit.
    contrast is the nominal foreground contrast relative to mid-gray:
    a disk has luminance 0.5 + polarity*contrast/2 on a 0.5 mean background.
    It is not a measured contrast against every point of a textured scene.
    motion_scale scales nuisance speed only; change approach_speed to change
    the physical loom. Background motion in loom_on_background is explicitly
    given in background_speed_px_s. Positive rotation is clockwise.
    """

    kind: str
    image_size: tuple[int, int] = (128, 128)
    duration_s: float = 2.0
    center_fraction: tuple[float, float] = (0.5, 0.5)
    focal_length_px: float = 96.0
    object_radius: float = 1.0
    initial_depth: float = 12.0
    approach_speed: float = 4.0
    reference_t_s: float | None = None
    motion_scale: float = 1.0
    direction_rad: float = 0.0
    contrast: float = 0.8
    polarity: int = 1
    texture_contrast: float = 0.4
    texture_kind: str = "texture"
    wavelength_px: float = 16.0
    background_speed_px_s: float = 0.0
    flicker_hz: float = 1.0
    flicker_amplitude: float = 0.15
    supersample: int = 2

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"Unknown stimulus kind {self.kind!r}")
        if len(self.image_size) != 2 or any(not isinstance(n, int) or n < 4 for n in self.image_size):
            raise ValueError("image_size must contain two integer dimensions >= 4")
        if len(self.center_fraction) != 2 or any(not np.isfinite(c) or not 0 <= c <= 1 for c in self.center_fraction):
            raise ValueError("center_fraction must contain two finite values in [0, 1]")
        for name in ("duration_s", "focal_length_px", "object_radius", "initial_depth", "approach_speed", "wavelength_px"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("motion_scale", "background_speed_px_s", "flicker_hz", "flicker_amplitude"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not np.isfinite(self.direction_rad):
            raise ValueError("direction_rad must be finite")
        if self.initial_depth <= self.approach_speed * self.duration_s:
            raise ValueError("The reference physical approach must end before contact")
        if self.reference_t_s is not None and (not np.isfinite(self.reference_t_s) or not 0 <= self.reference_t_s <= self.duration_s):
            raise ValueError("reference_t_s must be within the stimulus interval")
        for name in ("contrast", "texture_contrast"):
            if not np.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.polarity not in (-1, 1):
            raise ValueError("polarity must be -1 or +1")
        if self.texture_kind not in ("texture", "grating"):
            raise ValueError("texture_kind must be 'texture' or 'grating'")
        if not isinstance(self.supersample, int) or not 1 <= self.supersample <= 8:
            raise ValueError("supersample must be an integer between 1 and 8")
        if self.kind == "linear_expand" and _reference(self)[1] - _reference(self)[2] * _reference(self)[0] <= 0:
            raise ValueError("The reference-matched linear disk must have positive radius throughout")


def _reference(spec: StimulusSpec) -> tuple[float, float, float]:
    t_ref = spec.duration_s / 2 if spec.reference_t_s is None else spec.reference_t_s
    z_ref = spec.initial_depth - spec.approach_speed * t_ref
    radius = spec.focal_length_px * spec.object_radius / z_ref
    speed = spec.focal_length_px * spec.object_radius * spec.approach_speed / z_ref**2
    return float(t_ref), float(radius), float(speed)


def _checked_time(spec: StimulusSpec, t: float) -> float:
    if not np.isfinite(t) or t < -1e-12 or t > spec.duration_s + 1e-12:
        raise ValueError("t must be finite and within [0, duration_s]")
    return float(np.clip(t, 0.0, spec.duration_s))


def timestamps(duration_s: float, schedule: str | Real | Sequence[float] = "30hz") -> np.ndarray:
    """Return monotonically increasing times, including exactly 0 and duration.

    A numerical schedule is a sample rate in Hz. Named schedules are 15hz,
    30hz, 60hz, and irregular. The latter alternates exactly 1/30 and 3/30 s,
    starting with the shorter interval (mean rate 15 Hz over complete pairs).
    A sequence specifies repeated intervals in seconds. Rational accumulation
    avoids floating-point drift; only the last interval is shortened if the
    duration is not an integer number of complete intervals.
    """
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be positive and finite")
    if isinstance(schedule, str):
        if schedule == "irregular":
            steps = (Fraction(1, 30), Fraction(3, 30))
        elif schedule in ("15hz", "30hz", "60hz"):
            steps = (Fraction(1, int(schedule[:-2])),)
        else:
            raise ValueError(f"Unknown timestamp schedule {schedule!r}")
    elif isinstance(schedule, Real):
        if not np.isfinite(schedule) or schedule <= 0:
            raise ValueError("Sample rate must be positive and finite")
        steps = (1 / Fraction(str(float(schedule))),)
    else:
        if len(schedule) == 0 or any(not np.isfinite(step) or step <= 0 for step in schedule):
            raise ValueError("Interval schedule must contain positive finite steps")
        raw_steps = tuple(Fraction(str(float(step))) for step in schedule)
        steps = tuple(step.limit_denominator(10**12) or step for step in raw_steps)
    duration = Fraction(str(float(duration_s)))
    current = Fraction(0)
    result = [0.0]
    index = 0
    if duration / min(steps) > 1_000_000:
        raise ValueError("Schedule exceeds the one-million interval allocation limit")
    while current < duration:
        current = min(current + steps[index % len(steps)], duration)
        result.append(float(current))
        index += 1
    return np.asarray(result, dtype=np.float64)


def truth(spec: StimulusSpec, t: float) -> dict:
    """Return analytical ground truth with explicit geometry and speed units.

    angular_diameter_rad is the on-axis-equivalent diameter 2*atan(radius/f).
    Exact horizontal and vertical angular spans are also provided so an
    off-axis disk is not silently treated as centered. true_ttc_s exists only
    for a physical approach; expansion alone is not a collision annotation.
    """
    t = _checked_time(spec, t)
    h, w = spec.image_size
    center = np.asarray(((w - 1) * spec.center_fraction[0], (h - 1) * spec.center_fraction[1]), dtype=float)
    t_ref, radius_ref, speed_ref = _reference(spec)
    nuisance_speed = spec.motion_scale * speed_ref
    direction = np.array((np.cos(spec.direction_rad), np.sin(spec.direction_rad)))
    has_disk = spec.kind in _DISK_KINDS
    radius, radial_speed, depth, depth_velocity, ttc = None, None, None, None, None
    if spec.kind in ("loom", "loom_on_background", "recede"):
        physical_t = spec.duration_s - t if spec.kind == "recede" else t
        depth = spec.initial_depth - spec.approach_speed * physical_t
        depth_velocity = spec.approach_speed if spec.kind == "recede" else -spec.approach_speed
        radius = spec.focal_length_px * spec.object_radius / depth
        radial_speed = -spec.focal_length_px * spec.object_radius * depth_velocity / depth**2
        if spec.kind != "recede":
            ttc = depth / spec.approach_speed
    elif spec.kind == "linear_expand":
        radius = radius_ref + speed_ref * (t - t_ref)
        radial_speed = speed_ref
    elif spec.kind == "translate":
        radius, radial_speed = radius_ref, 0.0
        center += direction * nuisance_speed * (t - t_ref)
    target_velocity = direction * nuisance_speed if spec.kind == "translate" else np.zeros(2)
    background_speed = nuisance_speed if spec.kind == "background_translation" else spec.background_speed_px_s if spec.kind == "loom_on_background" else 0.0
    omega = nuisance_speed / radius_ref if spec.kind == "rotation" else 0.0
    angular = horizontal = vertical = None
    if has_disk:
        angular = float(2 * np.arctan(radius / spec.focal_length_px))
        offset = center - np.array(((w - 1) / 2, (h - 1) / 2))
        horizontal, vertical = (np.arctan((offset + radius) / spec.focal_length_px) - np.arctan((offset - radius) / spec.focal_length_px)).tolist()
    return {
        "t_s": t,
        "kind": spec.kind,
        "motion_class": {
            "loom": "physical_approach", "loom_on_background": "physical_approach_with_background_motion",
            "recede": "physical_recede", "linear_expand": "arbitrary_linear_enlargement",
            "translate": "constant_size_translation", "background_translation": "background_translation",
            "rotation": "background_rotation", "flicker": "stationary_texture_luminance_modulation", "static": "stationary_texture",
        }[spec.kind],
        "center_xy_px": center.tolist() if has_disk else None,
        "radius_px": float(radius) if radius is not None else None,
        "radial_speed_px_s": float(radial_speed) if radial_speed is not None else None,
        "angular_diameter_rad": angular,
        "angular_diameter_definition": "on_axis_equivalent_2atan_projected_radius_over_focal_length",
        "horizontal_angular_span_rad": horizontal,
        "vertical_angular_span_rad": vertical,
        "depth": float(depth) if depth is not None else None,
        "depth_velocity_per_s": float(depth_velocity) if depth_velocity is not None else None,
        "true_ttc_s": float(ttc) if ttc is not None else None,
        "target_velocity_px_s": target_velocity.tolist(),
        "background_velocity_px_s": (direction * background_speed).tolist(),
        "rotation_rad_s": float(omega),
        "reference_t_s": t_ref,
        "reference_radius_px": radius_ref,
        "reference_boundary_speed_px_s": speed_ref,
        "nuisance_speed_px_s": nuisance_speed,
        "speed_matching_definition": "translation_equals_motion_scale_times_physical_loom_boundary_speed_at_reference; rotation_matches_only_at_reference_radius",
        "disk_fully_visible": bool(radius <= center[0] + 0.5 and radius <= w - 0.5 - center[0] and radius <= center[1] + 0.5 and radius <= h - 0.5 - center[1]) if has_disk else None,
    }


@lru_cache(maxsize=16)
def _grid(image_size: tuple[int, int], supersample: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_size
    x = (np.arange(w * supersample, dtype=np.float64) + 0.5) / supersample - 0.5
    y = (np.arange(h * supersample, dtype=np.float64) + 0.5) / supersample - 0.5
    xx, yy = np.meshgrid(x, y)
    xx.flags.writeable = yy.flags.writeable = False
    return xx, yy


def _texture(spec: StimulusSpec, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    k = 2 * np.pi / spec.wavelength_px
    if spec.texture_kind == "grating":
        pattern = np.sin(k * (x * np.cos(spec.direction_rad) + y * np.sin(spec.direction_rad)))
    else:
        pattern = (np.sin(k * x) + np.sin(k * (0.77 * y) + 0.73) + np.sin(k * (0.6 * x + 0.8 * y) + 1.31)) / 3
    return 0.5 + spec.polarity * spec.texture_contrast * pattern / 2


def render_frame(spec: StimulusSpec, t: float) -> np.ndarray:
    """Render an antialiased float32 grayscale frame in [0, 1].

    The disk is clipped by image boundaries; it is never periodically wrapped.
    The background is an infinite analytical periodic texture. Supersampling
    and a one-subpixel linear boundary ramp avoid binary raster jumps. No
    optical-flow library, frame index, random state, or previous frame is used.
    """
    t = _checked_time(spec, t)
    ground_truth = truth(spec, t)
    x, y = _grid(tuple(spec.image_size), spec.supersample)
    textured = spec.kind in ("background_translation", "rotation", "flicker", "static", "loom_on_background")
    if textured:
        vx, vy = ground_truth["background_velocity_px_s"]
        tx, ty = x - vx * t, y - vy * t
        if spec.kind == "rotation":
            h, w = spec.image_size
            cx, cy = (w - 1) * spec.center_fraction[0], (h - 1) * spec.center_fraction[1]
            angle = ground_truth["rotation_rad_s"] * t
            dx, dy = tx - cx, ty - cy
            tx = np.cos(angle) * dx + np.sin(angle) * dy + cx
            ty = -np.sin(angle) * dx + np.cos(angle) * dy + cy
        frame = _texture(spec, tx, ty)
    else:
        frame = np.full(x.shape, 0.5, dtype=np.float64)
    if spec.kind == "flicker":
        frame += spec.flicker_amplitude * np.sin(2 * np.pi * spec.flicker_hz * t)
    if spec.kind in _DISK_KINDS:
        cx, cy = ground_truth["center_xy_px"]
        radius = ground_truth["radius_px"]
        distance = np.hypot(x - cx, y - cy)
        coverage = np.clip((radius - distance) * spec.supersample + 0.5, 0.0, 1.0)
        foreground = 0.5 + spec.polarity * spec.contrast / 2
        frame = frame * (1 - coverage) + foreground * coverage
    # Clip before pixel-area averaging: every subpixel is a bounded sensor value.
    np.clip(frame, 0.0, 1.0, out=frame)
    h, w = spec.image_size
    s = spec.supersample
    return frame.reshape(h, s, w, s).mean(axis=(1, 3)).astype(np.float32)
