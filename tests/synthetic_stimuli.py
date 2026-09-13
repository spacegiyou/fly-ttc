"""Deterministic stimuli for tests only; never presented as Nexar samples."""

from __future__ import annotations

import numpy as np


def stimulus_clips(size: int = 192, n_frames: int = 60) -> dict[str, list[np.ndarray]]:
    """Matched expanding/receding disks, moving grid, and uniform flashes.

    Disk radius grows quadratically from 12 to 70 pixels (up to 1.95 px/pair)
    over 60 frames. The receding clip is its exact reverse. The grid translates
    one pixel per pair, comparable to the disk's mean boundary displacement.
    The edge disk has identical radii and speeds, centered on the right edge.
    """
    y, x = np.mgrid[:size, :size].astype(np.float32)
    radii = 12 + 58 * np.linspace(0, 1, n_frames) ** 2

    def disk(radius: float, cx: float) -> np.ndarray:
        # A one-pixel anti-alias ramp keeps fractional boundary motion smooth.
        return np.clip(np.hypot(x - cx, y - (size - 1) / 2) - radius + 0.5, 0, 1).astype(np.float32)

    center = [disk(radius, (size - 1) / 2) for radius in radii]
    edge = [disk(radius, size - 1) for radius in radii]
    translate = [
        (0.5 + 0.5 * np.sin(2 * np.pi * (x - frame) / 16) * np.sin(2 * np.pi * y / 16)).astype(np.float32)
        for frame in range(n_frames)
    ]
    flicker = [np.full((size, size), 0.2 if frame % 2 else 0.8, np.float32) for frame in range(n_frames)]
    return {"loom": center, "recede": center[::-1], "translate": translate, "flicker": flicker, "edge": edge}
