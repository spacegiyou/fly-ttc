"""OpenCV decoding on the source frame-index clock."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import warnings

import cv2
import numpy as np


def probe_video(path: str | Path, metadata_fps: float | None = None) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0 or n_frames <= 0:
            raise OSError(f"Invalid video timing: {path}; fps={fps}, n_frames={n_frames}")
        if metadata_fps is not None and abs(fps - metadata_fps) >= 5:
            warnings.warn(f"Actual fps {fps:.3f} differs from metadata fps {metadata_fps:.3f}: {path}", stacklevel=2)
        return {"duration_s": n_frames / fps, "fps": fps, "n_frames": n_frames,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()


def iter_video_frames(path: str | Path, start_s: float, end_s: float, target_fps: float,
                      short_side: int, time_offset_s: float = 0.0) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (source frame_index / fps + offset, float32 gray image in [0,1]).

    Input window boundaries use the same adjusted clock as yielded timestamps.
    The first frame at/after each sampling deadline is selected. No repeated or
    interpolated frames are generated when target_fps exceeds source fps.
    """
    if target_fps <= 0 or short_side <= 0:
        raise ValueError("target_fps and short_side must be positive")
    if not np.isfinite([start_s, end_s, target_fps, time_offset_s]).all():
        raise ValueError("Video timing values must be finite")
    info = probe_video(path)
    fps = info["fps"]
    source_start = max(0.0, start_s - time_offset_s)
    source_end = min(info["duration_s"], end_s - time_offset_s)
    if source_end < source_start:
        return
    first = max(0, int(np.ceil(source_start * fps - 1e-8)))
    cap = cv2.VideoCapture(str(path))
    decoded = 0
    try:
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        index = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        next_deadline = source_start
        interval = 1 / min(target_fps, fps)
        while index < info["n_frames"] and index / fps <= source_end + 1e-9:
            ok, frame = cap.read()
            if not ok:
                raise OSError(f"Video decoding stopped at frame {index}: {path}")
            source_t = index / fps
            index += 1
            if source_t + 1e-9 < next_deadline:
                continue
            h, w = frame.shape[:2]
            scale = short_side / min(h, w)
            frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                               interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            decoded += 1
            yield source_t + time_offset_s, gray
            # Advance deadlines from the requested clock, avoiding accumulated frame jitter.
            while next_deadline <= source_t + 1e-9:
                next_deadline += interval
        if decoded == 0 and source_end > source_start:
            raise OSError(f"No frames decoded in requested window: {path}")
    finally:
        cap.release()
