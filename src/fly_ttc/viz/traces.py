"""Shared-scale score panels and honest empty-data figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def unavailable_figure(path, title, reason, dpi=130):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, reason, ha="center", va="center", transform=ax.transAxes, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def shared_limits(frames: dict[str, pd.DataFrame], theta: float | None):
    arrays = [pd.to_numeric(df.S, errors="coerce").to_numpy() for df in frames.values() if "S" in df]
    values = np.concatenate(arrays) if arrays else np.array([])
    values = values[np.isfinite(values)]
    if theta is not None and np.isfinite(theta):
        values = np.append(values, theta)
    if not len(values):
        return [-1.0, 1.0]
    low, high = float(values.min()), float(values.max())
    margin = max((high - low) * 0.08, 0.1)
    return [low - margin, high + margin]


def plot_traces(rows, frames, path, theta, y_limits, label, n=12, dpi=130):
    """Draw up to n clips, labeling unavailable slots rather than duplicating."""
    columns = 3
    nrows = max(1, int(np.ceil(n / columns)))
    fig, axes = plt.subplots(nrows, columns, figsize=(15, 2.7 * nrows), squeeze=False)
    available = [row for _, row in rows.iterrows() if str(row.video_id) in frames][:n]
    for index, ax in enumerate(axes.flat):
        if index >= n:
            ax.axis("off")
            continue
        ax.set_ylim(y_limits)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Looming score (a.u.)")
        ax.grid(alpha=0.2)
        if index >= len(available):
            ax.text(0.5, 0.5, "No additional scored clip", ha="center", va="center", transform=ax.transAxes)
            continue
        row = available[index]
        data = frames[str(row.video_id)]
        ax.plot(data.t, data.S, linewidth=1.3, color="#1868b7", label="S(t)")
        if theta is not None:
            ax.axhline(theta, color="#333333", linestyle="--", linewidth=1, label="theta")
        if int(row.label) == 1:
            alert, event = row.get("time_of_alert"), row.get("time_of_event")
            if pd.notna(alert):
                ax.axvline(float(alert), color="#e38a15", linewidth=1.2, label="alert")
            if pd.notna(event):
                ax.axvline(float(event), color="#c82d2d", linewidth=1.2, label="event")
                end = max(float(data.t.max()), float(event))
                ax.axvspan(float(event), end, color="#999999", alpha=0.12)
        ax.set_title(f"{row.video_id} | {row.get('split', 'unspecified')}", fontsize=10)
        if index == 0:
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{label} score traces: {len(available)} clips | shared y-axis; post-event shaded/excluded", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(Path(path), dpi=dpi)
    plt.close(fig)
    return [str(row.video_id) for row in available]
