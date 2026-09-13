#!/usr/bin/env python3
"""Render English publication figures from saved scores, without model inference.

Default uses the small, public score-only source bundle beside the figures.
``--refresh-sources`` reads the original local CSV/JSON/parquet artifacts and
updates that bundle. It never reads videos, runs optical flow, or tunes a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
BG = "#FAF9F5"
INK = "#18252D"
MUTED = "#586871"
GRID = "#DFE4E3"
TEAL = "#007F79"
AMBER = "#C76026"
SLATE = "#81969D"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_sources() -> None:
    paths = {
        "clips": ROOT / "outputs/v0/clip_scores.csv",
        "threshold": ROOT / "outputs/v0/threshold.yaml",
        "comparison": ROOT / "outputs/ego_motion/comparison.csv",
        "metrics": ROOT / "outputs/ego_motion/metrics.json",
        "synthetic": ROOT / "outputs/synthetic/synthetic_summary.json",
        "audit": ROOT / "outputs/review_response/audit.json",
    }
    clips = pd.read_csv(paths["clips"], dtype={"video_id": str})
    comparison = pd.read_csv(paths["comparison"])
    metrics = json.loads(paths["metrics"].read_text())
    synthetic = json.loads(paths["synthetic"].read_text())
    audit = json.loads(paths["audit"].read_text())
    hero = []
    traces = []
    for video_id in ("00300", "01538"):
        row = clips.loc[clips.video_id == video_id].iloc[0]
        fields = ["video_id", "label", "scene", "split", "s_peak", "t_peak",
                  "time_of_alert", "time_of_event", "theta_used"]
        record = {k: (None if pd.isna(row[k]) else row[k]) for k in fields}
        record["label"] = int(record["label"])
        path = ROOT / f"outputs/v0/frames/{video_id}.parquet"
        record["trace_available"] = path.exists()
        if path.exists():
            paths[f"trace_{video_id}"] = path
            trace = pd.read_parquet(path)[["t", "S"]]
            if record["label"]:
                trace = trace[trace.t < record["time_of_event"]]
            assert np.isclose(trace.S.max(), record["s_peak"], atol=1e-12)
            trace.insert(0, "video_id", video_id)
            traces.append(trace)
        hero.append(record)
    confirmation = comparison[comparison.split == "confirmation"]
    assert len(confirmation) == 4 and (confirmation.n_scored == 100).all()
    syn = audit["synthetic"]["stimulus_scores"]
    assert np.isclose(syn["loom"], synthetic["stimuli"]["loom"]["late_mean_S"])
    assert np.isclose(syn["translate_1px"], synthetic["stimuli"]["translate"]["late_mean_S"])
    source = {
        "schema_version": 1,
        "scope": "Saved scalar scores only; no video pixels or model inference.",
        "original_files": {str(p.relative_to(ROOT)): sha(p) for p in paths.values()},
        "hero": hero,
        "theta": float(yaml.safe_load(paths["threshold"].read_text())["theta"]),
        "hero_trace_policy": "Event clip: t < time_of_event. Negative: all saved analysis frames. Original clip timestamps; common y scale.",
        "confirmation": confirmation[["model", "auroc", "auprc", "n_positive", "n_negative", "auroc_ci_low", "auroc_ci_high"]].to_dict("records"),
        "primary_comparison": {k: metrics["primary_comparison"][k] for k in ["candidate", "comparator", "method", "outcomes"]},
        "synthetic": [
            {"label": "Expanding disk", "score": syn["loom"]},
            {"label": "Receding disk", "score": syn["recede"]},
            {"label": "Translate 1 px/frame", "score": syn["translate_1px"]},
            {"label": "Translate 2 px/frame", "score": syn["translate_2px"]},
            {"label": "Translate 4 px/frame", "score": syn["translate_4px"]},
            {"label": "Uniform flicker", "score": synthetic["stimuli"]["flicker"]["late_mean_S"]},
        ],
        "synthetic_metric": audit["synthetic"]["note"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figure_sources.json").write_text(json.dumps(source, indent=2) + "\n")
    if traces:
        pd.concat(traces).to_csv(OUT / "hero_scores.csv", index=False, float_format="%.17g")


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 16,
        "figure.facecolor": BG, "axes.facecolor": BG,
        "text.color": INK, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": False, "savefig.facecolor": BG,
        "lines.solid_capstyle": "round",
    })


def canvas(width: int = 1600, height: int = 900):
    return plt.figure(figsize=(width / 100, height / 100), dpi=100)


def top(fig, label: str, title: str, subtitle: str = "") -> None:
    fig.text(.05, .946, label, fontsize=13, color=TEAL, weight="bold", va="top")
    fig.text(.05, .88, title, fontsize=32, weight="bold", va="top")
    if subtitle:
        fig.text(.05, .802, subtitle, fontsize=16, color=MUTED, va="top")


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=100)
    plt.close(fig)


def x_card(source: dict) -> None:
    fig = canvas()
    top(fig, "FLY-TTC  /  THE COUNTEREXAMPLE", "Global expansion is not collision.",
        "Same frozen detector. Two examples from the first 80 evaluation clips.")
    traces_path = OUT / "hero_scores.csv"
    traces = pd.read_csv(traces_path, dtype={"video_id": str}) if traces_path.exists() else None
    colors = [TEAL, AMBER]
    for i, row in enumerate(source["hero"]):
        ax = fig.add_axes([.09 + i * .48, .305, .38, .34])
        color = colors[i]
        ax.set_ylim(0, 18)
        ax.set_yticks([0, 4, 8, 12, 16])
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.tick_params(length=0, pad=9, labelsize=15)
        ax.set_axisbelow(True)
        ax.axhline(source["theta"], color=MUTED, linewidth=1.8, linestyle=(0, (4, 4)), zorder=2)
        ax.set_ylabel("Frozen detector score S", fontsize=15, labelpad=10)
        title = "Event (highway)" if i == 0 else "Non-event (higher peak)"
        ax.text(0, 1.25, title, transform=ax.transAxes, fontsize=24, weight="bold", color=color)
        ax.text(0, 1.12, f"Clip {row['video_id']}  /  peak {row['s_peak']:.2f}",
                transform=ax.transAxes, fontsize=17, color=MUTED)
        data = traces[traces.video_id == row["video_id"]] if traces is not None else None
        if data is not None and len(data) and row["trace_available"]:
            ax.plot(data.t, data.S, color=color, linewidth=3.2, zorder=3)
            ax.fill_between(data.t, 0, data.S, color=color, alpha=.08)
            if i == 0:
                ax.set_xlim(2, 10.42)
                ax.set_xticks([2, 4, 6, 8, 10])
                ax.axvline(row["time_of_alert"], color=SLATE, linestyle=(0, (2, 3)), linewidth=1.5)
                ax.axvline(row["time_of_event"], color=INK, linestyle=(0, (5, 3)), linewidth=1.6)
                ax.text(row["time_of_alert"] - .13, 17.65, "alert", color=MUTED, fontsize=13, ha="right", va="top")
                ax.text(row["time_of_event"] + .06, 17.65, "event", color=INK, fontsize=13, rotation=90, va="top")
            else:
                ax.set_xlim(15.72, 24.9)
                ax.set_xticks([16, 18, 20, 22, 24])
            ax.scatter([row["t_peak"]], [row["s_peak"]], s=65, color=color, edgecolor=BG, linewidth=1.5, zorder=4)
            ax.set_xlabel("Time in original clip (s)", fontsize=16, labelpad=10)
        else:
            ax.bar([0], [row["s_peak"]], width=.55, color=color)
            ax.set_xlim(-.75, .75)
            ax.set_xticks([0], ["Saved pre-event peak" if row["label"] else "Saved clip peak"])
            ax.set_xlabel("Peak only; frame trace unavailable", fontsize=15)
    fig.text(.09, .185, f"Dashed horizontal line: frozen threshold θ = {source['theta']:.2f}. Event trace stops before the event.",
             fontsize=15, color=MUTED)
    fig.add_artist(Line2D([.05, .95], [.143, .143], transform=fig.transFigure, color=GRID, lw=1.2))
    fig.text(.05, .083, "Held-out AUROC 0.49 · not a connectome", fontsize=29, weight="bold")
    fig.text(.05, .037, "AUROC: separate, clip-disjoint confirmation set (new 100). Traces above: first 80.",
             fontsize=14, color=MUTED)
    save(fig, "x_card.png")


def confirmation(source: dict) -> None:
    fig = canvas()
    top(fig, "FLY-TTC  /  THE REPLICATION", "Disjoint confirmation set (100 clips)",
        "50 events + 50 non-events. Frozen calibration. All four AUROCs are near chance.")
    by_model = {r["model"]: r for r in source["confirmation"]}
    names = ["v0_original", "raw_flow_no_blob", "affine_residual_no_blob", "frame_difference"]
    labels = ["v0 original", "Raw flow\n(no blob)", "Affine residual\n(no blob)", "Frame difference"]
    values = [by_model[n]["auroc"] for n in names]
    ax = fig.add_axes([.1, .28, .81, .46])
    ax.set_ylim(.45, .70)
    ax.set_yticks(np.arange(.45, .701, .05))
    ax.set_ylabel("AUROC", labelpad=14)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=10, labelsize=17)
    ax.bar(np.arange(4), np.array(values) - .45, bottom=.45, width=.56,
           color=[INK, SLATE, TEAL, AMBER], zorder=3)
    ax.axhline(.50, color=MUTED, linestyle=(0, (4, 4)), linewidth=1.7, zorder=4)
    ax.text(3.53, .503, "chance", fontsize=14, ha="left", color=MUTED)
    ax.set_xlim(-.65, 3.85)
    ax.set_xticks(np.arange(4), labels)
    for i, value in enumerate(values):
        ax.text(i, max(.5, value) + .017, f"{value:.3f}", ha="center", fontsize=27, weight="bold")
    d = source["primary_comparison"]["outcomes"]["delta_auroc"]
    fig.text(.1, .158, f"Camera-motion subtraction: ΔAUROC {d['estimate']:+.4f}", fontsize=24, weight="bold")
    fig.text(.1, .109, f"Affine residual minus raw flow (both without blob); paired 95% CI [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}].",
             fontsize=17, color=MUTED)
    fig.text(.1, .047, "Descriptive bar scale starts at 0.45. Full uncertainty intervals and definitions are in the saved metrics.",
             fontsize=13, color=MUTED)
    save(fig, "confirmation_metrics.png")


def selectivity(source: dict) -> None:
    fig = canvas()
    top(fig, "FLY-TTC  /  THE SYNTHETIC CHECK", "Synthetic gate (not Nexar performance)",
        "Original v0 proxy · raw late-mean scores · no real-video normalization")
    ax = fig.add_axes([.265, .28, .63, .45])
    items = source["synthetic"]
    values = [r["score"] for r in items]
    y = np.arange(len(items))
    colors = [TEAL, SLATE, AMBER, AMBER, AMBER, SLATE]
    ax.barh(y, values, color=colors, height=.58, zorder=3)
    ax.set_yticks(y, [r["label"] for r in items], fontsize=18)
    ax.invert_yaxis()
    ax.set_xlim(0, .74)
    ax.set_xticks([0, .2, .4, .6])
    ax.set_xlabel("Raw score S (final 15 frames)", labelpad=12, fontsize=17)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=12, labelsize=17)
    ax.axvline(values[0], color=TEAL, lw=1.4, linestyle=(0, (2, 3)), alpha=.6)
    for i, value in enumerate(values):
        ax.text(value + .011, i, f"{value:.3f}", va="center", fontsize=20, weight="bold", color=colors[i])
    fig.add_artist(Line2D([.05, .95], [.155, .155], transform=fig.transFigure, color=GRID, lw=1.2))
    fig.text(.05, .099, "Faster translation outscores expansion.", fontsize=28, weight="bold")
    fig.text(.05, .048, "This is a proxy bug, not a fly.", fontsize=22, color=MUTED)
    save(fig, "synthetic_selectivity.png")


def header(source: dict) -> None:
    fig = canvas(1280, 640)
    fig.text(.06, .89, "A MEASURED NEGATIVE RESULT", fontsize=14, weight="bold", color=TEAL)
    fig.text(.06, .60, "fly-ttc", fontsize=90, weight="bold")
    fig.text(.065, .43, "Looming proxy on public dashcam crashes · AUROC 0.49", fontsize=21)
    fig.text(.065, .33, "New, clip-disjoint confirmation set: 100 clips", fontsize=17, color=MUTED)
    fig.add_artist(Line2D([.06, .94], [.235, .235], transform=fig.transFigure, color=GRID, lw=1.5))
    fig.text(.065, .13, "NOT A CONNECTOME", fontsize=16, weight="bold", color=MUTED)
    fig.text(.94, .13, "github.com/spacegiyou/fly-ttc", fontsize=17, ha="right", color=INK)
    fig.text(.94, .78, "0.49", fontsize=63, weight="bold", ha="right", color=TEAL)
    fig.text(.935, .725, "CONFIRMATION AUROC", fontsize=12, ha="right", color=MUTED)
    save(fig, "repo_header.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-sources", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.refresh_sources:
        refresh_sources()
    source = json.loads((OUT / "figure_sources.json").read_text())
    style()
    x_card(source)
    confirmation(source)
    selectivity(source)
    header(source)
    print("Rendered 4 English figures from saved scores; no inference performed.")


if __name__ == "__main__":
    main()
