"""Download, evaluate and inspect the fixed v0 looming proxy on Nexar train."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
import logging
from pathlib import Path
import platform
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from fly_ttc.paths import load_config

LOG = logging.getLogger("fly_ttc")


def _model_config(config):
    from fly_ttc.models.v0_expansion import V0Config

    values = dict(config["v0"])
    values["normalization_std_floor"] = config["eval"]["normalization_std_floor"]
    names = {field.name for field in fields(V0Config)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"Unknown v0 configuration fields: {sorted(unknown)}")
    return V0Config(**values)


def extract_clip(row, config, snapshot_targets=None, scorer=None):
    """The only analysis frame loop, shared by run and debug/diagnostics."""
    from fly_ttc.preprocess.video import iter_video_frames, probe_video
    from fly_ttc.models.v0_expansion import ExpansionScorer

    path = Path(row["video_path"])
    if not path.is_file():
        return pd.DataFrame(), {"status": "missing_video"}, []
    if int(row["label"]) == 1 and pd.isna(row["time_of_event"]):
        return pd.DataFrame(), {"status": "no_event_time"}, []
    try:
        info = probe_video(path)
        video_cfg = config["video"]
        metadata_fps = row.get("fps", np.nan)
        if pd.notna(metadata_fps) and abs(float(metadata_fps) - info["fps"]) >= video_cfg["fps_warning_delta"]:
            LOG.warning("%s: metadata fps=%s differs from actual fps=%s", row["video_id"], metadata_fps, info["fps"])
        offset = video_cfg["time_offset_s"]
        if int(row["label"]) == 1:
            start = max(offset, float(row["time_of_event"]) - video_cfg["pos_window_pre_s"])
            end = min(info["duration_s"] + offset, float(row["time_of_event"]) + video_cfg["pos_window_post_s"])
        else:
            margin = video_cfg["negative_margin_s"]
            length = video_cfg["pos_window_pre_s"] + video_cfg["pos_window_post_s"]
            start = max(margin, (info["duration_s"] - length) / 2) + offset
            end = min(info["duration_s"] - margin, start - offset + length) + offset
        if end <= start:
            raise ValueError("No usable analysis window")
        scorer = scorer or ExpansionScorer(config=_model_config(config), flow_config=config["flow"])
        records, snapshots = [], {}
        targets = list(snapshot_targets or [])
        for t, gray in iter_video_frames(path, start_s=start, end_s=end,
                                          target_fps=video_cfg["target_fps"],
                                          short_side=video_cfg["short_side"], time_offset_s=offset):
            scores = scorer.update(gray)
            records.append({"t": float(t), **scores})
            for index, target in enumerate(targets):
                if scores.get("flow_valid", True) and t <= target:
                    # Blur the entire frame before storing; no original pixels enter reports.
                    ksize = int(config["report"]["privacy_blur_ksize"])
                    if ksize < 3 or ksize % 2 == 0:
                        raise ValueError("privacy_blur_ksize must be odd and >=3")
                    blurred = cv2.GaussianBlur(np.clip(gray * 255, 0, 255).astype(np.uint8), (ksize, ksize), 0)
                    snapshots[index] = {"image": cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB),
                                        "flow": scorer.last_flow.copy(),
                                        "radial": np.maximum(scorer.last_radial, 0).copy(),
                                        "t": t, "requested_t": target}
                    snapshots[index].update(getattr(scorer, "snapshot_fields", {}))
        frame_df = pd.DataFrame(records)
        if "flow_valid" in frame_df:
            frame_df = frame_df.loc[frame_df["flow_valid"].astype(bool)].reset_index(drop=True)
        if frame_df.empty:
            raise ValueError("No decodable frame pairs in analysis window")
        return frame_df, {**info, "status": "ok"}, [snapshots[i] for i in sorted(snapshots)]
    except (ValueError, OSError, RuntimeError, cv2.error) as exc:
        LOG.warning("%s: decode_error: %s", row["video_id"], exc)
        return pd.DataFrame(), {"status": "decode_error"}, []


def _write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [clean(v) for v in item]
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, (np.bool_,)):
            return bool(item)
        return item
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_v0(config_path):
    from fly_ttc.data.nexar import load_manifest
    from fly_ttc.data.sampling import stratified_split
    from fly_ttc.models.v0_expansion import fit_normalization, score_components
    from fly_ttc.models.thresholds import choose_threshold
    from fly_ttc.eval.lead_time import lead_times
    from fly_ttc.eval.metrics import summarize_metrics
    from fly_ttc.eval.failure_tags import failure_tags

    config = load_config(config_path)
    _model_config(config)
    if config["flow"]["method"] != "farneback":
        raise ValueError("v0 currently supports flow.method=farneback only; RAFT is optional and not implemented")
    if config["video"]["target_fps"] <= 0 or config["video"]["short_side"] < 3:
        raise ValueError("target_fps must be positive and short_side must be at least 3")
    destination = Path(config["paths"]["output_dir"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stage a complete run before publishing. A failed rerun cannot mix old
    # reports with new calibration or delete the previous run's frame traces.
    out = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    for directory in ("frames", "_report_frames", "diagnostics"):
        target = out / directory
        target.mkdir(exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = load_manifest(config["paths"]["manifest"])
    manifest.to_csv(out / "manifest.csv", index=False)
    evaluation_ids, validation_ids = stratified_split(manifest, config["eval"]["val_frac"], config["seed"])
    validation_set = set(map(str, validation_ids))
    raw_frames, infos = {}, {}
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="v0 flow", unit="clip"):
        video_id = str(row["video_id"])
        frame_df, info, _ = extract_clip(row, config)
        infos[video_id] = info
        if info["status"] == "ok":
            raw_frames[video_id] = frame_df
        tqdm.write(f"{video_id}: {info['status']} ({len(frame_df)} frame pairs)")
    negative_val_ids = [str(row.video_id) for row in manifest.itertuples()
                        if int(row.label) == 0 and str(row.video_id) in validation_set and str(row.video_id) in raw_frames]
    if not negative_val_ids:
        raise ValueError("No usable negative validation clips; cannot calibrate. Check manifest and downloaded videos.")
    model_cfg = _model_config(config)
    normalization = fit_normalization([raw_frames[i] for i in negative_val_ids], model_cfg)
    normalization.update(source="negative_validation", video_ids=negative_val_ids)
    for frame_df in raw_frames.values():
        frame_df["S"] = score_components(frame_df, normalization, model_cfg)
    maxima = [float(raw_frames[i]["S"].max()) for i in negative_val_ids]
    threshold = choose_threshold(maxima, config["eval"]["fpr_target"])
    theta = float(threshold["theta"])
    threshold.update(normalization=normalization, seed=int(config["seed"]),
                     evaluation_ids=list(map(str, evaluation_ids)), validation_ids=list(map(str, validation_ids)),
                     usable_negative_validation_ids=negative_val_ids, comparison="S > theta")
    (out / "threshold.yaml").write_text(yaml.safe_dump(threshold, sort_keys=False), encoding="utf-8")
    reference_frames = pd.concat([raw_frames[i] for i in negative_val_ids], ignore_index=True)
    percentile = config["failure_tags"]["reference_percentile"]
    reference = {f"{key}_p95": float(np.percentile(reference_frames[key], percentile))
                 for key in ("S_rad", "flow_horizontal", "flow_median")}
    _write_json(out / "failure_reference.json", reference)
    rows = []
    for _, row in manifest.iterrows():
        video_id = str(row["video_id"])
        info = infos[video_id]
        result = {"video_id": video_id, "label": int(row["label"]), "scene": row.get("scene", "Unknown"),
                  "weather": row.get("weather", "Unknown"), "light": row.get("light_conditions", "Unknown"),
                  "time_of_alert": row["time_of_alert"], "time_of_event": row["time_of_event"],
                  "duration_s": info.get("duration_s", np.nan), "fps": info.get("fps", np.nan),
                  "theta_used": theta, "n_frames": 0, "status": info["status"],
                  "split": "validation" if video_id in validation_set else "evaluation", "failure_tags": ""}
        if info["status"] == "ok":
            frame_df = raw_frames[video_id]
            frame_df["above_theta"] = frame_df["S"] > theta
            result.update(lead_times(frame_df["t"].to_numpy(), frame_df["S"].to_numpy(), theta,
                                     row["time_of_alert"], row["time_of_event"],
                                     lead_windows_ms=config["eval"]["lead_windows_ms"],
                                     alert_window_pre_s=config["eval"]["alert_window_pre_s"]))
            result["n_frames"] = len(frame_df)
            result["failure_tags"] = ";".join(failure_tags(frame_df, result, reference, config))
            save_dir = "frames" if result["label"] == 1 or result.get("false_positive", False) else "_report_frames"
            frame_df.to_parquet(out / save_dir / f"{video_id}.parquet", index=False)
        rows.append(result)
    clips = pd.DataFrame(rows)
    clips.to_csv(out / "clip_scores.csv", index=False)
    metrics = summarize_metrics(clips)
    _write_json(out / "metrics.json", metrics)
    eligible = clips.loc[(clips["status"] == "ok") & (clips["label"] == 1) & clips["hit_event"].fillna(False).astype(bool)]
    good = eligible.loc[eligible["failure_tags"].str.contains("good_loom", na=False)]
    if not eligible.empty:
        selected = (good if not good.empty else eligible).iloc[0]
        source_row = manifest.loc[manifest["video_id"].astype(str) == selected["video_id"]].iloc[0]
        offsets = config["report"]["snapshot_offsets_s"]
        alert = float(source_row["time_of_alert"])
        event = float(source_row["time_of_event"])
        targets = [alert + offsets[0], alert + offsets[1], event + offsets[2]]
        _, _, snapshots = extract_clip(source_row, config, targets)
        if snapshots:
            np.savez_compressed(out / "diagnostics" / f"{selected['video_id']}.npz",
                                **{key: np.stack([snapshot[key] for snapshot in snapshots])
                                   for key in ("image", "flow", "radial", "t", "requested_t")})
    environment = {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
                   "platform": platform.platform(), "model": "v0", "biological_claim": "looming proxy inspired by LPLC2/GF"}
    _write_json(out / "environment.json", environment)
    _write_json(out / "run_state.json", {"status": "complete", "n_requested": len(manifest), "n_usable": len(raw_frames)})
    if destination.exists():
        backup = out.with_name(out.name.replace(".staging-", ".previous-"))
        destination.rename(backup)
        LOG.info("Previous run preserved at %s", backup)
    out.rename(destination)
    LOG.info("Saved %s (%d/%d usable clips)", destination, len(raw_frames), len(manifest))
    return destination


def debug_clip(args):
    from fly_ttc.data.nexar import load_manifest
    from fly_ttc.models.v0_expansion import score_components

    config = load_config(args.config)
    manifest = load_manifest(config["paths"]["manifest"])
    matches = manifest.loc[manifest["video_id"].astype(str) == args.video_id]
    if matches.empty:
        raise ValueError(f"Video ID not present in manifest: {args.video_id}")
    run = Path(config["paths"]["output_dir"])
    threshold_path = run / "threshold.yaml"
    if not threshold_path.exists():
        raise ValueError("Run v0 calibration first; debug uses the saved negative-validation statistics")
    threshold = yaml.safe_load(threshold_path.read_text())
    frame_df, info, _ = extract_clip(matches.iloc[0], config)
    if info["status"] != "ok":
        raise ValueError(f"Cannot decode {args.video_id}: {info['status']}")
    frame_df["S"] = score_components(frame_df, threshold["normalization"], _model_config(config))
    frame_df["above_theta"] = frame_df["S"] > threshold["theta"]
    destination = run / "debug"
    destination.mkdir(exist_ok=True)
    if args.save_frames:
        frame_df.to_parquet(destination / f"{args.video_id}.parquet", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame_df["t"], frame_df["S"], label="v0 proxy")
    ax.axhline(threshold["theta"], ls="--", color="gray", label="theta")
    for column, color in (("time_of_alert", "orange"), ("time_of_event", "red")):
        value = matches.iloc[0][column]
        if pd.notna(value):
            ax.axvline(value, color=color, label=column)
    ax.set(xlabel="Time (s)", ylabel="Score (arbitrary units)", title=args.video_id)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / f"{args.video_id}.png", dpi=config["report"]["dpi"])
    plt.close(fig)
    LOG.info("Debug trace: %s", destination / f"{args.video_id}.png")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "experiment":
        from fly_ttc.experiments.ego_motion import main as experiment_main
        return experiment_main(argv[1:])
    if argv and argv[0] == "download":
        from fly_ttc.data.nexar import download_main
        return download_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="Download Nexar metadata or a balanced train subset")
    sub.add_parser("experiment", help="Frozen four-arm motion ablation and new confirmation clips")
    run = sub.add_parser("run", help="Extract flow, calibrate using negative validation, and evaluate")
    run.add_argument("--model", choices=["v0", "v1"], default="v0")
    run.add_argument("--config", default="configs/default.yaml")
    report = sub.add_parser("report", help="Build Korean Markdown and HTML reports")
    report.add_argument("--run", default="outputs/v0")
    debug = sub.add_parser("debug", help="Inspect one clip with the saved calibration")
    debug.add_argument("--video-id", required=True)
    debug.add_argument("--model", choices=["v0", "v1"], default="v0")
    debug.add_argument("--config", default="configs/default.yaml")
    debug.add_argument("--save-frames", action="store_true", help="Save score rows only; never original frames")
    args = parser.parse_args(argv)
    try:
        if getattr(args, "model", "v0") == "v1":
            raise ValueError("v1 is not implemented. Complete and inspect outputs/v0/REPORT.md before starting v1.")
        if args.command == "run":
            run_v0(args.config)
        elif args.command == "report":
            from fly_ttc.viz.report import make_report
            make_report(Path(args.run))
        elif args.command == "debug":
            debug_clip(args)
        return 0
    except (ValueError, FileNotFoundError, OSError) as exc:
        LOG.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
