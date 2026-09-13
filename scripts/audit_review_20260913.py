#!/usr/bin/env python3
"""Read-only audit of the supplied review and frozen v0/affine artifacts.

Writes only to outputs/review_response; does not decode/download Nexar videos,
fit statistics, alter thresholds, or modify any frozen experiment artifact.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from io import BytesIO
from pathlib import Path
import platform
import re
import sys
import zipfile

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from fly_ttc.models.v0_expansion import ExpansionScorer, V0Config, score_components
from fly_ttc.preprocess.expansion import center_weights, divergence, radial_flow
from synthetic_stimuli import stimulus_clips


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(frame: pd.DataFrame) -> dict:
    """Recompute from continuous scores and recorded decisions independently."""
    y = frame.label.to_numpy(dtype=int)
    pred = frame.predicted_positive.to_numpy(dtype=bool)
    return dict(n=len(frame), AUROC=float(roc_auc_score(y, frame.s_peak)),
                AP=float(average_precision_score(y, frame.s_peak)),
                FPR=float(pred[y == 0].mean()), TPR=float(pred[y == 1].mean()))


def eligible_frames(frame: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    result = frame.loc[frame.t < float(row.time_of_event)] if row.label else frame
    if not len(result):
        raise ValueError(f"No eligible frames for {row.video_id}")
    return result


def peak_fit_record(frame: pd.DataFrame, row: pd.Series) -> dict:
    """Peak-state association, explicitly not an event-level fallback cause."""
    eligible = eligible_frames(frame, row)
    peak = eligible.loc[eligible.S.idxmax()]
    rejected = ~eligible.fit_valid.astype(bool)
    return dict(video_id=row.video_id, split=row.split, label=int(row.label),
                predicted_positive=bool(row.predicted_positive),
                peak_fit_valid=bool(peak.fit_valid), peak_fit_reason=str(peak.fit_reason),
                t_peak=float(peak.t), n_frames=len(eligible),
                rejected_frames=int(rejected.sum()),
                rejected_frame_fraction=float(rejected.mean()),
                above_theta_rejected_frames=int((rejected & eligible.above_theta.astype(bool)).sum()),
                above_theta_accepted_frames=int((~rejected & eligible.above_theta.astype(bool)).sum()))


def classify_pytest_log(log: str) -> dict:
    failures = [line for line in log.splitlines() if line.startswith("FAILED ")]
    counts = Counter()
    for line in failures:
        if "No module named 'markdown'" in line:
            counts["missing_markdown_dependency"] += 1
        elif "Unable to find a usable engine" in line:
            counts["missing_parquet_write_engine_dependency"] += 1
        elif "2.7755575615628914e-17 == 0" in line:
            counts["wilson_zero_boundary_roundoff"] += 1
        else:
            counts["unclassified"] += 1
    match = re.search(r"(\d+) failed, (\d+) passed in ([\d.]+)s", log)
    return dict(failed=int(match[1]), passed=int(match[2]), elapsed_s=float(match[3]),
                failure_breakdown=dict(counts), failures=failures)


def synthetic_audit() -> dict:
    clips = stimulus_clips()
    y, x = np.mgrid[:192, :192].astype(np.float32)
    cases = {key: clips[key] for key in ("loom", "recede")}
    for speed in (1, 2, 4):
        cases[f"translate_{speed}px"] = [
            (0.5 + 0.5 * np.sin(2 * np.pi * (x - speed * t) / 16)
             * np.sin(2 * np.pi * y / 16)).astype(np.float32) for t in range(60)]
    observed = {}
    for name, frames in cases.items():
        scorer = ExpansionScorer()
        observed[name] = float(np.mean([scorer.update(frame)["S"] for frame in frames][-15:]))
    # 256x455 is the configured 256px short side at a rounded 16:9 aspect.
    # The supplied scalar audit did not record its array shape. This explicit
    # reconstruction matches those scalars; shape affects the center weighting.
    uniform = []
    for speed in (1.0, 2.0, 4.0):
        flow = np.zeros((256, 455, 2), np.float32)
        flow[..., 0] = speed
        weights = center_weights(flow.shape[:2])
        uniform.append(dict(translation_px=speed,
                            S_div=float(np.mean(np.maximum(divergence(flow), 0) * weights)),
                            S_rad=float(np.mean(np.maximum(radial_flow(flow), 0) * weights))))
    return dict(stimulus_scores=observed, analytic_uniform_translation=uniform,
                uniform_flow_shape=[256, 455],
                uniform_shape_provenance="Explicit 256px short side/rounded 16:9 reconstruction; review JSON omitted shape",
                selection_counterexample=observed["translate_2px"] > observed["loom"],
                note="Original 192px/60-frame deterministic stimuli, final 15-frame mean, no Nexar normalization")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    original_input = ROOT.parent / "fly_project_review_2026-09-13"
    default_input = original_input if original_input.exists() else ROOT / "docs/review_inputs/fly_project_review_2026-09-13"
    parser.add_argument("--review-dir", type=Path, default=default_input)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/review_response")
    args = parser.parse_args()
    supplied = sorted(p for p in args.review_dir.iterdir() if p.is_file())
    supplied.append(args.review_dir.parent / "FLY_PROJECT_REVIEW_2026-09-13_KO.md")
    review = json.loads((args.review_dir / "independent_audit.json").read_text())
    protected = {str(p.relative_to(ROOT)): sha(p)
                 for name in ("v0", "ego_motion", "synthetic")
                 for p in (ROOT / "outputs" / name).rglob("*") if p.is_file()}
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  review_inputs=[dict(path=str(p), size_bytes=p.stat().st_size, sha256=sha(p)) for p in supplied],
                  identical_markdown=(supplied[0].read_bytes() == supplied[-1].read_bytes()),
                  scope="Saved data recomputation and synthetic counterexamples only; no Nexar rerun or calibration")
    result["environment"] = dict(python=platform.python_version(), platform=platform.platform(),
                                 opencv=cv2.__version__, numpy=np.__version__,
                                 pyarrow=importlib.metadata.version("pyarrow"),
                                 markdown=importlib.metadata.version("markdown"))
    result["original_environment"] = json.loads((ROOT / "outputs/ego_motion/environment.json").read_text())
    result["review_environment"] = dict(python=review["test_rerun"]["python"], **review["runtime"])
    result["review_pytest_log"] = classify_pytest_log((args.review_dir / "pytest_review.txt").read_text())
    result["review_pytest_breakdown_matches"] = result["review_pytest_log"]["failure_breakdown"] == review["test_rerun"]["failure_breakdown"]

    clips = pd.read_csv(ROOT / "outputs/ego_motion/clip_scores.csv", dtype={"video_id": str})
    summary = [dict(split=split, model=model, **metrics(group))
               for (split, model), group in clips.groupby(["split", "model"])]
    expected = {(v["split"], v["model"]): v for v in review["clip_metrics"]}
    deltas = [abs(v[key] - expected[v["split"], v["model"]][key])
              for v in summary for key in ("n", "AUROC", "AP", "FPR", "TPR")]
    result["clip_metrics"] = summary
    result["clip_metric_max_abs_difference_from_review"] = max(deltas)
    result["clip_rows"] = len(clips)
    result["unique_clips"] = int(clips.video_id.nunique())

    frames = {}
    total_rows = 0
    for name in ("v0", "ego_motion"):
        for path in sorted((ROOT / "outputs" / name).rglob("*.parquet")):
            frame = pd.read_parquet(path, engine="pyarrow")
            if not np.isfinite(frame.select_dtypes(include="number")).all().all():
                raise ValueError(f"Nonfinite numeric data: {path}")
            if not (frame.t.diff().dropna() > 0).all():
                raise ValueError(f"Nonincreasing time: {path}")
            frames[str(path.relative_to(ROOT))] = frame
            total_rows += len(frame)
    result["parquet"] = dict(files=len(frames), rows=total_rows, engine="pyarrow",
                             files_matches_review=len(frames) == review["frame_files_decoded"],
                             rows_matches_review=total_rows == review["frame_rows"])
    peak_errors = []
    compared = 0
    timing = []
    timing_expected = {v["id"]: v for v in review["timing_checks"]}
    affine_records = []
    for _, row in clips.iterrows():
        key = f"outputs/ego_motion/frames/{row.model}/{row.video_id}.parquet"
        if key not in frames:
            continue
        frame = frames[key]
        eligible = eligible_frames(frame, row)
        peak = eligible.loc[eligible.S.idxmax()]
        compared += 1
        if abs(float(peak.S) - row.s_peak) > 1e-10 or abs(float(peak.t) - row.t_peak) > 1e-10:
            peak_errors.append(dict(video_id=row.video_id, model=row.model))
        if row.model == "raw_flow_no_blob":
            dt = np.diff(frame.t.to_numpy())
            timing.append(dict(id=row.video_id, fps=float(row.fps), min_dt=float(dt.min()),
                               max_dt=float(dt.max()), dt_ratio=float(dt.max() / dt.min()),
                               n_intervals=len(dt)))
        if row.model == "affine_residual_no_blob":
            affine_records.append(peak_fit_record(frame, row))
    result["peak_checks"] = dict(rows=compared, errors=peak_errors)
    result["timing_checks"] = timing
    dt_delta = max(abs(v[k] - timing_expected[v["id"]][k]) for v in timing
                   for k in ("fps", "min_dt", "max_dt", "dt_ratio", "n_intervals"))
    result["timing_summary"] = dict(files=len(timing), intervals=sum(v["n_intervals"] for v in timing),
                                     minimum_dt=min(v["min_dt"] for v in timing),
                                     maximum_dt=max(v["max_dt"] for v in timing),
                                     clips_with_dt_ratio_over_1_01=sum(v["dt_ratio"] > 1.01 for v in timing),
                                     maximum_difference_from_review=dt_delta,
                                     scope="Retained positive and false-positive clips only, not all 200 clips")
    threshold = yaml.safe_load((ROOT / "outputs/v0/threshold.yaml").read_text())
    config = yaml.safe_load((ROOT / "outputs/v0/config.yaml").read_text())
    norms = threshold["normalization"]
    original_frames = [v for k, v in frames.items() if k.startswith("outputs/v0/frames/")]
    errors = [np.max(np.abs(score_components(f, norms, V0Config(**config["v0"])) - f.S.to_numpy())) for f in original_frames]
    result["v0_component_scores"] = dict(files=len(errors), max_abs_error=float(max(errors)))

    fit = pd.DataFrame(affine_records)
    summaries = []
    for split, group in fit[fit.label.eq(1)].groupby("split"):
        for state, part in group.groupby("peak_fit_valid"):
            summaries.append(dict(split=split, peak_fit_valid=bool(state), positives=len(part),
                                  missed_positives=int((~part.predicted_positive).sum()),
                                  missed_positive_fraction=float((~part.predicted_positive).mean()),
                                  peak_reasons=dict(Counter(part.peak_fit_reason))))
    result["positive_peak_fit_state_summary"] = summaries
    result["affine_retained_clip_records"] = affine_records
    confirmation_positive = fit[fit.split.eq("confirmation") & fit.label.eq(1)]
    rejection_totals = []
    for missed, part in confirmation_positive.groupby(~confirmation_positive.predicted_positive):
        rejection_totals.append(dict(missed_positive=bool(missed), clips=len(part),
                                    rejected_frames=int(part.rejected_frames.sum()), total_frames=int(part.n_frames.sum()),
                                    mean_clip_rejected_fraction=float(part.rejected_frame_fraction.mean())))
    result["confirmation_positive_rejection_summary"] = rejection_totals
    result["fallback_interpretation"] = "Peak fit state and per-frame rejection are descriptive associations; EMA and original score threshold remain. All positive traces retained. Correct-negative traces not retained, so no whole-negative frame-state rates are estimated."
    result["synthetic"] = synthetic_audit()
    result["synthetic"]["review_score_max_abs_difference"] = max(
        abs(v - review["synthetic_translation_speed_audit"][k])
        for k, v in result["synthetic"]["stimulus_scores"].items())
    result["synthetic"]["review_uniform_max_abs_difference"] = max(
        abs(a[k] - b[k]) for a, b in zip(result["synthetic"]["analytic_uniform_translation"], review["analytic_uniform_translation"])
        for k in ("S_div", "S_rad"))
    archive_path = ROOT / "exports/fly_ttc_review.zip"
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            manifest = json.loads(archive.read("fly_ttc/EXPORT_MANIFEST.json"))
            failures = []
            parsed = Counter()
            for entry in manifest["files"]:
                content = archive.read(entry["name"])
                if len(content) != entry["size"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                    failures.append(entry["name"])
                extension = Path(entry["name"]).suffix
                if extension == ".json":
                    json.loads(content)
                    parsed["json"] += 1
                elif extension == ".csv":
                    pd.read_csv(BytesIO(content))
                    parsed["csv"] += 1
                elif extension == ".py":
                    ast.parse(content)
                    parsed["python"] += 1
            result["existing_review_archive"] = dict(sha256=sha(archive_path), files=len(archive.namelist()),
                                                     payload_files=len(manifest["files"]), hash_size_errors=failures,
                                                     parsed_payloads=dict(parsed))
    from fly_ttc.experiments.ego_motion import get_config, verify_frozen
    frozen_config, root = get_config(ROOT / "configs/default.yaml")
    verify_frozen(frozen_config, root, json.loads((ROOT / "outputs/ego_motion/protocol.json").read_text()))
    changed = [path for path, expected_sha in protected.items() if sha(ROOT / path) != expected_sha]
    result["preservation"] = dict(frozen_protocol_verified=True, protected_files=len(protected), changed_files=changed)
    if changed:
        raise ValueError(f"Protected artifacts changed: {changed}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    pd.DataFrame(affine_records).to_csv(args.out / "affine_peak_fit_states.csv", index=False)
    print(json.dumps({k: result[k] for k in ("clip_rows", "unique_clips", "clip_metric_max_abs_difference_from_review", "parquet", "timing_summary", "positive_peak_fit_state_summary", "confirmation_positive_rejection_summary", "synthetic", "preservation")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
