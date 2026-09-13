"""Frozen four-arm experiment, retaining original calibration and fresh clips."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shlex
import subprocess
import sys
import warnings

from huggingface_hub import snapshot_download
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from fly_ttc.cli import _model_config, _write_json, extract_clip
from fly_ttc.data.nexar import NEXAR_REPO_ID, _integrity, load_manifest, validate_manifest, write_manifest
from fly_ttc.data.sampling import sample_subset
from fly_ttc.eval.lead_time import lead_times
from fly_ttc.models.ego_variants import MODELS, EgoVariantScorer, model_components, model_config
from fly_ttc.models.thresholds import choose_threshold
from fly_ttc.models.v0_expansion import fit_normalization, score_components
from fly_ttc.paths import load_config, resolve_path
from fly_ttc.preprocess.ego_motion import EgoMotionConfig

LOG = logging.getLogger("fly_ttc.experiment")
SCORING_FILES = (
    "src/fly_ttc/cli.py", "src/fly_ttc/paths.py", "src/fly_ttc/data/nexar.py",
    "src/fly_ttc/data/sampling.py", "src/fly_ttc/preprocess/video.py",
    "src/fly_ttc/preprocess/flow.py", "src/fly_ttc/preprocess/expansion.py",
    "src/fly_ttc/preprocess/ego_motion.py", "src/fly_ttc/models/v0_expansion.py",
    "src/fly_ttc/models/ego_variants.py", "src/fly_ttc/models/thresholds.py",
    "src/fly_ttc/eval/lead_time.py", "src/fly_ttc/eval/metrics.py",
    "src/fly_ttc/eval/comparison.py", "src/fly_ttc/experiments/ego_motion.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def manifest_semantics(df):
    """Freeze annotations/provenance, excluding mutable download integrity fields.

    Twelve significant digits exceed Nexar's timestamp precision and tolerate
    harmless CSV binary-float round trips. IDs, labels and categories stay exact.
    """
    columns = ["video_id", "label", "time_of_alert", "time_of_event", "scene", "weather",
               "light_conditions", "source_repo", "source_revision", "source_path", "source_sha256"]
    rows = []
    for _, row in df.sort_values("video_id").iterrows():
        rows.append({key: None if pd.isna(row[key]) else format(float(row[key]), ".12g")
                     if key in ("time_of_alert", "time_of_event") else str(row[key]) for key in columns})
    return config_digest(rows)


def get_config(path):
    config = load_config(path)
    path = Path(path).resolve()
    root = path.parent.parent if path.parent.name == "configs" else Path.cwd()
    for key in ("baseline_run", "output_dir", "source_manifest", "confirmation_manifest"):
        config["followup"][key] = str(resolve_path(config["followup"][key], root))
    if tuple(config["followup"]["models"]) != MODELS:
        raise ValueError("This experiment requires all four prespecified models")
    if config["ego_motion"]["method"] != "affine":
        raise ValueError("Primary frozen comparison requires affine compensation")
    if config["v0"]["score"] != "combined":
        raise ValueError("Original v0 comparison requires its original combined score")
    return config, root


def select_confirmation(source, original_ids, n_pos, n_neg, seed):
    remaining = source.loc[~source.video_id.astype(str).isin(set(map(str, original_ids)))].copy()
    selected = sample_subset(remaining, n_pos, n_neg, seed)
    if set(selected.video_id.astype(str)) & set(map(str, original_ids)):
        raise ValueError("Confirmation overlaps original clips")
    return selected


def verify_frozen(config, root, protocol):
    if config_digest(config) != protocol["config_sha256"]:
        raise ValueError("Configuration changed after freeze; do not retune this confirmation set")
    for path, expected in protocol["scoring_file_sha256"].items():
        if digest(root / path) != expected:
            raise ValueError(f"Scoring code changed after freeze: {path}")
    baseline = Path(config["followup"]["baseline_run"])
    for path, expected in protocol["baseline_file_sha256"].items():
        if digest(baseline / path) != expected:
            raise ValueError(f"Original v0 artifact changed: {path}")
    if "source_manifest_sha256" in protocol:
        if digest(config["followup"]["source_manifest"]) != protocol["source_manifest_sha256"]:
            raise ValueError("Pinned source manifest changed after freeze")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Missing .*video")
            selected = load_manifest(config["followup"]["confirmation_manifest"])
        if manifest_semantics(selected) != protocol["confirmation_semantics_sha256"]:
            raise ValueError("Confirmation annotations or source provenance changed after freeze")


def prepare(config_path):
    config, root = get_config(config_path)
    follow = config["followup"]
    out, baseline = Path(follow["output_dir"]), Path(follow["baseline_run"])
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        verify_frozen(config, root, protocol)
        return protocol
    # These tests exercise actual Farneback stimuli before any new video score.
    command = [sys.executable, "-m", "pytest", "tests/test_expansion.py", "tests/test_ego_motion.py",
               "tests/test_ego_variants.py", "-q"]
    tested = subprocess.run(command, cwd=root, capture_output=True, text=True)
    (out / "synthetic_gate.txt").write_text(tested.stdout + tested.stderr)
    if tested.returncode:
        raise ValueError("Synthetic gate failed; inspect synthetic_gate.txt before real-video scoring")
    original = load_manifest(baseline / "manifest.csv")
    scores = pd.read_csv(baseline / "clip_scores.csv", dtype={"video_id": str})
    previous = yaml.safe_load((baseline / "threshold.yaml").read_text())
    previous_config = yaml.safe_load((baseline / "config.yaml").read_text())
    for key in ("video", "flow", "v0", "eval"):
        if config[key] != previous_config[key]:
            raise ValueError(f"Original analysis settings must remain unchanged: {key}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Missing .*video")
        source = load_manifest(follow["source_manifest"])
    selected = select_confirmation(source, original.video_id, follow["confirmation_n_pos"],
                                   follow["confirmation_n_neg"], follow["confirmation_seed"])
    if selected.source_repo.nunique() != 1 or selected.source_repo.iloc[0] != NEXAR_REPO_ID:
        raise ValueError("Only the verified Nexar train source is supported")
    if selected.source_revision.nunique() != 1 or selected.source_revision.iloc[0] != original.source_revision.iloc[0]:
        raise ValueError("Confirmation must use the same pinned source revision")
    if set(selected.source_sha256.dropna()) & set(original.source_sha256.dropna()):
        raise ValueError("Confirmation has a duplicate source file checksum")
    write_manifest(selected, follow["confirmation_manifest"])
    validation_ids = list(map(str, previous["validation_ids"]))
    exploratory_ids = list(map(str, previous["evaluation_ids"]))
    if set(scores.video_id) != set(validation_ids + exploratory_ids):
        raise ValueError("Original split/manifest mismatch")
    frozen_files = {str(p.relative_to(baseline)): digest(p) for p in sorted(baseline.rglob("*")) if p.is_file()}
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": config_digest(config),
        "scoring_file_sha256": {path: digest(root / path) for path in SCORING_FILES},
        "baseline_file_sha256": frozen_files, "source_revision": str(selected.source_revision.iloc[0]),
        "source_manifest_sha256": digest(follow["source_manifest"]),
        "models": list(MODELS), "primary_contrast": "affine_residual_no_blob minus raw_flow_no_blob",
        "primary_endpoint": "paired clip-stratified bootstrap delta AUROC on confirmation",
        "secondary_endpoints": ["paired delta FPR", "paired delta TPR", "alert/early hits", "peak leads"],
        "validation_ids": validation_ids, "exploratory_ids": exploratory_ids,
        "confirmation_ids": selected.video_id.astype(str).tolist(),
        "confirmation_source_sha256": dict(zip(selected.video_id.astype(str), selected.source_sha256)),
        "confirmation_semantics_sha256": manifest_semantics(selected),
        "negative_calibration_ids": original.loc[original.video_id.isin(validation_ids) & original.label.eq(0), "video_id"].tolist(),
        "confirmation_selection": {"n_pos": follow["confirmation_n_pos"], "n_neg": follow["confirmation_n_neg"],
                                   "seed": follow["confirmation_seed"], "exclude": "all original 100 IDs and identical source SHA256"},
        "synthetic_gate": {"command": shlex.join(command), "exit_code": tested.returncode},
        "interpretation": "Original 80 are exploratory reuse; new clips are clip-disjoint, not verified trip-disjoint. No confirmation-based tuning or winner selection.",
    }
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    _write_json(protocol_path, protocol)
    LOG.info("Frozen protocol, scoring code and %d fresh clip IDs: %s", len(selected), protocol_path)
    return protocol


def download_confirmation(config_path):
    config, root = get_config(config_path)
    protocol = prepare(config_path)
    follow = config["followup"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Missing .*video")
        selected = load_manifest(follow["confirmation_manifest"])
    if selected.video_id.astype(str).tolist() != protocol["confirmation_ids"]:
        raise ValueError("Frozen confirmation IDs changed")
    for row in selected.itertuples():
        if row.source_sha256 != protocol["confirmation_source_sha256"][row.video_id]:
            raise ValueError("Frozen source checksums changed")
    try:
        snapshot_download(NEXAR_REPO_ID, repo_type="dataset", revision=protocol["source_revision"],
                          token=os.environ.get("HF_TOKEN") or False, local_dir=config["paths"]["raw_dir"],
                          allow_patterns=selected.source_path.tolist(), max_workers=4)
    finally:
        write_manifest(_integrity(selected), follow["confirmation_manifest"])
    verify_frozen(config, root, protocol)
    return Path(follow["confirmation_manifest"])


def calibrate(cache, original, protocol, config, original_threshold):
    """Use only declared old negative validation frames for every new arm."""
    neg_ids = protocol["negative_calibration_ids"]
    observed = original.set_index("video_id").loc[neg_ids]
    if not observed.label.eq(0).all() or not set(neg_ids).issubset(protocol["validation_ids"]):
        raise ValueError("Calibration IDs are not original negative validation clips")
    if any(i not in cache for i in neg_ids):
        raise ValueError("All frozen negative calibration clips must decode; no replacement IDs")
    thresholds = {}
    base_cfg = _model_config(config)
    for model in MODELS:
        if model == "v0_original":
            thresholds[model] = original_threshold
            continue
        cfg = model_config(base_cfg, model)
        components = [model_components(cache[i], model) for i in neg_ids]
        norm = fit_normalization(components, cfg)
        norm.update(video_ids=neg_ids, source="original_negative_validation")
        maxima = [float(score_components(frame, norm, cfg).max()) for frame in components]
        thresholds[model] = {**choose_threshold(maxima, config["eval"]["fpr_target"]),
                             "normalization": norm, "validation_ids": protocol["validation_ids"],
                             "negative_calibration_ids": neg_ids}
    return thresholds


def run_experiment(config_path):
    config, root = get_config(config_path)
    follow = config["followup"]
    out, baseline = Path(follow["output_dir"]), Path(follow["baseline_run"])
    protocol = prepare(config_path)
    verify_frozen(config, root, protocol)
    if (out / "metrics.json").exists():
        raise ValueError("Completed frozen experiment already exists; use --phase report to regenerate its report")
    original = load_manifest(baseline / "manifest.csv")
    original = _integrity(original)
    if original.sha256.isna().any():
        raise ValueError("Original calibration/comparison download is incomplete")
    confirmation = load_manifest(follow["confirmation_manifest"])
    if confirmation.video_id.tolist() != protocol["confirmation_ids"]:
        raise ValueError("Frozen confirmation manifest changed")
    confirmation = _integrity(confirmation)
    if confirmation.sha256.isna().any():
        raise ValueError("Confirmation download incomplete. Run --phase download; never replace missing IDs")
    for row in confirmation.itertuples():
        if row.sha256 != protocol["confirmation_source_sha256"][row.video_id]:
            raise ValueError("Confirmation file differs from frozen SHA256")
    original_threshold = yaml.safe_load((baseline / "threshold.yaml").read_text())
    manifest = pd.concat([original, confirmation], ignore_index=True)
    if manifest.video_id.duplicated().any():
        raise ValueError("Original/confirmation clip overlap")
    manifest["experiment_split"] = manifest.video_id.map(
        lambda i: "validation" if i in protocol["validation_ids"] else "exploratory" if i in protocol["exploratory_ids"] else "confirmation")
    manifest.to_csv(out / "manifest.csv", index=False)
    cache_dir = out / ".component_cache"
    cache_dir.mkdir(exist_ok=True)
    cache, infos, fit_quality = {}, {}, []
    protocol_hash = digest(out / "protocol.json")
    for phase in ("validation", "exploratory", "confirmation"):
        if phase == "confirmation":
            verify_frozen(config, root, protocol)
        phase_rows = manifest.loc[manifest.experiment_split.eq(phase)]
        for _, row in tqdm(phase_rows.iterrows(), total=len(phase_rows), desc=phase, unit="clip"):
            video_id = str(row.video_id)
            cache_identity = hashlib.sha256((protocol_hash + str(row.sha256)).encode()).hexdigest()
            # Cached components are private score rows, never original images.
            cached, info_path = cache_dir / f"{video_id}.parquet", cache_dir / f"{video_id}.json"
            if cached.exists() and info_path.exists():
                frames, info = pd.read_parquet(cached), json.loads(info_path.read_text())
                if info.get("component_sha256") != digest(cached):
                    raise ValueError(f"Component cache checksum mismatch: {video_id}")
                if info.get("cache_identity") != cache_identity:
                    raise ValueError(f"Component cache belongs to a different frozen protocol or video: {video_id}")
            else:
                scorer = EgoVariantScorer(_model_config(config), config["flow"], EgoMotionConfig(**config["ego_motion"]))
                frames, info, _ = extract_clip(row, config, scorer=scorer)
                if info["status"] == "ok":
                    frames.to_parquet(cached, index=False)
                    info["component_sha256"] = digest(cached)
                    info["cache_identity"] = cache_identity
                    _write_json(info_path, info)
            infos[video_id] = info
            if info["status"] == "ok":
                cache[video_id] = frames
                quality_frames = frames.loc[frames.t < row.time_of_event] if row.label else frames
                fit_quality.append(dict(video_id=video_id, split=phase, n_frames=len(quality_frames),
                                        fit_valid_fraction=float(quality_frames.fit_valid.mean()),
                                        inlier_fraction_mean=float(quality_frames.inlier_fraction.mean()),
                                        fit_error_px_median=float(quality_frames.fit_error_px.median())))
            tqdm.write(f"{video_id}: {info['status']} ({len(frames)} frame pairs)")
        if phase == "validation":
            thresholds = calibrate(cache, original, protocol, config, original_threshold)
            (out / "thresholds.yaml").write_text(yaml.safe_dump(thresholds, sort_keys=False))
            _write_json(out / "calibration_frozen.json", {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "thresholds_sha256": digest(out / "thresholds.yaml"),
                "before_scoring": ["exploratory", "confirmation"], "negative_ids": protocol["negative_calibration_ids"]})
    rows = []
    frames_root = out / "frames"
    for model in MODELS:
        (frames_root / model).mkdir(parents=True, exist_ok=True)
    for _, row in manifest.iterrows():
        video_id, info = str(row.video_id), infos[str(row.video_id)]
        for model in MODELS:
            threshold, cfg = thresholds[model], model_config(_model_config(config), model)
            result = dict(video_id=video_id, label=int(row.label), model=model, split=row.experiment_split,
                          scene=row.scene, light=row.light_conditions, weather=row.weather,
                          time_of_alert=row.time_of_alert, time_of_event=row.time_of_event,
                          theta_used=threshold["theta"], duration_s=info.get("duration_s"), fps=info.get("fps"),
                          status=info["status"], n_frames=0)
            if info["status"] == "ok":
                frame = model_components(cache[video_id], model)
                frame["S"] = score_components(frame, threshold["normalization"], cfg)
                frame["above_theta"] = frame.S > threshold["theta"]
                result.update(lead_times(frame.t.to_numpy(), frame.S.to_numpy(), threshold["theta"],
                                         row.time_of_alert, row.time_of_event,
                                         lead_windows_ms=config["eval"]["lead_windows_ms"],
                                         alert_window_pre_s=config["eval"]["alert_window_pre_s"]))
                result["n_frames"] = len(frame)
                if result["label"] or result.get("false_positive", False):
                    frame.to_parquet(frames_root / model / f"{video_id}.parquet", index=False)
            rows.append(result)
    clips = pd.DataFrame(rows)
    old = pd.read_csv(baseline / "clip_scores.csv", dtype={"video_id": str}).set_index("video_id")
    recreated = clips.loc[clips.model.eq("v0_original") & clips.video_id.isin(old.index)].set_index("video_id").loc[old.index]
    # Refuse an ablation comparison if the supposedly unchanged original moved.
    for column in ("s_peak", "t_peak", "theta_used", "lead_to_event_s"):
        if not np.allclose(recreated[column], old[column], equal_nan=True, atol=1e-8, rtol=1e-8):
            raise ValueError(f"Original v0 reproduction differs: {column}")
    verify_frozen(config, root, protocol)
    clips.to_csv(out / "clip_scores.csv", index=False)
    pd.DataFrame(fit_quality).to_csv(out / "fit_quality.csv", index=False)
    from fly_ttc.eval.comparison import summarize_comparison
    metrics = summarize_comparison(clips, config)
    _write_json(out / "audit.json", {"original_clips_reproduced": len(recreated), "baseline_unchanged": True,
                                    "confirmation_requested": len(confirmation),
                                    "confirmation_decoded": sum(i in cache for i in confirmation.video_id),
                                    "confirmation_sha256_verified": len(confirmation), "clip_id_overlap": 0,
                                    "calibration_ids": protocol["negative_calibration_ids"],
                                    "scoring_code_unchanged_since_freeze": True})
    # Diagnostic IDs were selected before new scores. Never select confirmation successes.
    diagnostics = out / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    for video_id in follow["diagnostic_ids"]:
        selected = manifest.loc[manifest.video_id.eq(str(video_id))]
        if selected.empty:
            continue
        row = selected.iloc[0]
        targets = ([float(row.time_of_alert), float(row.time_of_event) + config["report"]["snapshot_offsets_s"][-1]] if row.label else
                   [float(old.loc[str(video_id), "t_peak"])])
        scorer = EgoVariantScorer(_model_config(config), config["flow"], EgoMotionConfig(**config["ego_motion"]))
        _, _, snapshots = extract_clip(row, config, targets, scorer=scorer)
        if snapshots:
            np.savez_compressed(diagnostics / f"{video_id}.npz",
                                **{key: np.stack([s[key] for s in snapshots]) for key in snapshots[0]})
    # Normal negative rows need not remain at frame resolution after aggregation.
    import shutil
    shutil.rmtree(cache_dir)
    _write_json(out / "metrics.json", metrics)
    _write_json(out / "run_state.json", {"status": "complete", "models": list(MODELS), "n_clips": len(manifest)})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--phase", choices=["prepare", "download", "run", "report"], required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.phase == "prepare":
            prepare(args.config)
        elif args.phase == "download":
            download_confirmation(args.config)
        elif args.phase == "run":
            run_experiment(args.config)
        else:
            from fly_ttc.viz.ego_report import make_ego_report
            config, _ = get_config(args.config)
            make_ego_report(config["followup"]["output_dir"])
        return 0
    except Exception as exc:
        token = os.environ.get("HF_TOKEN")
        message = str(exc).replace(token, "[REDACTED]") if token else str(exc)
        command = shlex.join([sys.executable, "scripts/run_ego_ablation.py", *(argv if argv is not None else sys.argv[1:])])
        if token:
            command = command.replace(token, "[REDACTED]")
        LOG.error("%s: %s", type(exc).__name__, message)
        LOG.error("Stopped; no dataset substitution. Exact command: %s", command)
        return 1
