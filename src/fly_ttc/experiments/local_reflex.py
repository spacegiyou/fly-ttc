"""Finite synthetic Stage A experiment; no road labels, downloads or training."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import itertools
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.integrate import trapezoid
from tqdm import tqdm
import yaml

from fly_ttc.models.local_reflex_proxy_not_zhao_star import LocalReflexConfig, TimestampEMD, pool_directional_motion
from fly_ttc.models.thresholds import choose_threshold
from fly_ttc.models.timed_flow import TimedFlowConfig, TimedFlowScorer, continuous_ema_alpha
from fly_ttc.models.v0_expansion import ExpansionScorer
from fly_ttc.synthetic.reflex_stimuli import StimulusSpec, render_frame, timestamps, truth


EXPANDING = {"loom", "linear_expand", "loom_on_background"}
MODEL_NAME = "proxy_not_zhao_star"
CODE_FILES = [
    "src/fly_ttc/experiments/local_reflex.py", "src/fly_ttc/synthetic/reflex_stimuli.py",
    "src/fly_ttc/models/local_reflex_proxy_not_zhao_star.py", "src/fly_ttc/models/timed_flow.py",
    "src/fly_ttc/models/v0_expansion.py", "src/fly_ttc/models/thresholds.py",
    "src/fly_ttc/preprocess/expansion.py", "src/fly_ttc/preprocess/flow.py",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def build_cases(config):
    """Declared sweeps, not a claimed random sample or a full factorial design."""
    rows = []
    base = dict(config["stimulus"])
    base["image_size"] = tuple(base["image_size"])
    matrix = config["matrix"]

    def add(split, suite, kind, schedule, **changes):
        values = {**base, **changes}
        # Match the declared texture and disk contrast amplitudes, but do not
        # imply equal edge counts, moving areas or total motion energy.
        values["texture_contrast"] = values.get("contrast", 0.8)
        if kind == "loom_on_background" and "background_speed_px_s" not in values:
            z = values["initial_depth"] - values["approach_speed"] * values["duration_s"] / 2
            values["background_speed_px_s"] = values["focal_length_px"] * values["object_radius"] * values["approach_speed"] / z**2
        spec = StimulusSpec(kind=kind, **values)
        rows.append(dict(case_id=f"{split}_{len(rows):04d}", split=split, suite=suite,
                         schedule=schedule, label=int(kind in EXPANDING), spec=asdict(spec)))

    cal = config["calibration"]
    for kind, polarity in itertools.product(matrix["kinds"], matrix["polarities"]):
        add("calibration", "calibration", kind, cal["schedule"], contrast=cal["contrast"],
            approach_speed=cal["approach_speed"], polarity=polarity)
    for kind, polarity, contrast, schedule in itertools.product(matrix["kinds"], matrix["polarities"], matrix["contrasts"], matrix["schedules"]):
        add("challenge", "core", kind, schedule, contrast=contrast, polarity=polarity)
    for kind, center, polarity, schedule in itertools.product(["loom", "recede", "translate"], matrix["offcenter_positions"], matrix["polarities"], matrix["schedules"]):
        add("challenge", "position", kind, schedule, center_fraction=tuple(center), polarity=polarity)
    for kind, speed, polarity in itertools.product(["loom", "recede", "translate", "background_translation", "rotation"], matrix["additional_approach_speeds"], matrix["polarities"]):
        add("challenge", "approach_speed", kind, "30hz", approach_speed=speed, polarity=polarity)
    for kind, scale, polarity, schedule in itertools.product(["translate", "background_translation", "rotation"], matrix["nuisance_speed_scales"], matrix["polarities"], ["15hz", "30hz"]):
        add("challenge", "nuisance_speed", kind, schedule, motion_scale=scale, polarity=polarity)
    for scale, direction, polarity, schedule in itertools.product(matrix["nuisance_speed_scales"], [0.0, np.pi/2], matrix["polarities"], ["15hz", "30hz"]):
        ref = truth(StimulusSpec(kind="loom", **base), base["duration_s"]/2)
        add("challenge", "background", "loom_on_background", schedule, direction_rad=direction,
            background_speed_px_s=scale*ref["reference_boundary_speed_px_s"], polarity=polarity)
    for direction, scale, schedule in itertools.product(matrix["translation_directions_rad"],matrix["grating_speed_scales"],["15hz","30hz"]):
        add("challenge","direction_grating","background_translation",schedule,
            texture_kind="grating",direction_rad=direction,motion_scale=scale,polarity=-1)
    for direction, scale, schedule in itertools.product(matrix["translation_directions_rad"][1:],[1.0,4.0],["15hz","30hz"]):
        add("challenge","direction_disk","translate",schedule,direction_rad=direction,motion_scale=scale,polarity=-1)
    return rows


def permutations(config):
    possibilities = [p for p in itertools.permutations(range(4)) if p != (0,1,2,3)]
    rng = np.random.default_rng(config["seed"])
    selected = rng.choice(len(possibilities), config["controls"]["n_direction_permutations"], replace=False)
    return {f"direction_permutation_{i}": possibilities[index] for i,index in enumerate(selected)}


def prepare(config_path):
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text())
    out = root / config["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        verify(root, config_path, protocol)
        return config, root, out, protocol
    tests = ["tests/test_local_reflex_proxy_not_zhao_star.py", "tests/test_reflex_stimuli.py", "tests/test_timed_flow.py", "tests/test_local_reflex_experiment.py"]
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], cwd=root, capture_output=True, text=True)
    (out / "pre_run_tests.txt").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError("Synthetic contracts failed; see pre_run_tests.txt")
    cases = build_cases(config)
    write_json(out / "cases.json", cases)
    protocol = dict(created_utc=utc(), config_sha256=digest(config_path),
                    code_sha256={p:digest(root/p) for p in CODE_FILES},
                    cases_sha256=digest(out/"cases.json"), n_cases=len(cases),
                    direction_permutations=permutations(config),
                    controls_budget="The seven EMD readouts have zero learned coefficients and share the same EMD state/RF support. Each model gets an independent cutoff from the same 12 synthetic negative calibration cases. The two optical-flow models are unmatched diagnostic comparators, not equal-compute baselines. Permutations are engineering controls, not connectome ablations, and no retraining comparison was performed.",
                    scope="Synthetic regression and diagnostic matrix; stimulus variants are correlated. No accident labels, random-population confidence intervals, external validation or biological advantage claim.")
    write_json(protocol_path, protocol)
    (out/"config.yaml").write_text(config_path.read_text())
    return config, root, out, protocol


def verify(root, config_path, protocol):
    if digest(config_path) != protocol["config_sha256"]:
        raise ValueError("Configuration changed after matrix freeze")
    for p, expected in protocol["code_sha256"].items():
        if digest(root/p) != expected:
            raise ValueError(f"Scoring code changed after matrix freeze: {p}")
    config = yaml.safe_load(Path(config_path).read_text())
    if digest(root/config["output_dir"]/"cases.json") != protocol["cases_sha256"]:
        raise ValueError("Frozen stimulus matrix changed")


def score_case(case, config, direction_controls):
    values = dict(case["spec"])
    for key in ("image_size", "center_fraction"):
        values[key] = tuple(values[key])
    spec = StimulusSpec(**values)
    model_config = LocalReflexConfig(**config["model"])
    emd = TimestampEMD(model_config)
    readouts = {MODEL_NAME:model_config, **{name:replace(model_config, direction_permutation=p) for name,p in direction_controls.items()}}
    states = {name:np.zeros((model_config.grid,)*2) for name in [*readouts, "simple_motion_energy"]}
    timed = TimedFlowScorer(TimedFlowConfig(**config["timed_flow"]))
    legacy = ExpansionScorer()
    traces, peak_maps = [], {}
    peak_values = {name:-np.inf for name in states}
    for t in timestamps(spec.duration_s, case["schedule"]):
        gray = render_frame(spec, float(t))
        frame = emd.update(gray, float(t))
        geom = truth(spec, float(t))
        outputs, centers = {}, None
        for name, cfg in readouts.items():
            pooled = pool_directional_motion(frame["directional_motion"], cfg, gray.shape)
            outputs[name] = pooled["activation_map"]
            centers = pooled["unit_centers_xy"]
            if name == MODEL_NAME:
                outputs["simple_motion_energy"] = pooled["simple_energy_map"]
        if frame["dt_s"] > 0:
            alpha = continuous_ema_alpha(frame["dt_s"], model_config.output_tau_s)
            for name, values in outputs.items():
                states[name] += alpha * (values - states[name])
        scores = {name:float(values.max()) for name,values in states.items()}
        scores["timed_global_flow"] = timed.update(gray, float(t))["S"]
        scores["v0_legacy_unscaled"] = legacy.update(gray)["S"]
        for name, score in scores.items():
            x = y = error = np.nan
            if name in states:
                y_index, x_index = np.unravel_index(np.argmax(states[name]), states[name].shape)
                if score > 0:
                    x, y = centers[y_index, x_index]
                    if geom["center_xy_px"] is not None:
                        error = float(np.linalg.norm(np.array([x,y]) - geom["center_xy_px"]))
                if t >= config["assessment"]["warmup_s"] and score > peak_values[name]:
                    peak_maps[name] = states[name].copy()
                    peak_values[name] = score
            traces.append(dict(case_id=case["case_id"], model=name, t_s=float(t), dt_s=frame["dt_s"],
                               S=score, peak_x_px=x, peak_y_px=y, localization_error_px=error,
                               radius_px=geom["radius_px"], true_ttc_s=geom["true_ttc_s"],
                               disk_fully_visible=geom["disk_fully_visible"]))
    frame = pd.DataFrame(traces)
    selected = frame.loc[frame.t_s >= config["assessment"]["warmup_s"]]
    rows = []
    for name, values in selected.groupby("model", sort=False):
        peak = values.loc[values.S.idxmax()]
        full = frame.loc[frame.model.eq(name)]
        start = config["assessment"]["warmup_s"]
        interior = full.loc[full.t_s.gt(start) & full.t_s.lt(spec.duration_s)]
        integral_t = np.r_[start, interior.t_s, spec.duration_s]
        integral_s = np.interp(integral_t, full.t_s, full.S)
        rows.append(dict(case_id=case["case_id"], split=case["split"], suite=case["suite"],
                         kind=spec.kind, schedule=case["schedule"], contrast=spec.contrast, polarity=spec.polarity,
                         center_x=spec.center_fraction[0], center_y=spec.center_fraction[1],
                         label=case["label"], model=name, s_peak=float(peak.S), t_peak_s=float(peak.t_s),
                         response_integral=float(trapezoid(integral_s, integral_t)),
                         localization_error_px=float(peak.localization_error_px), n_frames=len(values),
                         first_observed_analysis_t_s=float(values.t_s.min()), last_observed_analysis_t_s=float(values.t_s.max()),
                         integral_start_s=start, integral_end_s=spec.duration_s,
                         disk_fully_visible_fraction=float(values.disk_fully_visible.dropna().mean()) if values.disk_fully_visible.notna().any() else np.nan,
                         n_total_frames=len(frame.loc[frame.model.eq(name)])))
    return rows, frame, peak_maps, centers


def calculate_metrics(scores, config):
    records = []
    for (split, model), group in scores.groupby(["split", "model"]):
        positive, negative = group.loc[group.label.eq(1)], group.loc[group.label.eq(0)]
        records.append(dict(split=split,model=model,n=len(group),n_positive=len(positive),n_negative=len(negative),
                            auroc=float(roc_auc_score(group.label,group.s_peak)),
                            ap=float(average_precision_score(group.label,group.s_peak)),
                            expansion_recall=float(positive.detected.mean()), nuisance_fpr=float(negative.detected.mean())))
    identity = scores.loc[scores.split.eq("challenge") & scores.model.eq(MODEL_NAME)]
    by_suite_schedule = []
    for (suite, schedule), group in identity.groupby(["suite", "schedule"]):
        pos, neg = group.loc[group.label.eq(1)], group.loc[group.label.eq(0)]
        by_suite_schedule.append(dict(suite=suite, schedule=schedule, n_positive=len(pos),n_negative=len(neg),
            recall=float(pos.detected.mean()) if len(pos) else None,
            fpr=float(neg.detected.mean()) if len(neg) else None))
    by_schedule = []
    for schedule, group in identity.loc[identity.suite.eq("core")].groupby("schedule"):
        by_schedule.append(dict(schedule=schedule,
                                recall=float(group.loc[group.label.eq(1),"detected"].mean()),
                                fpr=float(group.loc[group.label.eq(0),"detected"].mean())))
    local = identity.loc[identity.suite.eq("position") & identity.kind.eq("loom")]
    # Localization coverage includes every requested loom, including zero responses/misses.
    tolerance = (config["stimulus"]["image_size"][1]-1-config["model"]["rf_size_px"])/(config["model"]["grid"]-1)
    localized = local.detected & local.localization_error_px.le(tolerance)
    core = identity.loc[identity.suite.eq("core")].copy()
    reference = core.loc[core.schedule.eq("60hz"),["kind","contrast","polarity","s_peak"]].rename(columns={"s_peak":"reference_peak"})
    compared = core.loc[~core.schedule.eq("60hz")].merge(reference,on=["kind","contrast","polarity"],validate="many_to_one")
    # Active 60 Hz stimuli only. Near-zero ratios have no meaningful relative scale.
    active_floor = max(float(reference.reference_peak.max())*1e-6,1e-12)
    active = compared.loc[compared.reference_peak > active_floor].copy()
    active["relative_error"] = abs(active.s_peak-active.reference_peak)/active.reference_peak
    sampling = float(active.relative_error.median()) if len(active) else None
    near_zero = compared.loc[compared.reference_peak <= active_floor]
    criteria = config["assessment"]
    gates = dict(
        all_schedules_recall=all(r["recall"]>=criteria["minimum_expansion_recall"] for r in by_schedule),
        all_schedules_nuisance_fpr=all(r["fpr"]<=criteria["maximum_nuisance_false_positive_rate"] for r in by_schedule),
        all_suites_schedules_recall=all(r["recall"] is None or r["recall"]>=criteria["minimum_expansion_recall"] for r in by_suite_schedule),
        all_suites_schedules_nuisance_fpr=all(r["fpr"] is None or r["fpr"]<=criteria["maximum_nuisance_false_positive_rate"] for r in by_suite_schedule),
        localized_detection=float(localized.mean())>=criteria["minimum_localized_detection_fraction"],
        median_sampling_stability=sampling is not None and sampling<=criteria["maximum_median_sampling_relative_error"],
    )
    return dict(results=records, identity_core_by_schedule=by_schedule, identity_by_suite_schedule=by_suite_schedule,
                localization=dict(n_requested=len(local),n_detected_and_localized=int(localized.sum()),
                                  fraction=float(localized.mean()), tolerance_px=tolerance),
                sampling=dict(reference="60hz",n_pairs=len(active),excluded_near_zero_pairs=len(compared)-len(active),
                              median_relative_peak_error=sampling, active_reference_floor=active_floor,
                              p90_relative_peak_error=float(active.relative_error.quantile(.9)) if len(active) else None,
                              max_relative_peak_error=float(active.relative_error.max()) if len(active) else None,
                              near_zero_reference_other_rate_alerts=int(near_zero.detected.sum()),
                              near_zero_reference_other_rate_max_peak=float(near_zero.s_peak.max()) if len(near_zero) else None),
                gates=gates, advance_to_roi=all(gates.values()),
                interpretation="Engineering advancement gates on a correlated synthetic diagnostic matrix. No biological wiring benefit or accident performance inference.")


def run(config_path):
    config,root,out,protocol = prepare(config_path)
    if (out/"run_state.json").exists():
        raise ValueError("Run already started; preserve its artifacts and use a separate output directory for another experiment")
    write_json(out/"run_state.json",dict(status="running",started_utc=utc()))
    cases = json.loads((out/"cases.json").read_text())
    all_rows,all_traces,cutoffs = [],[],{}
    sample_maps = {}
    started = time.perf_counter()
    # Calibration is scored and persisted before any challenge response is computed.
    for split in ["calibration","challenge"]:
        if split == "challenge":
            write_json(out/"challenge_started.json",dict(created_utc=utc(),threshold_sha256=digest(out/"thresholds.json")))
        for case in tqdm([c for c in cases if c["split"]==split],desc=split):
            rows,frame,maps,centers = score_case(case,config,protocol["direction_permutations"])
            all_rows.extend(rows)
            all_traces.append(frame)
            if case["suite"] == "core" and case["schedule"] == "30hz" and case["spec"]["contrast"] == 0.8 and case["spec"]["polarity"] == -1:
                sample_maps[case["case_id"]] = dict(kind=case["spec"]["kind"],map=maps[MODEL_NAME].tolist(),centers=centers.tolist())
        if split=="calibration":
            scores=pd.DataFrame(all_rows)
            for model,group in scores.loc[scores.label.eq(0)].groupby("model"):
                cutoffs[model]={**choose_threshold(group.s_peak,config["calibration"]["false_positive_target"]),
                                "calibration_case_ids":group.case_id.tolist()}
            write_json(out/"thresholds.json",cutoffs)
    scores=pd.DataFrame(all_rows)
    scores["theta"]=scores.model.map({name:value["theta"] for name,value in cutoffs.items()})
    scores["detected"]=scores.s_peak > scores.theta
    traces=pd.concat(all_traces,ignore_index=True)
    scores.to_csv(out/"case_scores.csv",index=False)
    traces.to_parquet(out/"traces.parquet",index=False)
    write_json(out/"sample_maps.json",sample_maps)
    write_json(out/"metrics.json",calculate_metrics(scores,config))
    verify(root,Path(config_path),protocol)
    write_json(out/"environment.json",dict(python=sys.version,platform=platform.platform(),
                 packages={p:version(p) for p in ["numpy","opencv-python-headless","pandas","pyarrow","scipy","scikit-learn"]},
                 measured_wall_s=time.perf_counter()-started,
                 timing_scope="Entire sequential synthetic runner including rendering, all nine readouts, optical flow, I/O. Not latency/power per deployed model."))
    write_json(out/"run_state.json",dict(status="complete",completed_utc=utc(),n_cases=len(cases),n_models=scores.model.nunique(),n_rows=len(scores)))
    from fly_ttc.viz.local_reflex_report import make_report
    make_report(out)
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/local_reflex.yaml")
    parser.add_argument("--phase",choices=["prepare","run","report"],default="run")
    args=parser.parse_args(argv)
    if args.phase=="run":
        print(run(args.config))
    elif args.phase=="prepare":
        print(prepare(args.config)[2])
    else:
        from fly_ttc.viz.local_reflex_report import make_report
        path=Path(args.config).resolve()
        config=yaml.safe_load(path.read_text())
        print(make_report(path.parent.parent/config["output_dir"]))


if __name__ == "__main__":
    main()
