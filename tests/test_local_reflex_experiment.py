"""Matrix, calibration exclusion, and all-condition advancement contracts."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fly_ttc.experiments.local_reflex import MODEL_NAME, build_cases, calculate_metrics, permutations, score_case
from fly_ttc.synthetic.reflex_stimuli import render_frame, StimulusSpec


def configuration():
    return yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/local_reflex.yaml").read_text())


def test_matrix_is_deterministic_and_contrasts_change_actual_texture():
    config=configuration()
    cases=build_cases(config)
    assert cases==build_cases(config)
    assert len(cases)==354
    assert len({c["case_id"] for c in cases})==354
    calibration=[c for c in cases if c["split"]=="calibration"]
    assert len(calibration)==18 and sum(c["label"]==0 for c in calibration)==12
    assert {c["spec"]["approach_speed"] for c in calibration}=={3.5}
    for kind in ["background_translation","rotation","flicker","static"]:
        pair=[c for c in cases if c["suite"]=="core" and c["spec"]["kind"]==kind and c["spec"]["polarity"]==1 and c["schedule"]=="30hz"]
        assert len(pair)==2
        assert not np.array_equal(*[render_frame(StimulusSpec(**c["spec"]),1.0) for c in pair])


def test_direction_controls_have_same_channels_and_no_identity_or_duplicates():
    controls=permutations(configuration())
    assert len(controls)==len(set(controls.values()))==5
    assert all(sorted(p)==[0,1,2,3] and p!=(0,1,2,3) for p in controls.values())
    assert controls==permutations(configuration())


def test_json_restored_case_runs_all_models_with_physical_timestamps():
    config=configuration()
    config["assessment"]["warmup_s"]=0.05
    spec=StimulusSpec(kind="static",duration_s=0.15)
    case=json.loads(json.dumps(dict(case_id="fixture",split="challenge",suite="core",label=0,schedule="irregular",spec=asdict(spec))))
    rows,frames,maps,centers=score_case(case,config,permutations(config))
    assert len(rows)==9
    assert frames.groupby("model").t_s.apply(lambda x: x.is_monotonic_increasing).all()
    assert np.isfinite(frames.S).all()
    # Dense flow may have small numerical displacement even for identical texture.
    emd_rows=frames.loc[~frames.model.isin(["timed_global_flow","v0_legacy_unscaled"])]
    assert (emd_rows.S==0).all()
    assert centers.shape==(5,5,2)
    assert len(maps)==7
    assert all(r["integral_start_s"]==0.05 and r["integral_end_s"]==0.15 for r in rows)


def test_failure_outside_core_prevents_advancement():
    config=configuration()
    rows=[]
    for kind,label in [("loom",1),("translate",0)]:
        for schedule in config["matrix"]["schedules"]:
            rows.append(dict(split="challenge",model=MODEL_NAME,suite="core",kind=kind,label=label,
                             contrast=.8,polarity=-1,schedule=schedule,s_peak=float(label),detected=bool(label),localization_error_px=0.0))
    rows.append(dict(split="challenge",model=MODEL_NAME,suite="position",kind="loom",label=1,
                     contrast=.8,polarity=-1,schedule="30hz",s_peak=1.0,detected=True,localization_error_px=0.0))
    rows.append(dict(split="challenge",model=MODEL_NAME,suite="nuisance_speed",kind="translate",label=0,
                     contrast=.8,polarity=-1,schedule="15hz",s_peak=2.0,detected=True,localization_error_px=0.0))
    result=calculate_metrics(pd.DataFrame(rows),config)
    assert result["gates"]["all_schedules_recall"]
    assert result["gates"]["all_schedules_nuisance_fpr"]
    assert not result["gates"]["all_suites_schedules_nuisance_fpr"]
    assert not result["advance_to_roi"]


def test_zero_or_missed_offcenter_responses_are_not_removed_from_localization():
    config=configuration()
    rows=[]
    for schedule in config["matrix"]["schedules"]:
        for label,kind in [(0,"translate"),(1,"loom")]:
            rows.append(dict(split="challenge",model=MODEL_NAME,suite="core",kind=kind,label=label,
                contrast=.8,polarity=-1,schedule=schedule,s_peak=float(label),detected=bool(label),localization_error_px=0.0))
    for detected,error in [(True,0.0),(False,np.nan)]:
        rows.append(dict(split="challenge",model=MODEL_NAME,suite="position",kind="loom",label=1,
            contrast=.8,polarity=-1,schedule="30hz",s_peak=float(detected),detected=detected,localization_error_px=error))
    result=calculate_metrics(pd.DataFrame(rows),config)
    assert result["localization"]["n_requested"]==2
    assert result["localization"]["fraction"]==0.5
    assert not result["gates"]["localized_detection"]
