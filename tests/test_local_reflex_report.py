"""Presentation must retain all conditions, zero responses and frozen decisions."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fly_ttc.experiments.local_reflex import build_cases, calculate_metrics, permutations
from fly_ttc.viz.local_reflex_report import FIGURES, IDENTITY, _localization_rows, make_report


@pytest.fixture
def recorded_run(tmp_path):
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/local_reflex.yaml").read_text())
    cases = build_cases(config)
    controls = permutations(config)
    models = [IDENTITY, *controls, "simple_motion_energy", "timed_global_flow", "v0_legacy_unscaled"]
    rows, traces, samples = [], [], {}
    for case in cases:
        spec = case["spec"]
        for i, model in enumerate(models):
            factor = i + 1
            response = (1 if case["label"] else .1) * factor
            failed_location = case["suite"] == "position" and spec["center_fraction"][0] == .25
            if failed_location:
                response = 0
            local_error = np.nan if response == 0 or model in {"timed_global_flow", "v0_legacy_unscaled"} else 3.0
            rows.append(dict(case_id=case["case_id"], split=case["split"], suite=case["suite"],
                             kind=spec["kind"], schedule=case["schedule"], contrast=spec["contrast"], polarity=spec["polarity"],
                             center_x=spec["center_fraction"][0], center_y=spec["center_fraction"][1],
                             label=case["label"], model=model, s_peak=response, theta=.5*factor,
                             detected=response>.5*factor, localization_error_px=local_error,
                             disk_fully_visible_fraction=1.0))
            for t, dt, score in [(0, 0, 0), (.3, .3, response/2), (.5, .2, response)]:
                traces.append(dict(case_id=case["case_id"], model=model, t_s=t, dt_s=dt, S=score))
        if case["suite"] == "core" and case["schedule"] == "30hz" and spec["contrast"] == .8 and spec["polarity"] == -1:
            y, x = np.mgrid[:5, :5]
            samples[case["case_id"]] = dict(kind=spec["kind"], map=(np.ones((5, 5))*(1 if case["label"] else .1)).tolist(),
                                          centers=np.stack([16+x*23.75, 16+y*23.75], axis=-1).tolist())
    frame = pd.DataFrame(rows)
    frame.to_csv(tmp_path / "case_scores.csv", index=False)
    pd.DataFrame(traces).to_parquet(tmp_path / "traces.parquet", index=False)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "protocol.json").write_text(json.dumps(dict(n_cases=len(cases), direction_permutations=controls)))
    (tmp_path / "metrics.json").write_text(json.dumps(calculate_metrics(frame, config)))
    (tmp_path / "sample_maps.json").write_text(json.dumps(samples))
    (tmp_path / "run_state.json").write_text(json.dumps(dict(status="complete", n_rows=len(frame))))
    return tmp_path, models


def test_complete_report_shows_all_models_and_preserves_scientific_inputs(recorded_run):
    out, models = recorded_run
    hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    path = make_report(out)
    text = path.read_text()
    assert all(model in text for model in models)
    assert "Nexar AUROC나 실도로 성능이 아니다" in text
    assert "진전 조건: 미충족" in text
    assert "전체 뇌를 실행하지 않았다" in text
    assert "다른 일정에서 cutoff를 넘은 경우" in text
    assert "../../../docs/LOCAL_REFLEX_MODEL_SOURCES.md" in text
    assert "생물학적 구조의 우위도 입증하지 않았다" in text
    assert "이번 실행에서 수행하지 않았다" in text
    assert all((out / "figures" / name).stat().st_size > 1000 for name in FIGURES)
    assert len(list((out / "figures").glob("*.png"))) == 5
    rendered = (out / "REPORT.html").read_text()
    assert rendered.count("<img ") == 5
    assert "<title>Local visual reflex" in rendered
    assert hashes == {name: sha256((out / name).read_bytes()).hexdigest() for name in hashes}


def test_localization_keeps_zero_response_in_denominator(recorded_run):
    out, models = recorded_run
    scores = pd.read_csv(out / "case_scores.csv")
    rows = _localization_rows(scores, models, tolerance=23.75)
    assert len(rows) == 7
    for row in rows:
        assert row["requested"] == 32
        assert row["localized"] == 24
        assert row["missed"] == 8
        assert row["zero_response"] == 8


def test_incomplete_run_and_missing_fixed_map_are_rejected(recorded_run):
    out, _ = recorded_run
    state = json.loads((out / "run_state.json").read_text())
    (out / "run_state.json").write_text(json.dumps({**state, "status": "running"}))
    with pytest.raises(ValueError, match="complete run"):
        make_report(out)
    (out / "run_state.json").write_text(json.dumps(state))
    maps = json.loads((out / "sample_maps.json").read_text())
    maps.pop(next(iter(maps)))
    (out / "sample_maps.json").write_text(json.dumps(maps))
    with pytest.raises(ValueError, match="every prespecified"):
        make_report(out)
