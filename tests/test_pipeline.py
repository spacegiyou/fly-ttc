"""Integration checks for calibration membership and serialized run artifacts."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fly_ttc import cli


def test_pipeline_calibrates_only_negative_validation_and_excludes_post_event(tmp_path, monkeypatch):
    from fly_ttc.data import nexar
    from fly_ttc.data.sampling import stratified_split

    config = yaml.safe_load((Path(__file__).parents[1] / "configs/default.yaml").read_text())
    out = tmp_path / "outputs"
    config["paths"] = {"manifest": str(tmp_path / "source.csv"), "raw_dir": str(tmp_path), "output_dir": str(out)}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    manifest = pd.DataFrame([
        {"video_id": f"{i:05d}", "label": label, "video_path": str(tmp_path / f"{i:05d}.mp4"),
         "time_of_event": 3.0 if label else np.nan, "time_of_alert": 2.0 if label else np.nan,
         "scene": "Urban", "weather": "Clear", "light_conditions": "Normal"}
        for i, label in enumerate([0, 0, 1, 1])
    ])
    _, validation = stratified_split(manifest, config["eval"]["val_frac"], config["seed"])
    expected_ids = sorted(set(validation) & set(manifest.loc[manifest.label.eq(0), "video_id"]))
    monkeypatch.setattr(nexar, "load_manifest", lambda _: manifest.copy())

    def extract(row, config, snapshot_targets=None):
        values = [1.0, 2.0, 3.0, 4.0] if row.label == 0 else [2.0, 3.0, 4.0, 10000.0]
        frames = pd.DataFrame({"t": [0.5, 1.5, 2.5, 3.5], "S_div": values, "S_rad": values,
                               "S_blob": np.zeros(4), "flow_median": values,
                               "flow_horizontal": values, "flow_valid": True})
        return frames, {"status": "ok", "duration_s": 4.0, "fps": 15.0}, []

    monkeypatch.setattr(cli, "extract_clip", extract)
    cli.run_v0(config_path)
    threshold = yaml.safe_load((out / "threshold.yaml").read_text())
    assert threshold["normalization"]["video_ids"] == expected_ids
    assert threshold["normalization"]["S_div"]["mean"] == 2.5
    assert set(threshold["evaluation_ids"]).isdisjoint(threshold["validation_ids"])
    assert threshold["val_fpr"] <= config["eval"]["fpr_target"]
    scores = pd.read_csv(out / "clip_scores.csv", dtype={"video_id": str})
    assert len(scores) == 4
    assert (scores.loc[scores.label.eq(1), "t_peak"] < 3).all()
    assert (scores.loc[scores.label.eq(1), "s_peak"] < 100).all()
    assert (out / "metrics.json").exists()
    assert len(list((out / "frames").glob("*.parquet"))) >= 2


def test_v1_is_explicitly_deferred(capsys):
    assert cli.main(["run", "--model", "v1"]) == 1


def test_failed_rerun_preserves_previous_complete_report(tmp_path, monkeypatch):
    from fly_ttc.data import nexar

    config = yaml.safe_load((Path(__file__).parents[1] / "configs/default.yaml").read_text())
    out = tmp_path / "v0"
    out.mkdir()
    previous = out / "REPORT.md"
    previous.write_text("Previous complete run")
    config["paths"]["output_dir"] = str(out)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    def invalid(_):
        raise ValueError("Invalid manifest")

    monkeypatch.setattr(nexar, "load_manifest", invalid)
    with pytest.raises(ValueError, match="Invalid manifest"):
        cli.run_v0(path)
    assert previous.read_text() == "Previous complete run"
