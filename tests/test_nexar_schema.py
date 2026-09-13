"""Schema, sampling, source provenance, and clock-contract regression tests."""
from pathlib import Path
from types import SimpleNamespace
import tempfile

import cv2
import numpy as np
import pandas as pd
import pytest

from fly_ttc.data.nexar import (
    HF_METADATA_COLUMNS, NEXAR_REQUIRED_COLUMNS, build_manifest, download_nexar,
    file_integrity, load_manifest, validate_manifest, write_manifest,
)
from fly_ttc.data.sampling import sample_subset, stratified_split
from fly_ttc.preprocess.video import iter_video_frames, probe_video


def manifest(n=12):
    return pd.DataFrame({
        "video_id": [f"{i:05d}" for i in range(n)],
        "label": [i % 2 for i in range(n)],
        "video_path": [f"/missing/{i:05d}.mp4" for i in range(n)],
        "time_of_alert": [3.0 if i % 2 else np.nan for i in range(n)],
        "time_of_event": [4.0 if i % 2 else np.nan for i in range(n)],
        "scene": ["Highway" if i % 3 == 0 else "Urban" for i in range(n)],
        "weather": ["Clear"] * n,
        "light_conditions": ["Dark" if i % 3 == 0 else "Normal" for i in range(n)],
    })


def test_required_columns_and_leading_zero_ids(tmp_path):
    df = manifest(2)
    path = write_manifest(df, tmp_path / "manifest.csv")
    with pytest.warns(UserWarning, match="Missing 2 video"):
        actual = load_manifest(path)
    assert set(NEXAR_REQUIRED_COLUMNS).issubset(actual)
    assert actual["video_id"].tolist() == ["00000", "00001"]
    assert actual.loc[0, "video_path"] == "/missing/00000.mp4"
    assert pd.isna(actual.loc[0, "time_of_event"])


@pytest.mark.parametrize("column,value", [("label", 2), ("label", 0.5), ("time_of_event", "bad"),
                                         ("time_of_event", -1), ("video_id", "../escape"),
                                         ("time_of_alert", float("inf"))])
def test_invalid_values_rejected(column, value):
    df = manifest(2).astype(object)
    df.loc[1, column] = value
    with pytest.raises(ValueError):
        validate_manifest(df)


def test_schema_missing_duplicate_and_clock_rejected():
    with pytest.raises(ValueError, match="Missing"):
        validate_manifest(manifest().drop(columns="time_of_alert"))
    df = manifest()
    df.loc[1, "video_id"] = df.loc[0, "video_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifest(df)
    df = manifest()
    df.loc[1, "time_of_alert"] = 5.0
    with pytest.raises(ValueError, match="exceeds"):
        validate_manifest(df)


def test_sampling_balanced_reproducible_and_order_invariant():
    df = manifest(100)
    first = sample_subset(df, 15, 15, seed=0)
    second = sample_subset(df.sample(frac=1, random_state=7), 15, 15, seed=0)
    assert first["video_id"].tolist() == second["video_id"].tolist()
    assert first["label"].value_counts().to_dict() == {0: 15, 1: 15}
    assert set(first["scene"]) == {"Urban", "Highway"}
    assert set(first["light_conditions"]) == {"Dark", "Normal"}
    missing_event_id = df.loc[1, "video_id"]
    df.loc[1, "time_of_event"] = np.nan
    assert missing_event_id not in sample_subset(df, 49, 50, seed=0)["video_id"].tolist()
    with pytest.raises(ValueError, match="only 49 eligible"):
        sample_subset(df, 50, 50, seed=0)


def test_split_no_overlap_both_labels_and_fixed_seed():
    df = manifest(100)
    evaluation, validation = stratified_split(df, 0.2, 0)
    assert len(evaluation) == 80 and len(validation) == 20
    assert not set(evaluation) & set(validation)
    assert set(evaluation) | set(validation) == set(df["video_id"])
    assert df.loc[df["video_id"].isin(validation), "label"].value_counts().to_dict() == {0: 10, 1: 10}
    assert (evaluation, validation) == stratified_split(df.iloc[::-1], 0.2, 0)
    with pytest.raises(ValueError, match="At least two"):
        stratified_split(manifest(2), 0.2, 0)


def test_local_kaggle_mapping_and_sha256(tmp_path):
    raw = tmp_path / "raw"
    (raw / "train").mkdir(parents=True)
    video = raw / "train/00001.mp4"
    video.write_bytes(b"local file integrity fixture")
    pd.DataFrame({"id": [1, 2], "target": [1, 0], "time_of_alert": [2., np.nan],
                  "time_of_event": [3., np.nan]}).to_csv(raw / "train.csv", index=False)
    out = build_manifest(raw / "train.csv", raw, tmp_path / "manifests/subset.csv")
    with pytest.warns(UserWarning, match="00002"):
        df = load_manifest(out)
    assert df["video_id"].tolist() == ["00001", "00002"]
    assert df.loc[0, "video_path"] == str(video)
    assert df.loc[0, "sha256"] == file_integrity(video)["sha256"]
    assert set(df["scene"]) == {"Unknown"}


def test_local_unexplained_filenames_fail(tmp_path):
    (tmp_path / "something-unrelated.mp4").write_bytes(b"fixture")
    pd.DataFrame({"id": [1], "target": [1], "time_of_alert": [2.],
                  "time_of_event": [3.]}).to_csv(tmp_path / "train.csv", index=False)
    with pytest.raises(ValueError, match="No CSV IDs match"):
        build_manifest(tmp_path / "train.csv", tmp_path, tmp_path / "out.csv")


def test_download_metadata_and_subset_are_exact_train_paths(tmp_path, monkeypatch):
    import fly_ttc.data.nexar as nexar
    recorded = []
    source_names = {"positive": ["00001.mp4", "00003.mp4"], "negative": ["00002.mp4", "00004.mp4"]}
    sha = "a" * 40
    tree = [SimpleNamespace(path=f"train/{label}/{name}", size=3, lfs=None)
            for label, names in source_names.items() for name in names]

    class FakeApi:
        def __init__(self, token):
            assert token is False
        def dataset_info(self, repo, revision):
            return SimpleNamespace(sha=sha)
        def list_repo_tree(self, *args, **kwargs):
            assert kwargs["path_in_repo"] == "train"
            assert kwargs["revision"] == sha
            return tree

    def snapshot(repo, **kwargs):
        recorded.append(kwargs["allow_patterns"])
        root = Path(kwargs["local_dir"])
        assert kwargs["revision"] == sha
        for path in kwargs["allow_patterns"]:
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.endswith("metadata.csv"):
                label = Path(path).parent.name
                rows = [{"file_name": name, "time_of_event": 4. if label == "positive" else None,
                         "time_of_alert": 3. if label == "positive" else None,
                         "scene": "Urban", "weather": "Clear", "light_conditions": "Normal",
                         "time_to_accident": None} for name in source_names[label]]
                pd.DataFrame(rows, columns=HF_METADATA_COLUMNS).to_csv(target, index=False)
            else:
                target.write_bytes(b"abc")

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(nexar, "HfApi", FakeApi)
    monkeypatch.setattr(nexar, "snapshot_download", snapshot)
    root, manifests = tmp_path / "raw", tmp_path / "manifests"
    metadata = download_nexar(metadata_only=True, raw_dir=root, manifest_dir=manifests)
    assert not list(root.rglob("*.mp4"))
    assert len(pd.read_csv(metadata)) == 4
    result = download_nexar(subset=2, seed=0, raw_dir=root, manifest_dir=manifests)
    actual = load_manifest(result)
    assert len(actual) == 2
    assert set(actual["source_revision"]) == {sha}
    assert actual["label"].sum() == 1
    assert all(path.startswith("train/") and path.endswith(".mp4") and "*" not in path for path in recorded[-1])
    assert actual["sha256"].notna().all()
    frozen_ids = actual["video_id"].tolist()
    repeated = download_nexar(subset=2, seed=0, raw_dir=root, manifest_dir=manifests)
    assert load_manifest(repeated)["video_id"].tolist() == frozen_ids


def test_video_reader_uses_source_clock_resampling_and_offset():
    fixture_dir = Path("outputs/synthetic/test_video_reader")
    fixture_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=fixture_dir) as temp:
        path = Path(temp) / "clock.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
        assert writer.isOpened()
        for i in range(40):
            writer.write(np.full((48, 64, 3), i * 5, np.uint8))
        writer.release()
        info = probe_video(path)
        assert info["fps"] == pytest.approx(20)
        assert info["duration_s"] == pytest.approx(2)
        samples = list(iter_video_frames(path, 1.25, 1.75, target_fps=10, short_side=24, time_offset_s=1))
        np.testing.assert_allclose([t for t, _ in samples], [1.25, 1.35, 1.45, 1.55, 1.65, 1.75])
        assert all(gray.shape == (24, 32) and gray.dtype == np.float32 for _, gray in samples)
        assert samples[0][1].mean() == pytest.approx(25/255, abs=0.015)
        with pytest.warns(UserWarning, match="differs from metadata"):
            probe_video(path, metadata_fps=30)


@pytest.mark.parametrize("column", ["time_of_alert", "time_of_event"])
def test_negative_rows_reject_event_or_alert_time(column):
    df = manifest(2)
    df.loc[0, column] = 2.0
    with pytest.raises(ValueError, match="Negative Nexar rows must have null"):
        validate_manifest(df)


def test_positive_missing_event_is_retained_for_runner_skip():
    df = manifest(2)
    df.loc[1, "time_of_event"] = np.nan
    actual = validate_manifest(df)
    assert actual.loc[1, "label"] == 1
    assert pd.isna(actual.loc[1, "time_of_event"])


def test_local_padding_uses_observed_stem_then_preserves_normalized_id(tmp_path):
    video = tmp_path / "0012.mp4"
    video.write_bytes(b"fixture")
    source = tmp_path / "train.csv"
    pd.DataFrame({"id": ["0012"], "target": [1], "time_of_alert": [2.],
                  "time_of_event": [3.]}).to_csv(source, index=False)
    output = build_manifest(source, tmp_path, tmp_path / "manifest.csv")
    actual = load_manifest(output)
    assert actual.loc[0, "video_id"] == "00012"
    assert actual.loc[0, "video_path"] == str(video)


def test_local_ambiguous_observed_stems_fail(tmp_path):
    for folder in ("first", "second"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "00012.mp4").write_bytes(b"fixture")
    source = tmp_path / "train.csv"
    pd.DataFrame({"id": [12], "target": [1], "time_of_alert": [2.],
                  "time_of_event": [3.]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="Ambiguous observed filenames"):
        build_manifest(source, tmp_path, tmp_path / "out.csv")


def test_download_failure_prints_exact_command_and_redacts_token(monkeypatch, capsys):
    import shlex
    import sys
    import fly_ttc.data.nexar as nexar
    token = "hf_unit_test_secret_not_a_real_token"
    monkeypatch.setenv("HF_TOKEN", token)
    def fail(**kwargs):
        raise ConnectionError(f"Cannot connect using credential {token}")
    monkeypatch.setattr(nexar, "download_nexar", fail)
    argv = ["--subset", "100", "--seed", "0", "--raw-dir", "folder with spaces"]
    assert nexar.download_main(argv) == 1
    output = capsys.readouterr().err
    expected = shlex.join([sys.executable, "scripts/download_nexar.py", *argv])
    assert f"Exact command: {expected}" in output
    assert "[REDACTED]" in output and token not in output
    assert "No alternate dataset was downloaded" in output
    # Redaction also covers the retry command if a secret accidentally occurs in an argument.
    assert nexar.download_main(["--raw-dir", token]) == 1
    assert token not in capsys.readouterr().err


def test_download_failure_preserves_module_invocation(monkeypatch, capsys):
    import shlex
    import sys
    import fly_ttc.data.nexar as nexar
    argv = ["--subset", "100", "--seed", "0"]
    original = [".venv/bin/python", "-m", "fly_ttc.cli", "download", *argv]
    monkeypatch.setattr(sys, "orig_argv", original)
    monkeypatch.setattr(sys, "argv", ["/repo/src/fly_ttc/cli.py", "download", *argv])
    def fail(**kwargs):
        raise ConnectionError("offline fixture")
    monkeypatch.setattr(nexar, "download_nexar", fail)
    assert nexar.download_main(argv) == 1
    assert f"Exact command: {shlex.join(original)}" in capsys.readouterr().err
