"""Review archive content and reproducibility checks using temporary fixtures."""

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


spec = importlib.util.spec_from_file_location("export_review", Path(__file__).parents[1] / "scripts" / "export_review.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def write(root, relative, payload=b"fixture"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


@pytest.fixture
def review_fixture(tmp_path):
    root = tmp_path / "repo"
    code = ["README.md", ".gitignore", "src/model.py", "tests/test_model.py"]
    artifacts = [
        "outputs/v0/REPORT.md", "outputs/v0/REPORT.html", "outputs/v0/figures/trace.png",
        "outputs/v0/frames/00001.parquet", "outputs/v0/clip_scores.csv",
        "outputs/v0_review/analysis.json", "outputs/ego_motion/config.yaml", "outputs/synthetic/report.json",
    ]
    excluded = [
        ".venv/lib/python.py", "src/nested/.venv/lib/python.py", "src/__pycache__/model.pyc",
        ".git/config", ".pytest_cache/result.json", "data/raw/nexar/00001.mp4",
        "data/manifests/subset.csv", "src/datasets/private.csv", "src/cache/features.json",
        "src/example.MP4", "src/raw/sample.csv", "src/.env", "src/.env.production",
        "outputs/v0/.staging/frames/00001.parquet", "outputs/v0/.componentcache/00001.parquet",
        "outputs/v0/component_cache/00001.parquet", "outputs/v0/component-cache/00001.parquet",
        "outputs/v0/_staging/00001.parquet", "outputs/v0/cache/summary.json",
        "outputs/v0/.venv/lib/frame.parquet", "outputs/v0/raw.mp4", "outputs/v0/arrays.npz",
        "outputs/v0/.hidden.json", "outputs/v0/report.svg", "outputs/unapproved/REPORT.md",
        "exports/old.zip", "EXPORT_MANIFEST.json",
    ]
    for name in code + artifacts + excluded:
        write(root, name, name.encode())
    return root, code, artifacts, excluded


def test_collection_excludes_cache_raw_symlinks_and_unapproved_artifacts(review_fixture, tmp_path):
    root, code, artifacts, excluded = review_fixture
    outside = write(tmp_path, "outside/report.parquet")
    (root / "outputs/v0/linked.parquet").symlink_to(outside)
    (root / "outputs/v0/linked_dir").symlink_to(outside.parent, target_is_directory=True)
    (root / "src/linked.py").symlink_to(outside)
    (root / "src/linked_dir").symlink_to(outside.parent, target_is_directory=True)
    candidates = code + excluded + ["src/linked.py", "src/linked_dir/report.parquet", "../outside/report.parquet", str(outside)]
    actual = exporter.collect_export_files(root, code_paths=candidates)
    assert [path.as_posix() for path in actual] == sorted(code + artifacts)


def test_archive_manifest_matches_payload_and_bytes_repeat(review_fixture):
    root, code, artifacts, excluded = review_fixture
    destination = root / "exports/review.zip"
    manifest = exporter.export_review(root, destination, code_paths=code + excluded)
    first = destination.read_bytes()
    exporter.export_review(root, destination, code_paths=list(reversed(code + excluded)))
    assert first == destination.read_bytes()
    assert manifest["file_count"] == len(code + artifacts)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        stored = json.loads(archive.read(exporter.MANIFEST_NAME))
        assert stored == manifest
        assert set(archive.namelist()) == {item["name"] for item in manifest["files"]} | {exporter.MANIFEST_NAME}
        for item in manifest["files"]:
            payload = archive.read(item["name"])
            assert item["size"] == len(payload)
            assert item["sha256"] == hashlib.sha256(payload).hexdigest()
            assert archive.getinfo(item["name"]).date_time == (1980, 1, 1, 0, 0, 0)


def test_artifact_parent_symlink_is_never_traversed(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside"
    write(outside, "v0/REPORT.md")
    (root / "outputs").symlink_to(outside, target_is_directory=True)
    assert exporter.collect_export_files(root, code_paths=[]) == []


def test_git_candidates_include_untracked_and_use_nul_delimiters(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    write(root, "tracked.py")
    write(root, "new file.py")

    def fake_run(command, **kwargs):
        assert command == ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
        assert kwargs["cwd"] == root
        assert kwargs["check"] and kwargs["capture_output"]
        return type("Result", (), {"stdout": b"tracked.py\0new file.py\0tracked.py\0"})()

    monkeypatch.setattr(exporter.subprocess, "run", fake_run)
    assert exporter.collect_export_files(root) == [Path("new file.py"), Path("tracked.py")]


def test_output_validation_does_not_overwrite_link_target(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = write(tmp_path, "outside.zip", b"preserve")
    (root / "review.zip").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        exporter.export_review(root, root / "review.zip", code_paths=[])
    assert outside.read_bytes() == b"preserve"
    with pytest.raises(ValueError, match=".zip"):
        exporter.export_review(root, root / "README.md", code_paths=[])
