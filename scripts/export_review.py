#!/usr/bin/env python3
"""Build a reproducible local review ZIP containing code and selected reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile


ARTIFACT_RUNS = {"v0", "v0_review", "ego_motion", "synthetic", "review_response"}
ARTIFACT_SUFFIXES = {".md", ".html", ".png", ".csv", ".json", ".yaml", ".parquet", ".txt"}
EXCLUDED_SUFFIXES = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg",
    ".wmv", ".flv", ".mts", ".m2ts", ".npz", ".pyc", ".pyo", ".zip",
}
EXCLUDED_DIRECTORY_NAMES = {
    ".venv", "venv", ".git", ".pytest_cache", "__pycache__", ".cache",
    "cache", "caches", "componentcache", "staging", "node_modules",
    "dataset", "datasets", "raw", "exports",
}
ARCHIVE_ROOT = "fly_ttc"
MANIFEST_NAME = f"{ARCHIVE_ROOT}/EXPORT_MANIFEST.json"
EXCLUSIONS = [
    "Virtual environments, git metadata, test/Python caches, node_modules, and export archives.",
    "All data/ contents, raw/dataset directories, video files, and .npz caches.",
    "Symlinks and paths outside the repository; hidden directories and hidden report files.",
    "Staging and component-cache directories, including underscore/hyphen name variants.",
    "Environment secret files (.env and .env.*).",
    "Reports outside outputs/{v0,v0_review,ego_motion,synthetic,review_response} or outside the artifact extension allowlist.",
]


def _excluded_directory(name: str) -> bool:
    normalized = name.casefold().replace("_", "").replace("-", "")
    return (
        name.startswith(".")
        or name.casefold() in EXCLUDED_DIRECTORY_NAMES
        or normalized in {"venv", "pycache", "pytestcache", "componentcache", "componentcaches", "staging", "cache", "caches"}
    )


def _safe_file(root: Path, relative: Path, artifact: bool) -> bool:
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        return False
    if relative.parts[0] == "data":
        return False
    if any(_excluded_directory(part) for part in relative.parts[:-1]):
        return False
    if relative.suffix.casefold() in EXCLUDED_SUFFIXES:
        return False
    name = relative.name.casefold()
    if name == ".env" or name.startswith(".env."):
        return False
    if artifact and (relative.name.startswith(".") or relative.suffix.casefold() not in ARTIFACT_SUFFIXES):
        return False
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return False
    return current.is_file()


def _git_code_paths(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, check=True, capture_output=True,
    )
    return [Path(os.fsdecode(value)) for value in result.stdout.split(b"\0") if value]


def collect_export_files(root, code_paths=None) -> list[Path]:
    """Return sorted repository-relative files without following symlinks.

    Code candidates come from git unless explicitly supplied. The latter is
    useful for deterministic fixture tests without altering a live repository.
    Artifact inclusion is independent of gitignore and never bypasses filters.
    """
    root = Path(root).resolve()
    selected = set()
    for candidate in _git_code_paths(root) if code_paths is None else code_paths:
        relative = Path(candidate)
        if relative.parts and relative.parts[0] == "outputs":
            continue
        if _safe_file(root, relative, artifact=False):
            selected.add(relative)
    for run in sorted(ARTIFACT_RUNS):
        folder = root / "outputs" / run
        if folder.is_symlink() or (root / "outputs").is_symlink() or not folder.is_dir():
            continue
        for directory, dirs, files in os.walk(folder, followlinks=False):
            dirs[:] = sorted(name for name in dirs if not _excluded_directory(name) and not (Path(directory) / name).is_symlink())
            for name in files:
                relative = (Path(directory) / name).relative_to(root)
                if _safe_file(root, relative, artifact=True):
                    selected.add(relative)
    # Avoid an ambiguous duplicate entry if a previous manifest was checked in.
    selected.discard(Path("EXPORT_MANIFEST.json"))
    return sorted(selected, key=lambda path: path.as_posix())


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def export_review(root, output, code_paths=None) -> dict:
    """Write an atomic, reproducible ZIP; return its content manifest.

    Fixed archive timestamps, permissions, ordering, and JSON formatting make
    repeated exports of unchanged bytes identical on the same ZIP runtime.
    The manifest hashes each payload file, excluding itself to avoid recursion.
    """
    root, output = Path(root).resolve(), Path(output)
    if not output.is_absolute():
        output = root / output
    output = output.absolute()
    if output.suffix.casefold() != ".zip":
        raise ValueError("Review archive output must have a .zip extension")
    if output.is_symlink():
        raise ValueError("Review archive output must not be a symlink")
    files = collect_export_files(root, code_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "archive_root": ARCHIVE_ROOT,
        "selection": "git ls-files --cached --others --exclude-standard plus explicitly allowed report artifacts",
        "artifact_runs": sorted(ARTIFACT_RUNS),
        "artifact_extensions": sorted(ARTIFACT_SUFFIXES),
        "exclusions": EXCLUSIONS,
        "manifest_hash_scope": "Payload files only; this manifest does not hash itself.",
        "files": [],
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".review-", suffix=".zip", dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for relative in files:
                artifact = relative.parts[0] == "outputs"
                if not _safe_file(root, relative, artifact):
                    raise RuntimeError(f"Source file changed or became unsafe during export: {relative}")
                payload = (root / relative).read_bytes()
                name = f"{ARCHIVE_ROOT}/{relative.as_posix()}"
                manifest["files"].append({"name": name, "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)})
                archive.writestr(_zip_info(name), payload, compresslevel=9)
            manifest["file_count"] = len(manifest["files"])
            manifest["uncompressed_payload_bytes"] = sum(item["size"] for item in manifest["files"])
            payload = (json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            archive.writestr(_zip_info(MANIFEST_NAME), payload, compresslevel=9)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="exports/fly_ttc_review.zip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = export_review(root, args.output)
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    print(f"{output}: {manifest['file_count']} files, {manifest['uncompressed_payload_bytes']} uncompressed payload bytes")


if __name__ == "__main__":
    main()
