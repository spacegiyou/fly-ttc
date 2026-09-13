"""Nexar train metadata, exact-file downloading, and local manifest validation.

Primary schema verified at:
https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/tree/main/train
The official VideoFolder schema assigns positive=1 and negative=0 by directory;
metadata.csv supplies an exact file_name, not an inferred label or video path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import warnings

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, snapshot_download

from fly_ttc.data.sampling import sample_subset

NEXAR_REPO_ID = "nexar-ai/nexar_collision_prediction"
NEXAR_REQUIRED_COLUMNS = (
    "video_id", "label", "video_path", "time_of_alert", "time_of_event",
    "scene", "weather", "light_conditions",
)
HF_METADATA_FILES = ("train/positive/metadata.csv", "train/negative/metadata.csv")
HF_METADATA_COLUMNS = (
    "file_name", "time_of_event", "time_of_alert", "light_conditions",
    "weather", "scene", "time_to_accident",
)


def _csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"video_id": str, "id": str, "file_name": str})


def _numeric(series: pd.Series, name: str) -> pd.Series:
    try:
        values = pd.to_numeric(series, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid numeric values in {name}") from exc
    if ((values.notna()) & (~np.isfinite(values))).any():
        raise ValueError(f"Non-finite values in {name}")
    return values


def validate_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Validate project columns strictly without inventing dataset annotations."""
    missing = set(NEXAR_REQUIRED_COLUMNS) - set(df)
    if missing:
        raise ValueError(f"Missing Nexar manifest columns: {sorted(missing)}")
    df = df.copy()
    if df.empty:
        raise ValueError("Nexar manifest is empty")
    if df["video_id"].isna().any():
        raise ValueError("video_id must not be null")
    df["video_id"] = df["video_id"].astype(str)
    if df["video_id"].duplicated().any():
        raise ValueError("Duplicate video_id in manifest")
    if not df["video_id"].map(lambda x: bool(re.fullmatch(r"[A-Za-z0-9_-]+", x))).all():
        raise ValueError("video_id must be a safe file stem containing letters, digits, '_' or '-'")
    df["label"] = _numeric(df["label"], "label")
    if not df["label"].isin([0, 1]).all():
        raise ValueError("label must be exactly 0 or 1")
    df["label"] = df["label"].astype(int)
    if df["video_path"].isna().any() or df["video_path"].astype(str).str.strip().eq("").any():
        raise ValueError("video_path must be explicit and nonempty")
    for column in ("time_of_alert", "time_of_event"):
        df[column] = _numeric(df[column], column)
        if df[column].dropna().lt(0).any():
            raise ValueError(f"{column} must be nonnegative")
    negative_times = df.loc[df["label"].eq(0), ["time_of_alert", "time_of_event"]]
    if negative_times.notna().any().any():
        raise ValueError("Negative Nexar rows must have null time_of_alert and time_of_event")
    valid_times = df["time_of_alert"].notna() & df["time_of_event"].notna()
    if (df.loc[valid_times, "time_of_alert"] > df.loc[valid_times, "time_of_event"]).any():
        raise ValueError("time_of_alert exceeds time_of_event")
    for column in ("scene", "weather", "light_conditions"):
        df[column] = df[column].fillna("Unknown").astype(str)
    return df


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load normalized CSV; relative video paths are relative to the manifest.

    Missing files are warned about and retained for the runner to record/skip.
    Metadata-only manifests therefore remain valid without local videos.
    """
    path = Path(path).resolve()
    df = validate_manifest(_csv(path))
    df["video_path"] = df["video_path"].map(
        lambda value: str((path.parent / value).resolve()) if not Path(value).is_absolute() else str(Path(value))
    )
    missing = df.loc[~df["video_path"].map(lambda value: Path(value).is_file()), "video_id"].tolist()
    if missing:
        warnings.warn(f"Missing {len(missing)} video(s); runner will skip only these IDs: {', '.join(missing)}", stacklevel=2)
    return df


def file_integrity(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        return {"size_bytes": None, "mtime_ns": None, "sha256": None}
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def _integrity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    checks = pd.DataFrame([file_integrity(path) for path in df["video_path"]], index=df.index, dtype=object)
    for column in checks:
        df[column] = checks[column]
    if "source_sha256" in df:
        present = df["sha256"].notna() & df["source_sha256"].notna()
        if df.loc[present, "sha256"].ne(df.loc[present, "source_sha256"]).any():
            raise ValueError("Downloaded SHA256 differs from the pinned HF source")
    for column in ("size_bytes", "mtime_ns", "source_size_bytes"):
        if column in df:
            df[column] = pd.array(df[column], dtype="Int64")
    return df


def write_manifest(df: pd.DataFrame, output: str | Path) -> Path:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    df = validate_manifest(df).copy()
    df["video_path"] = df["video_path"].map(lambda value: os.path.relpath(Path(value).resolve(), output.parent))
    temp = output.with_suffix(output.suffix + ".tmp")
    df.to_csv(temp, index=False)
    temp.replace(output)
    return output


def _hf_manifest(raw_dir: Path, *, tree: dict | None = None, revision: str | None = None) -> pd.DataFrame:
    parts = []
    for source, label in zip(HF_METADATA_FILES, (1, 0)):
        csv_path = raw_dir / source
        data = _csv(csv_path)
        missing = set(HF_METADATA_COLUMNS) - set(data)
        if missing:
            raise ValueError(f"Unexpected HF metadata schema in {source}: missing {sorted(missing)}")
        data["source_path"] = data["file_name"].map(lambda name: str(Path(source).parent / str(name)))
        if data["file_name"].isna().any() or not data["file_name"].map(lambda name: Path(name).name == name and name.endswith(".mp4")).all():
            raise ValueError(f"Unsafe or unexpected file_name in {source}")
        if tree is not None:
            unmatched = sorted(set(data["source_path"]) - set(tree))
            if unmatched:
                raise ValueError(f"Metadata filenames absent from HF train tree: {unmatched}")
            data["source_size_bytes"] = data["source_path"].map(lambda p: tree[p]["size"])
            data["source_sha256"] = data["source_path"].map(lambda p: tree[p].get("sha256"))
        data["video_id"] = data["file_name"].map(lambda name: Path(name).stem)
        data["label"] = label
        data["video_path"] = data["source_path"].map(lambda source_path: str((raw_dir / source_path).resolve()))
        data["source_repo"] = NEXAR_REPO_ID
        data["source_revision"] = revision
        parts.append(data)
    return validate_manifest(pd.concat(parts, ignore_index=True)).sort_values("video_id").reset_index(drop=True)


def build_manifest(source: str | Path, raw_dir: str | Path, output: str | Path,
                   *, subset: int | None = None, seed: int = 0) -> Path:
    """Build from official HF metadata directories or local Kaggle train CSV.

    A normalized CSV with explicit video_path is also supported. Local paths in
    source CSVs are resolved from the CSV parent; missing files keep their IDs.
    Kaggle IDs first try train/{id:05d}.mp4 and then exact observed filename stems.
    Ambiguous or unexplained filename mappings fail rather than guess.
    """
    source, raw_dir = Path(source).resolve(), Path(raw_dir).resolve()
    if source.is_dir():
        data = _hf_manifest(source)
    else:
        data = _csv(source)
        if set(NEXAR_REQUIRED_COLUMNS).issubset(data):
            data["video_path"] = data["video_path"].map(lambda p: str((source.parent / p).resolve()) if not Path(p).is_absolute() else p)
            data = validate_manifest(data)
        else:
            required = {"id", "target", "time_of_alert", "time_of_event"}
            if not required.issubset(data):
                raise ValueError(f"Local CSV must contain {sorted(required)} or all normalized columns")
            videos = sorted(raw_dir.rglob("*.mp4"))
            by_stem: dict[str, list[Path]] = {}
            for video in videos:
                by_stem.setdefault(video.stem, []).append(video)
            mapped, ids, matched = [], [], set()
            for source_id in data["id"]:
                if pd.isna(source_id):
                    raise ValueError("Null Kaggle id")
                value = str(source_id)
                canonical_id = f"{int(value):05d}" if value.isdecimal() else value
                canonical = raw_dir / "train" / f"{canonical_id}.mp4"
                candidates = list(dict.fromkeys(by_stem.get(canonical_id, []) + by_stem.get(value, [])))
                if canonical.is_file():
                    path = canonical
                elif len(candidates) == 1:
                    path = candidates[0]
                elif len(candidates) > 1:
                    raise ValueError(f"Ambiguous observed filenames for id={value}: {candidates}")
                else:
                    path = canonical
                if path.is_file():
                    matched.add(path)
                mapped.append(str(path))
                ids.append(canonical_id)
            if videos and not matched:
                raise ValueError("No CSV IDs match observed mp4 stems; provide an explicit normalized video_path mapping")
            data = data.rename(columns={"target": "label"})
            data["video_id"], data["video_path"] = ids, mapped
            for column in ("scene", "weather", "light_conditions"):
                if column not in data:
                    data[column] = "Unknown"
            data["source_repo"] = "local Nexar train (user supplied)"
            data["source_path"] = str(source)
            data = validate_manifest(data)
    if subset is not None:
        if subset <= 0 or subset % 2:
            raise ValueError("subset must be a positive even number")
        data = sample_subset(data, subset // 2, subset // 2, seed)
    data["subset_seed"] = seed
    return write_manifest(_integrity(data), output)


def download_nexar(*, metadata_only: bool = False, subset: int = 100, full: bool = False,
                   seed: int = 0, raw_dir: str | Path = "data/raw/nexar",
                   manifest_dir: str | Path = "data/manifests", revision: str = "main") -> Path:
    """Download only pinned train metadata and requested exact video paths.

    HF_TOKEN is the sole explicit credential input. No saved-login token is read
    when it is absent. A network error propagates; no other dataset is substituted.
    Existing subset manifests freeze the revision and selected IDs on reruns.
    """
    if subset <= 0 or subset % 2:
        raise ValueError("subset must be a positive even number")
    if metadata_only and full:
        raise ValueError("metadata_only and full are mutually exclusive")
    raw_dir, manifest_dir = Path(raw_dir).resolve(), Path(manifest_dir).resolve()
    output = manifest_dir / ("train.csv" if full or metadata_only else f"subset_{subset}_seed{seed}.csv")
    previous = None
    if output.exists() and not (full or metadata_only):
        previous = validate_manifest(_csv(output))
        if "source_revision" not in previous or previous["source_revision"].nunique() != 1:
            raise ValueError(f"Existing subset lacks a unique pinned HF revision: {output}")
        revision = str(previous["source_revision"].iloc[0])
    token = os.environ.get("HF_TOKEN") or False
    api = HfApi(token=token)
    commit = api.dataset_info(NEXAR_REPO_ID, revision=revision).sha
    tree = {}
    for item in api.list_repo_tree(NEXAR_REPO_ID, path_in_repo="train", recursive=True, repo_type="dataset", revision=commit):
        if hasattr(item, "size") and item.path.endswith(".mp4"):
            lfs = getattr(item, "lfs", None)
            digest = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
            tree[item.path] = {"size": item.size, "sha256": digest}
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(NEXAR_REPO_ID, repo_type="dataset", revision=commit, token=token,
                      local_dir=raw_dir, allow_patterns=[*HF_METADATA_FILES, "LICENSE", "README.md"], max_workers=4)
    data = _hf_manifest(raw_dir, tree=tree, revision=commit)
    provenance = {"repo_id": NEXAR_REPO_ID, "revision": commit, "split": "train",
                  "train_rows": len(data), "label_counts": data["label"].value_counts().sort_index().to_dict(),
                  "metadata_files": list(HF_METADATA_FILES), "remote_videos": tree}
    (manifest_dir / "nexar_source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    write_manifest(_integrity(data), manifest_dir / "train.csv")
    if metadata_only:
        return manifest_dir / "train.csv"
    if not full:
        if previous is None:
            data = sample_subset(data, subset // 2, subset // 2, seed)
        else:
            expected = previous["video_id"].tolist()
            if len(expected) != subset or not set(expected).issubset(data["video_id"]):
                raise ValueError("Frozen subset IDs do not match the requested size / pinned metadata")
            data = data.set_index("video_id").loc[expected].reset_index()
    data["subset_seed"] = seed
    # Freeze IDs before any potentially interrupted video download.
    write_manifest(_integrity(data), output)
    try:
        snapshot_download(NEXAR_REPO_ID, repo_type="dataset", revision=commit, token=token,
                          local_dir=raw_dir, allow_patterns=data["source_path"].tolist(), max_workers=4)
    finally:
        # Any completed files remain usable, with integrity recorded even after a failure.
        write_manifest(_integrity(data), output)
    return output


def download_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download official Nexar train metadata or a balanced fixed subset")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--metadata-only", action="store_true")
    mode.add_argument("--subset", type=int, default=100)
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-dir", default="data/raw/nexar")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args(argv)
    try:
        output = download_nexar(**vars(args))
    except Exception as exc:
        retry_args = list(argv) if argv is not None else sys.argv[1:]
        actual_cli_call = len(sys.argv) > 1 and sys.argv[1] == "download" and retry_args == sys.argv[2:]
        if (argv is None or actual_cli_call) and hasattr(sys, "orig_argv"):
            command = shlex.join(sys.orig_argv)
        else:
            # Programmatic calls have no shell invocation; emit an equivalent script command.
            command = shlex.join([sys.executable, "scripts/download_nexar.py", *retry_args])
        message = str(exc)
        token = os.environ.get("HF_TOKEN")
        if token:
            message = message.replace(token, "[REDACTED]")
            command = command.replace(token, "[REDACTED]")
        print(f"Nexar download failed ({type(exc).__name__}): {message}", file=sys.stderr)
        print(f"Exact command: {command}", file=sys.stderr)
        print("Stopped. No alternate dataset was downloaded. See data/raw/README.md for manual Nexar placement.", file=sys.stderr)
        return 1
    print(f"Manifest: {output}")
    return 0
