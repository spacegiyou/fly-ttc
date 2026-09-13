"""Configuration and paths, resolved against the project instead of the shell cwd."""

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    root = path.parent.parent if path.parent.name == "configs" else Path.cwd()
    for key, value in config["paths"].items():
        config["paths"][key] = str(resolve_path(value, root))
    return config


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else ((root or Path.cwd()) / path).resolve()
