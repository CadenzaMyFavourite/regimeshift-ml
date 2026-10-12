"""Small provenance and serialization helpers; no hidden experiment state."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, (Path,)):
            return str(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"Cannot serialize {type(item)}")

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def provenance() -> dict:
    packages = ["regimeshift-ml", "numpy", "polars", "pyarrow", "scikit-learn", "lightgbm",
                "scipy", "joblib", "threadpoolctl"]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    code_root = Path(__file__).parent
    lock = code_root.parents[1] / "uv.lock"
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "git_commit": commit,
        "dependency_lock_sha256": file_hash(lock) if lock.exists() else None,
        "source_hashes": {p.name: file_hash(p) for p in sorted(code_root.glob("*.py"))},
    }
