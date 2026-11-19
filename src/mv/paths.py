"""Central paths. Images live outside git under DATA_ROOT (env MV_DATA), code and small tables in the repo."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
SPLITS = REPO_ROOT / "splits"
RESULTS = REPO_ROOT / "results"
MANIFEST = REPO_ROOT / "manifests" / "manifest.csv"


def data_root() -> Path:
    """Root of the image store.

    Lab PC / laptop: e.g. ``D:/muellvision_data``; Colab: ``/content/drive/MyDrive/muellvision_data``.
    Defaults to ``<repo>/data`` (git-ignored).
    """
    return Path(os.environ.get("MV_DATA", REPO_ROOT / "data")).expanduser().resolve()


def runs_root() -> Path:
    """Where Ultralytics writes runs. Env MV_RUNS, default ``<data_root>/runs`` so Colab runs survive on Drive."""
    return Path(os.environ.get("MV_RUNS", data_root() / "runs")).expanduser().resolve()
