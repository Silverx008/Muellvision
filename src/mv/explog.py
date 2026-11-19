"""Append one row per training/eval run to results/experiments.csv (the experiment log of the concept)."""

from __future__ import annotations

import csv
import datetime as dt
import platform
import subprocess
from pathlib import Path

from mv.paths import RESULTS

FIELDS = ["date", "run", "model", "fold", "split", "cfg_hash", "seed", "epochs", "imgsz", "git_commit",
          "ultralytics", "device", "P", "R", "mAP50", "mAP50_95", "infer_ms", "notes"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return platform.processor() or platform.machine()


def append(row: dict, path: Path = RESULTS / "experiments.csv") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    full = {"date": dt.datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
            "device": device_name(), **row}
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: full.get(k, "") for k in FIELDS})
