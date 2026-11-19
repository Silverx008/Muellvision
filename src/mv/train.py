"""Thin wrapper around ultralytics YOLO.train() so every run is named, seeded and logged the same way."""

from __future__ import annotations

from pathlib import Path

from mv.config import config_hash, load_yaml
from mv.paths import CONFIGS, runs_root


def run_name(split: str, model: str, part: str, tag: str = "") -> str:
    p = part.replace("fold_", "f")
    return f"{split}_{model}_{p}" + (f"_{tag}" if tag else "")


def weights_for(model: str) -> str:
    variants = load_yaml(CONFIGS / "models.yaml")["variants"]
    return variants[model]["weights"] if model in variants else model


def train(model: str, data_yaml: Path, cfg_path: Path, name: str, device: str | None = None,
          overrides: dict | None = None, resume: bool = False) -> dict:
    from ultralytics import YOLO, __version__

    cfg = load_yaml(cfg_path)
    cfg.update(overrides or {})
    project = runs_root() / "detect"
    last = project / name / "weights" / "last.pt"
    if resume and last.exists():
        yolo = YOLO(str(last))
        yolo.train(resume=True)
    else:
        yolo = YOLO(weights_for(model))
        kw = dict(cfg, data=str(data_yaml), project=str(project), name=name, exist_ok=True)
        if device:
            kw["device"] = device
        yolo.train(**kw)
    best = project / name / "weights" / "best.pt"
    return {"run": name, "best": best, "cfg": cfg, "cfg_hash": config_hash(cfg), "ultralytics": __version__}


def is_finished(name: str) -> bool:
    return (runs_root() / "detect" / name / "weights" / "best.pt").exists() and (
        runs_root() / "detect" / name / "results.csv"
    ).exists()
