"""Validation wrapper: metrics overall and per class, plus per-image speed from Ultralytics."""

from __future__ import annotations

from pathlib import Path

from mv.paths import runs_root


def validate(weights: Path, data_yaml: Path, imgsz: int = 640, device: str | None = None,
             split: str = "val", name: str | None = None) -> dict:
    from ultralytics import YOLO

    m = YOLO(str(weights))
    kw = dict(data=str(data_yaml), imgsz=imgsz, split=split, plots=True, verbose=False)
    if device:
        kw["device"] = device
    if name:
        kw.update(project=str(runs_root() / "val"), name=name, exist_ok=True)
    r = m.val(**kw)
    out = {"P": float(r.box.mp), "R": float(r.box.mr), "mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
           "infer_ms": float(r.speed.get("inference", float("nan")))}
    names = m.names
    for i, c in enumerate(r.box.ap_class_index):
        out[f"mAP50_{names[int(c)]}"] = float(r.box.ap50[i])
    return out
