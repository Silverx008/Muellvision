"""Ingest a Label Studio 'YOLO' export (images/, labels/, classes.txt) as one own recording series (test set)."""

from __future__ import annotations

import shutil
from pathlib import Path

from mv.config import class_names
from mv.data.yolo import IMG_EXTS


def check_classes(export: Path) -> None:
    f = export / "classes.txt"
    if not f.exists():
        return
    got = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
    if got != class_names():
        raise ValueError(f"classes.txt order {got} != configs/classes.yaml {class_names()}. Fix the LS config.")


def ingest_series(export: Path, series: str, dst_root: Path, source: str = "own") -> int:
    check_classes(export)
    imgs = sorted(p for p in (export / "images").rglob("*") if p.suffix.lower() in IMG_EXTS)
    out = dst_root / source / series
    if (out / "images").exists() and any((out / "images").iterdir()):
        raise FileExistsError(f"{out} already has images. Use a new series name.")
    for k, img in enumerate(imgs, 1):
        stem = f"{series}_{k:04d}"
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out / "images" / f"{stem}{img.suffix.lower()}")
        lbl = export / "labels" / f"{img.stem}.txt"
        # Label Studio writes no file for an image without boxes: an empty file = reviewed background image
        (out / "labels" / f"{stem}.txt").write_text(lbl.read_text() if lbl.exists() else "")
    return len(imgs)
