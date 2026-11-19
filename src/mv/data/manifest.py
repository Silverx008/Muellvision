"""Manifest: one row per image in the data store. Everything downstream (splits, checks, Doku tables) reads it.

Data store layout under DATA_ROOT (env MV_DATA)::

    own/<series>/images/<series>_0001.jpg      own photos = held-out TEST SET only
    own/<series>/labels/<series>_0001.txt      series = YYYYMMDD_<szenario>_<licht>
    taco/<batch>/images|labels/...             public, CV folds (grouped by TACO batch)
    realwaste/<Folder>/images|labels|labels_auto/...   public, CV folds (grouped per image)
    trashnet/<Folder>/images|labels|labels_auto/...

``labels/`` holds reviewed labels only. Pre-labels from scripts/prelabel.py go to ``labels_auto/`` and do not
count as labelled until a human has reviewed them and moved them into ``labels/``.
"""

from __future__ import annotations

import csv
from pathlib import Path

from mv.config import class_names
from mv.data.yolo import iter_images, label_path, read_labels, sha1

PUBLIC_SOURCES = {"taco", "realwaste", "trashnet"}
OWN = "own"


def _fields() -> list[str]:
    return ["image_id", "path", "source", "series", "group", "split_role", "scenario", "light", "labelled"] + [
        f"n_{c}" for c in class_names()
    ] + ["sha1"]


def parse_series(series: str) -> tuple[str, str]:
    """'20261125_band_tageslicht' -> ('band', 'tageslicht'). Unknown parts are empty."""
    parts = series.split("_")
    return (parts[1] if len(parts) > 1 else "", "_".join(parts[2:]) if len(parts) > 2 else "")


def group_of(source: str, series: str, image_id: str) -> str:
    """Grouping unit for the CV split: images of one group never end up on both sides of a fold.

    own: recording series; TACO: photo batch (same place/camera/session); RealWaste/TrashNet: no session
    info, so each image is its own group.
    """
    if source in (OWN, "taco"):
        return f"{source}:{series}"
    return f"{source}:{image_id}"


def scan(root: Path, with_sha1: bool = True) -> list[dict]:
    rows = []
    names = class_names()
    for source_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        source = source_dir.name
        if source not in PUBLIC_SOURCES | {OWN}:
            continue  # e.g. runs/, _lists/
        for img in iter_images(source_dir):
            rel = img.relative_to(root)
            series = rel.parts[1]
            lbl = label_path(img)
            boxes = read_labels(lbl)
            counts = [sum(b.cls == i for b in boxes) for i in range(len(names))]
            scen, light = parse_series(series) if source == OWN else ("", "")
            rows.append(
                {
                    "image_id": img.stem,
                    "path": rel.as_posix(),
                    "source": source,
                    "series": series,
                    "group": group_of(source, series, img.stem),
                    "split_role": "test" if source == OWN else "cv",
                    "scenario": scen,
                    "light": light,
                    "labelled": int(lbl.exists()),
                    **{f"n_{c}": n for c, n in zip(names, counts)},
                    "sha1": sha1(img) if with_sha1 else "",
                }
            )
    return rows


def write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_fields())
        w.writeheader()
        w.writerows(rows)


def read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["labelled"] = int(r["labelled"])
        for c in class_names():
            r[f"n_{c}"] = int(r[f"n_{c}"])
    return rows


def dominant_class(row: dict) -> int:
    """Class id with most instances, or len(classes) for an empty image (used to stratify splits)."""
    counts = [row[f"n_{c}"] for c in class_names()]
    return counts.index(max(counts)) if max(counts) > 0 else len(counts)
