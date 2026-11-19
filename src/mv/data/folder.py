"""Ingest a folder-per-class dataset (e.g. the local Muelvision_Dataset = RealWaste + TrashNet).

Copies images into DATA_ROOT/<source>/<Folder>/images/<source>_<folder>_<stem>.jpg. No labels are written:
these datasets only have image-level classes. Run scripts/prelabel.py next, then review in Label Studio.
"""

from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path

from mv.config import load_yaml
from mv.data.yolo import IMG_EXTS
from mv.paths import CONFIGS

MAP_FILE = CONFIGS / "maps" / "folder.yaml"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def detect_source(path: Path, sources: dict, default: str) -> str:
    """Identify the public source by image size, then file name pattern."""
    from PIL import Image

    with Image.open(path) as im:
        size = list(im.size)
    for name, spec in sources.items():
        if spec.get("size") == size and re.match(spec["pattern"], path.name):
            return name
    for name, spec in sources.items():
        if spec.get("size") == size:
            return name
    return default


def ingest(src: Path, dst_root: Path, default_source: str = "folderset", dry_run: bool = False) -> Counter:
    cfg = load_yaml(MAP_FILE)
    stats: Counter = Counter()
    for folder in sorted(p for p in src.iterdir() if p.is_dir()):
        spec = cfg["folders"].get(folder.name)
        if spec is None:
            stats[f"unmapped_folder:{folder.name}"] += 1
            continue
        if spec["role"] == "skip":
            continue
        for img in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS):
            source = detect_source(img, cfg.get("sources", {}), default_source)
            out = dst_root / source / folder.name / "images" / f"{source}_{slug(folder.name)}_{slug(img.stem)}.jpg"
            stats[f"{source}/{folder.name}"] += 1
            if dry_run or out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out)
    return stats
