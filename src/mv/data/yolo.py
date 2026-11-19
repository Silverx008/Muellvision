"""Read/write YOLO label files and image <-> label path mapping (Ultralytics layout: .../images/x.jpg -> .../labels/x.txt)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Box:
    cls: int
    xc: float
    yc: float
    w: float
    h: float

    def line(self) -> str:
        return f"{self.cls} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"


def label_path(img: Path) -> Path:
    parts = list(img.parts)
    idx = len(parts) - 1 - parts[::-1].index("images")  # last 'images' component, as Ultralytics does
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(path: Path) -> list[Box]:
    if not path.exists():
        return []
    boxes = []
    for ln in path.read_text().splitlines():
        if ln.strip():
            c, *v = ln.split()
            boxes.append(Box(int(c), *map(float, v[:4])))
    return boxes


def write_labels(path: Path, boxes: list[Box]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(b.line() + "\n" for b in boxes))


def coco_to_yolo(bbox: list[float], img_w: int, img_h: int, cls: int) -> Box | None:
    """COCO [x, y, w, h] in pixels -> normalised YOLO box, clipped to the image. None if degenerate."""
    x, y, w, h = bbox
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(img_w), x + w), min(float(img_h), y + h)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return Box(cls, (x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h, (x1 - x0) / img_w, (y1 - y0) / img_h)


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_images(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS and "images" in p.parts)
