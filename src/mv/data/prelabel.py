"""Box proposals for images that only have an image-level class (RealWaste/TrashNet folders).

Uses an open-vocabulary detector (Ultralytics YOLO-World) prompted with text for the candidate classes of each
folder (configs/maps/folder.yaml). Proposals go to ``labels_auto/`` and are NOT training labels until a human
has reviewed them in Label Studio and the reviewed file is in ``labels/``.
"""

from __future__ import annotations

import csv
from pathlib import Path

from mv.config import class_id, load_yaml
from mv.data.folder import MAP_FILE
from mv.data.yolo import Box, write_labels

PROMPTS: dict[str, list[str]] = {
    "pet_bottle": ["plastic bottle", "PET bottle"],
    "plastic_film": ["plastic bag", "plastic wrapper", "plastic film"],
    "beverage_carton": ["drink carton", "milk carton", "juice carton"],
    "metal_can": ["metal can", "aluminium can", "tin can"],
    "paper_cardboard": ["cardboard box", "cardboard", "sheet of paper", "newspaper"],
}


def prompt_table(candidates: list[str]) -> tuple[list[str], list[int]]:
    texts, cls = [], []
    for name in candidates:
        for t in PROMPTS[name]:
            texts.append(t)
            cls.append(class_id(name))
    return texts, cls


def prelabel_folder(folder: Path, candidates: list[str], weights: str = "yolov8s-worldv2.pt",
                    conf: float = 0.05, max_boxes: int = 3, device: str | None = None) -> list[dict]:
    """Writes labels_auto/*.txt for every image in folder/images. Returns one log row per image."""
    from ultralytics import YOLOWorld

    all_classes = list(PROMPTS)
    texts, cls_of = prompt_table(candidates or all_classes)
    model = YOLOWorld(weights)
    model.set_classes(texts)
    log = []
    imgs = sorted((folder / "images").glob("*.jpg"))
    for res in model.predict(source=[str(p) for p in imgs], conf=conf, stream=True, verbose=False,
                             device=device, imgsz=640):
        p = Path(res.path)
        boxes = []
        if res.boxes is not None and len(res.boxes):
            order = res.boxes.conf.argsort(descending=True)[:max_boxes]
            for i in order.tolist():
                xc, yc, w, h = res.boxes.xywhn[i].tolist()
                boxes.append((float(res.boxes.conf[i]), Box(cls_of[int(res.boxes.cls[i])], xc, yc, w, h)))
        # negative folders: only keep confident proposals, the rest becomes an empty (background) label
        if not candidates:
            boxes = [b for b in boxes if b[0] >= 0.3]
        write_labels(folder / "labels_auto" / f"{p.stem}.txt", [b for _, b in boxes])
        log.append({"image": p.name, "n_boxes": len(boxes), "top_conf": round(boxes[0][0], 3) if boxes else 0.0,
                    "needs_attention": int(bool(candidates) and (not boxes or boxes[0][0] < 0.2))})
    return log


def prelabel_all(root: Path, sources: list[str], **kw) -> Path:
    cfg = load_yaml(MAP_FILE)
    rows = []
    for source in sources:
        for folder in sorted(p for p in (root / source).iterdir() if p.is_dir()):
            spec = cfg["folders"].get(folder.name)
            if not spec or spec["role"] == "skip":
                continue
            for r in prelabel_folder(folder, spec["candidates"], **kw):
                rows.append({"source": source, "folder": folder.name, **r})
    out = root / "_prelabel_log.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["source"])
        w.writeheader()
        w.writerows(rows)
    return out
