"""TACO (http://tacodataset.org, annotations CC BY 4.0): download and convert COCO -> YOLO with configs/maps/taco.yaml.

Raw download: DATA_ROOT/_raw/taco/annotations.json + DATA_ROOT/_raw/taco/batch_N/*.jpg
Converted:    DATA_ROOT/taco/batch_N/images/batch_N_000006.jpg + labels/...
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mv.config import class_id, load_yaml
from mv.data.yolo import Box, coco_to_yolo, write_labels
from mv.paths import CONFIGS

MAP_FILE = CONFIGS / "maps" / "taco.yaml"


def download(raw: Path, workers: int = 8, limit: int | None = None) -> tuple[int, int]:
    """Fetch annotations.json and all images (skips files that already exist). Returns (ok, failed)."""
    import requests
    from tqdm import tqdm

    cfg = load_yaml(MAP_FILE)
    raw.mkdir(parents=True, exist_ok=True)
    ann_path = raw / "annotations.json"
    if not ann_path.exists():
        r = requests.get(cfg["annotations_url"], timeout=60)
        r.raise_for_status()
        ann_path.write_bytes(r.content)
    images = json.loads(ann_path.read_text())["images"][:limit]

    def fetch(img: dict) -> bool:
        dst = raw / img["file_name"]
        if dst.exists() and dst.stat().st_size > 0:
            return True
        dst.parent.mkdir(parents=True, exist_ok=True)
        for url in (img.get("flickr_url"), img.get("flickr_640_url")):
            if not url:
                continue
            try:
                r = requests.get(url, timeout=60)
                if r.ok and r.content:
                    dst.write_bytes(r.content)
                    return True
            except requests.RequestException:
                pass
        return False

    with ThreadPoolExecutor(workers) as ex:
        res = list(tqdm(ex.map(fetch, images), total=len(images), desc="TACO images"))
    return sum(res), len(res) - sum(res)


def build_category_map(categories: list[dict], name_map: dict[str, str]) -> dict[int, int]:
    """TACO category id -> Müllvision class id, for mapped categories only."""
    return {c["id"]: class_id(name_map[c["name"]]) for c in categories if c["name"] in name_map}


def convert_annotations(coco: dict, name_map: dict[str, str]) -> dict[str, list[Box]]:
    """file_name -> YOLO boxes (mapped categories only). Images with only unmapped objects get []."""
    cat = build_category_map(coco["categories"], name_map)
    imgs = {i["id"]: i for i in coco["images"]}
    out: dict[str, list[Box]] = {i["file_name"]: [] for i in coco["images"]}
    for a in coco["annotations"]:
        if a["category_id"] not in cat:
            continue
        im = imgs[a["image_id"]]
        b = coco_to_yolo(a["bbox"], im["width"], im["height"], cat[a["category_id"]])
        if b:
            out[im["file_name"]].append(b)
    return out


def convert(raw: Path, dst_root: Path, max_side: int | None = None) -> Counter:
    """Write images (EXIF-corrected, optionally downscaled) and labels to dst_root/taco/. Returns stats."""
    from PIL import Image, ImageOps
    from tqdm import tqdm

    cfg = load_yaml(MAP_FILE)
    max_side = max_side or cfg.get("max_side")
    coco = json.loads((raw / "annotations.json").read_text())
    labels = convert_annotations(coco, cfg["map"])
    sizes = {i["file_name"]: (i["width"], i["height"]) for i in coco["images"]}
    stats: Counter = Counter()
    per_class: dict[int, int] = defaultdict(int)
    for fname, boxes in tqdm(labels.items(), desc="convert TACO"):
        src = raw / fname
        if not src.exists():
            stats["missing"] += 1
            continue
        batch, stem = Path(fname).parent.name, Path(fname).stem
        out_img = dst_root / "taco" / batch / "images" / f"{batch}_{stem}.jpg"
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if im.size != sizes[fname]:
                stats["size_mismatch"] += 1  # annotation made on a different orientation: skip, don't guess
                continue
            if max_side and max(im.size) > max_side:
                im.thumbnail((max_side, max_side))
            out_img.parent.mkdir(parents=True, exist_ok=True)
            im.save(out_img, quality=92)
        write_labels(out_img.parent.parent / "labels" / f"{out_img.stem}.txt", boxes)
        stats["background" if not boxes else "with_objects"] += 1
        for b in boxes:
            per_class[b.cls] += 1
    stats.update({f"inst_{k}": v for k, v in per_class.items()})
    return stats
