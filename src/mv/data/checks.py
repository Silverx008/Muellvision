"""Label sanity checks, run before every split (scripts/check_labels.py)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from mv.config import class_names
from mv.data.yolo import label_path

TARGET_PER_CLASS = 250
EMPTY_SHARE = (0.05, 0.10)


def check_label_file(path: Path, n_classes: int) -> list[str]:
    errors = []
    for i, ln in enumerate(path.read_text().splitlines(), 1):
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) != 5:
            errors.append(f"{path}:{i}: expected 5 values, got {len(parts)} (segments/OBB not supported)")
            continue
        try:
            c = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"{path}:{i}: not numeric")
            continue
        if not 0 <= c < n_classes:
            errors.append(f"{path}:{i}: class id {c} outside 0..{n_classes - 1}")
        if not all(0 < v <= 1 for v in (xc, yc, w, h)):
            errors.append(f"{path}:{i}: coordinates must be in (0, 1]")
        if w * h < 1e-6:
            errors.append(f"{path}:{i}: zero-size box")
    return errors


def check_rows(rows: list[dict], root: Path) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings) for labelled rows of the manifest."""
    n = len(class_names())
    errors, warnings = [], []
    labelled = [r for r in rows if r["labelled"]]
    for r in labelled:
        lbl = label_path(root / r["path"])
        errors += check_label_file(lbl, n)

    # duplicates are merged into one group by the split; reported so they can be cleaned up
    dup = Counter(r["sha1"] for r in rows if r["sha1"])
    for h, k in dup.items():
        if k > 1:
            ids = [r["image_id"] for r in rows if r["sha1"] == h]
            warnings.append(f"exact duplicate images ({k}x): {', '.join(ids[:5])}")

    unl = len(rows) - len(labelled)
    if unl:
        warnings.append(f"{unl} images without reviewed labels are ignored by the split")

    own = [r for r in labelled if r["split_role"] == "test"]
    dup_cross = {r["sha1"] for r in own} & {r["sha1"] for r in labelled if r["split_role"] == "cv"} - {""}
    if dup_cross:
        errors.append(f"{len(dup_cross)} test images also appear in the public CV data")
    for c in class_names():
        tot = sum(r[f"n_{c}"] for r in own)
        if tot < TARGET_PER_CLASS:
            warnings.append(f"test set (own photos): {c} has {tot} instances (target {TARGET_PER_CLASS})")
    if own:
        empty = sum(all(r[f"n_{c}"] == 0 for c in class_names()) for r in own) / len(own)
        if not EMPTY_SHARE[0] <= empty <= EMPTY_SHARE[1]:
            warnings.append(f"test set (own photos): {empty:.1%} empty images (target 5-10 %)")
    return errors, warnings
