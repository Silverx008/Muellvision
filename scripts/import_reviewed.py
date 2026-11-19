"""Bring reviewed labels into labels/ so the images count as labelled.

Review export from Label Studio (YOLO format):
  python scripts/import_reviewed.py --folder realwaste/Plastic --export path/to/ls_export
Accept proposals unreviewed (ONLY for an early pipeline test; mark the run as such in the notes):
  python scripts/import_reviewed.py --folder realwaste/Plastic --accept-auto
"""

import argparse
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401

from mv.data.labelstudio import check_classes
from mv.paths import data_root

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--folder", required=True, help="relative to DATA_ROOT, e.g. realwaste/Plastic")
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--export", type=Path)
g.add_argument("--accept-auto", action="store_true")
a = ap.parse_args()

folder = data_root() / a.folder
stems = {p.stem for p in (folder / "images").glob("*")}
dst = folder / "labels"
dst.mkdir(exist_ok=True)
if a.export:
    check_classes(a.export)
    src_dir = a.export / "labels"
    # LS prefixes exported names with a hash ("1a2b3c4d-<name>.txt"); match on the original stem
    n = 0
    for f in src_dir.glob("*.txt"):
        stem = f.stem
        if stem not in stems and "-" in stem:
            stem = stem.split("-", 1)[1]
        if stem in stems:
            shutil.copy2(f, dst / f"{stem}.txt")
            n += 1
    print(f"imported {n} reviewed label files into {dst}")
else:
    n = 0
    for f in (folder / "labels_auto").glob("*.txt"):
        shutil.copy2(f, dst / f.name)
        n += 1
    print(f"accepted {n} UNREVIEWED proposals into {dst}")
