"""Convert downloaded TACO (COCO) to YOLO labels with configs/maps/taco.yaml into DATA_ROOT/taco/.

  python scripts/convert_taco.py [--max-side 1280]
"""

import argparse

import _bootstrap  # noqa: F401
from mv.config import class_names
from mv.data.taco import convert
from mv.paths import data_root

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--max-side", type=int)
a = ap.parse_args()

stats = convert(data_root() / "_raw" / "taco", data_root(), max_side=a.max_side)
names = class_names()
for k, v in sorted(stats.items()):
    if k.startswith("inst_"):
        k = "instances " + names[int(k[5:])]
    print(f"{k:32s} {v}")
