"""Measure inference time on this machine and append to results/bench.csv.

  python scripts/bench.py --weights best.pt --images path/to/folder [--device 0 --half]
Runs unchanged on the Uno Q (with an NCNN export: --weights best_ncnn_model --imgsz 320).
"""

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401

from mv.bench import benchmark
from mv.explog import device_name
from mv.paths import RESULTS

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--weights", required=True)
ap.add_argument("--images", required=True, type=Path, help="folder with a few sample frames (val images)")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--device")
ap.add_argument("--half", action="store_true")
ap.add_argument("--platform", default="", help="label for the Doku table, e.g. 'Colab T4', 'Uno Q'")
a = ap.parse_args()

import cv2  # noqa: E402

frames = [cv2.imread(str(p)) for p in sorted(a.images.glob("*.jpg"))[:50]]
if not frames:
    raise SystemExit(f"no .jpg in {a.images}")
stats = benchmark(a.weights, frames, a.imgsz, a.warmup, a.n, a.device, a.half)
row = {"platform": a.platform or device_name(), **stats}
out = RESULTS / "bench.csv"
new = not out.exists()
with open(out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if new:
        w.writeheader()
    w.writerow(row)
print(row)
