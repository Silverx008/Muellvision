"""Propose boxes (YOLO-World) for folder datasets without boxes. Writes labels_auto/, never labels/.

  python scripts/prelabel.py --sources realwaste trashnet [--device 0]

Runs on CPU (~0.5 s/image on the laptop) or much faster on Colab. Then review in Label Studio
(docs/labelling_workflow.md) and bring the reviewed labels back with scripts/import_reviewed.py.
"""

import argparse

import _bootstrap  # noqa: F401

from mv.data.prelabel import prelabel_all
from mv.paths import data_root

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--sources", nargs="+", default=["realwaste", "trashnet"])
ap.add_argument("--weights", default="yolov8s-worldv2.pt")
ap.add_argument("--conf", type=float, default=0.05)
ap.add_argument("--device")
a = ap.parse_args()

log = prelabel_all(data_root(), a.sources, weights=a.weights, conf=a.conf, device=a.device)
print(f"log: {log}  (needs_attention=1 -> no or weak proposal, check these first)")
