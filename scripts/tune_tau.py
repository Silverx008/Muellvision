"""Tune the Sortierrest threshold tau per CV fold on the val predictions -> results/tau.csv.

  python scripts/tune_tau.py --model yolo26n --split v1 --max-wrong 0.05
"""

import argparse
import csv
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from mv.data.yolo import label_path, read_labels
from mv.paths import RESULTS, SPLITS, data_root, runs_root
from mv.residual import choose_tau, match_image, sweep
from mv.splits import read_list
from mv.train import run_name

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", default="yolo26n")
ap.add_argument("--split", default="v1")
ap.add_argument("--max-wrong", type=float, default=0.05, help="max share of wrong ejections")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--device")
a = ap.parse_args()

from ultralytics import YOLO  # noqa: E402

root = data_root()
out_rows, curves = [], []
for fold_dir in sorted((SPLITS / a.split).glob("fold_*")):
    k = int(fold_dir.name.split("_")[1])
    w = runs_root() / "detect" / run_name(a.split, a.model, fold_dir.name) / "weights" / "best.pt"
    if not w.exists():
        print(f"missing {w}")
        continue
    model = YOLO(str(w))
    events = []
    paths = [str(root / p) for p in read_list(fold_dir / "val.txt")]
    for res in model.predict(paths, conf=0.01, imgsz=a.imgsz, stream=True, verbose=False, device=a.device):
        h, wd = res.orig_shape
        gt = read_labels(label_path(Path(res.path)))
        gt_xyxy = [[(b.xc - b.w / 2) * wd, (b.yc - b.h / 2) * h, (b.xc + b.w / 2) * wd, (b.yc + b.h / 2) * h]
                   for b in gt]
        events += match_image(gt_xyxy, [b.cls for b in gt], res.boxes.xyxy.cpu().numpy(),
                              res.boxes.cls.cpu().numpy().astype(int), res.boxes.conf.cpu().numpy())
    rows = sweep(events)
    best = choose_tau(rows, a.max_wrong)
    out_rows.append({"model": a.model, "fold": k, **best})
    curves += [{"model": a.model, "fold": k, **r} for r in rows]
    print(f"fold {k}: tau={best['tau']:.2f} correct={best['correct_rate']:.3f} wrong={best['wrong_rate']:.3f}")

if out_rows:
    taus = np.array([r["tau"] for r in out_rows])
    out_rows.append({"model": a.model, "fold": "mean", "tau": round(float(taus.mean()), 3)})
    out_rows.append({"model": a.model, "fold": "std", "tau": round(float(taus.std(ddof=1)) if len(taus) > 1 else 0, 3)})
    for fname, rows in (("tau.csv", out_rows), ("tau_curves.csv", curves)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(RESULTS / fname, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
    print(f"final tau (mean over folds): {taus.mean():.2f} -> results/tau.csv")
