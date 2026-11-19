"""Re-validate every finished CV run and aggregate mean ± std per variant -> results/cv_summary.csv (table 6).

  python scripts/eval_cv.py --split v1
"""

import argparse
import csv

import _bootstrap  # noqa: F401
import numpy as np

from mv.config import load_yaml
from mv.evaluate import validate
from mv.paths import CONFIGS, RESULTS, SPLITS, runs_root
from mv.splits import materialize
from mv.train import is_finished, run_name

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--split", default="v1")
ap.add_argument("--device")
a = ap.parse_args()

models = list(load_yaml(CONFIGS / "models.yaml")["variants"])
folds = sorted(int(p.name.split("_")[1]) for p in (SPLITS / a.split).glob("fold_*"))
per_run, summary = [], []
for m in models:
    vals = []
    for k in folds:
        name = run_name(a.split, m, f"fold_{k}")
        if not is_finished(name):
            print(f"missing {name}")
            continue
        r = validate(runs_root() / "detect" / name / "weights" / "best.pt", materialize(a.split, f"fold_{k}"),
                     device=a.device, name=f"{name}_eval")
        per_run.append({"model": m, "fold": k, **r})
        vals.append(r)
    if vals:
        row = {"model": m, "n_folds": len(vals)}
        for key in vals[0]:
            x = np.array([v.get(key, np.nan) for v in vals], float)
            std = float(np.nanstd(x, ddof=1)) if len(x) > 1 else 0.0
            row[f"{key}_mean"], row[f"{key}_std"] = round(float(np.nanmean(x)), 4), round(std, 4)
        summary.append(row)

RESULTS.mkdir(exist_ok=True)
for fname, rows in (("cv_runs.csv", per_run), ("cv_summary.csv", summary)):
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("model", "fold", "n_folds"), k))
        with open(RESULTS / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"-> results/{fname}")
