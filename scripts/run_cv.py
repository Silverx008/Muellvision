"""Run all variants x folds sequentially; skips finished runs (safe to restart after a Colab disconnect).

  python scripts/run_cv.py --models all --folds all
  python scripts/run_cv.py --models yolo26n yolo11n --folds 0 1
"""

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from mv.config import load_yaml
from mv.paths import CONFIGS
from mv.train import is_finished, run_name

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--models", nargs="+", default=["all"])
ap.add_argument("--folds", nargs="+", default=["all"])
ap.add_argument("--split", default="v1")
ap.add_argument("--cfg", default=str(CONFIGS / "train" / "base.yaml"))
ap.add_argument("--device")
a, extra = ap.parse_known_args()

models = list(load_yaml(CONFIGS / "models.yaml")["variants"]) if a.models == ["all"] else a.models
n_folds = len(list((CONFIGS.parent / "splits" / a.split).glob("fold_*")))
folds = list(range(n_folds)) if a.folds == ["all"] else [int(f) for f in a.folds]
here = Path(__file__).parent
for m in models:
    for k in folds:
        name = run_name(a.split, m, f"fold_{k}")
        if is_finished(name):
            print(f"skip {name} (finished)")
            continue
        cmd = [sys.executable, str(here / "train.py"), "--model", m, "--fold", str(k), "--split", a.split,
               "--cfg", a.cfg, "--resume", *extra] + (["--device", a.device] if a.device else [])
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
