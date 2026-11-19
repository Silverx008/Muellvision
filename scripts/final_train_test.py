"""Final model: retrain the chosen variant on ALL public data, then evaluate ONCE on the own-photo test set.

  python scripts/final_train_test.py --model yolo26n --split v1 --i-know-this-uses-the-test-set

The test set is used exactly once (project rule). The result goes to results/test_final.csv.
"""

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
from mv import explog
from mv.evaluate import validate
from mv.paths import CONFIGS, RESULTS
from mv.splits import materialize
from mv.train import run_name, train

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True)
ap.add_argument("--split", default="v1")
ap.add_argument("--cfg", type=Path, default=CONFIGS / "train" / "base.yaml")
ap.add_argument("--device")
ap.add_argument("--i-know-this-uses-the-test-set", dest="ok", action="store_true")
a = ap.parse_args()

out = RESULTS / "test_final.csv"
if not a.ok:
    raise SystemExit("Refusing: pass --i-know-this-uses-the-test-set (run this once, for the final model).")
if out.exists():
    raise SystemExit(f"{out} exists: the test set was already used. Do not evaluate on it again.")

data_yaml = materialize(a.split, "final")
name = run_name(a.split, a.model, "final")
# 'val' in the final yaml is the test set. With val=False Ultralytics does not validate during training (no
# epoch or checkpoint selection on the test set); it trains the fixed number of epochs and we evaluate last.pt.
res = train(a.model, data_yaml, a.cfg, name, device=a.device, overrides={"val": False})
last = Path(res["best"]).with_name("last.pt")
m = validate(last, data_yaml, imgsz=res["cfg"].get("imgsz", 640), device=a.device, name=f"{name}_test")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "split", "weights", *m])
    w.writeheader()
    w.writerow({"model": a.model, "split": a.split, "weights": str(last), **m})
explog.append({"run": name, "model": a.model, "fold": "test", "split": a.split, "cfg_hash": res["cfg_hash"],
               "ultralytics": res["ultralytics"], **{k: round(v, 4) for k, v in m.items()
                                                    if k in ("P", "R", "mAP50", "mAP50_95", "infer_ms")},
               "notes": "final model on own test set"})
print(m)
