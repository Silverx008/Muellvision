"""Train one variant on one CV fold and log it to results/experiments.csv.

Colab / lab PC (CUDA):   python scripts/train.py --model yolo26n --fold 0
Laptop smoke test (CPU): python scripts/train.py --model yolo26n --fold 0 --smoke
Tuning experiment:       python scripts/train.py --model yolo26n --fold 0 --cfg configs/train/aug_1.yaml --tag aug1
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from mv import explog
from mv.evaluate import validate
from mv.paths import CONFIGS
from mv.splits import materialize
from mv.train import run_name, train

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", default="yolo26n", help="variant from configs/models.yaml or a .pt path")
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--split", default="v1")
ap.add_argument("--cfg", type=Path, default=CONFIGS / "train" / "base.yaml")
ap.add_argument("--tag", default="")
ap.add_argument("--device", help="0 for the first GPU, cpu, ... (default: Ultralytics picks)")
ap.add_argument("--smoke", action="store_true", help="tiny CPU run with configs/train/smoke.yaml")
ap.add_argument("--resume", action="store_true", help="continue an interrupted run (Colab disconnects)")
ap.add_argument("--notes", default="")
a = ap.parse_args()

if a.smoke:
    a.cfg, a.tag, a.device = CONFIGS / "train" / "smoke.yaml", a.tag or "smoke", a.device or "cpu"
part = f"fold_{a.fold}"
data_yaml = materialize(a.split, part)
name = run_name(a.split, Path(a.model).stem, part, a.tag)
res = train(a.model, data_yaml, a.cfg, name, device=a.device, resume=a.resume)
m = validate(res["best"], data_yaml, imgsz=res["cfg"].get("imgsz", 640), device=a.device, name=f"{name}_val")
explog.append({"run": name, "model": Path(a.model).stem, "fold": a.fold, "split": a.split,
               "cfg_hash": res["cfg_hash"], "seed": res["cfg"].get("seed"), "epochs": res["cfg"].get("epochs"),
               "imgsz": res["cfg"].get("imgsz"), "ultralytics": res["ultralytics"],
               **{k: round(v, 4) for k, v in m.items() if k in ("P", "R", "mAP50", "mAP50_95", "infer_ms")},
               "notes": a.notes or ("smoke" if a.smoke else "")})
print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
