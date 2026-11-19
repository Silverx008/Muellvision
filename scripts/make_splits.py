"""Write splits/<version>/: test.txt (own photos) + fold_0..4/{train,val}.txt (public data) + split_report.md.

  python scripts/make_splits.py --version v1 --seed 42
  python scripts/make_splits.py --version v1 --freeze      # at M3: no further changes to v1
"""

import argparse

import _bootstrap  # noqa: F401

from mv.data import manifest
from mv.paths import MANIFEST, SPLITS
from mv.splits import make_splits, write_splits

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--version", default="v1")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--freeze", action="store_true", help="mark an existing split version as frozen")
a = ap.parse_args()

if a.freeze:
    (SPLITS / a.version / "FROZEN").write_text("Frozen. Create a new version for any change.\n")
    print(f"froze splits/{a.version}")
    raise SystemExit
rows = manifest.read(MANIFEST)
split = make_splits(rows, n_folds=a.folds, seed=a.seed)
out = write_splits(split, rows, a.version, a.seed)
print((out / "split_report.md").read_text())
if not split["test"]:
    print("NOTE: test.txt is empty (no own photos yet). Fine for CV; needed before final_train_test.py.")
