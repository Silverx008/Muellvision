"""Check labels before splitting. Exit code 1 on errors.

  python scripts/check_labels.py
"""

import sys

import _bootstrap  # noqa: F401

from mv.config import class_names
from mv.data import manifest
from mv.data.checks import check_rows
from mv.paths import MANIFEST, data_root
from mv.splits import instances_per_class

rows = manifest.read(MANIFEST)
errors, warnings = check_rows(rows, data_root())
lab = [r for r in rows if r["labelled"]]
for role in ("cv", "test"):
    part = [r for r in lab if r["split_role"] == role]
    inst = instances_per_class(part)
    print(f"{role:5s} {len(part):6d} images  " + "  ".join(f"{c}={inst.get(c, 0)}" for c in class_names()))
for w in warnings:
    print("WARN ", w)
for e in errors[:200]:
    print("ERROR", e)
print(f"{len(errors)} errors, {len(warnings)} warnings")
sys.exit(1 if errors else 0)
