"""Ingest a Label Studio YOLO export of own photos as one recording series (test set only).

  python scripts/ingest_labelstudio.py --export path/to/export --series 20261204_band_tageslicht
"""

import argparse
import re
from pathlib import Path

import _bootstrap  # noqa: F401
from mv.data.labelstudio import ingest_series
from mv.paths import data_root

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--export", required=True, type=Path)
ap.add_argument("--series", required=True, help="YYYYMMDD_<szenario>_<licht>")
a = ap.parse_args()

if not re.fullmatch(r"\d{8}_[a-z0-9-]+_[a-z0-9-]+", a.series):
    raise SystemExit("series must look like 20261204_band_tageslicht (lowercase, no spaces)")
n = ingest_series(a.export, a.series, data_root())
print(f"{n} images -> {data_root() / 'own' / a.series}. Next: scripts/build_manifest.py")
