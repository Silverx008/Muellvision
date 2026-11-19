"""Copy a local folder-per-class dataset (e.g. Muelvision_Dataset) into the data store.

  python scripts/ingest_folder.py --src "C:\\Users\\schueler\\Downloads\\Muelvision_Dataset"
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from mv.data.folder import ingest
from mv.paths import data_root

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", required=True, type=Path)
ap.add_argument("--dry-run", action="store_true", help="only count, copy nothing")
a = ap.parse_args()

stats = ingest(a.src, data_root(), dry_run=a.dry_run)
for k, v in sorted(stats.items()):
    print(f"{k:40s} {v}")
print(f"\n-> {data_root()}  (next: scripts/prelabel.py, then review in Label Studio)")
