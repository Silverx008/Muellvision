"""Download TACO annotations and images into DATA_ROOT/_raw/taco (resumable; ~1500 images, a few GB).

  python scripts/download_taco.py [--limit 50]
"""

import argparse

import _bootstrap  # noqa: F401
from mv.data.taco import download
from mv.paths import data_root

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--limit", type=int, help="only the first N images (smoke test)")
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()

ok, failed = download(data_root() / "_raw" / "taco", workers=a.workers, limit=a.limit)
print(f"downloaded/present: {ok}, failed: {failed}. Re-run to retry failures.")
