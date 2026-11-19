"""Scan the data store and (re)write manifests/manifest.csv.

  python scripts/build_manifest.py [--no-sha1]
"""

import argparse
from collections import Counter

import _bootstrap  # noqa: F401
from mv.data import manifest
from mv.paths import MANIFEST, data_root

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--no-sha1", action="store_true", help="faster, but disables the duplicate check")
a = ap.parse_args()

rows = manifest.scan(data_root(), with_sha1=not a.no_sha1)
manifest.write(rows, MANIFEST)
by = Counter((r["source"], r["labelled"]) for r in rows)
for (src, lab), n in sorted(by.items()):
    print(f"{src:10s} {'labelled' if lab else 'unlabelled':10s} {n}")
print(f"-> {MANIFEST}")
