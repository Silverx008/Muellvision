"""Grouped split: own photos -> test set (used once), public data -> stratified grouped 5-fold CV.

Committed split files (``splits/<version>/``) hold image paths relative to DATA_ROOT, so the same split works
on Colab, the lab PC and the laptop. Ultralytics needs absolute paths, so :func:`materialize` writes absolute
lists plus a ``data.yaml`` under ``DATA_ROOT/_lists/<version>/`` right before training.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import yaml
from sklearn.model_selection import StratifiedGroupKFold

from mv.config import class_names
from mv.data.manifest import OWN, dominant_class
from mv.paths import SPLITS, data_root


def make_splits(rows: list[dict], n_folds: int = 5, seed: int = 42, stratify_source: bool = True) -> dict:
    """Returns {'test': [paths], 'folds': [(train_paths, val_paths), ...]} from labelled manifest rows."""
    rows = [r for r in rows if r["labelled"]]
    test = sorted(r["path"] for r in rows if r["split_role"] == "test")
    pool = [r for r in rows if r["split_role"] == "cv"]
    if any(r["source"] == OWN for r in pool):
        raise AssertionError("own photos must never be in the CV folds")
    if len({r["group"] for r in pool}) < n_folds:
        raise ValueError(f"need at least {n_folds} groups of labelled public data, got {len(pool)} images")

    groups = merge_duplicate_groups(pool)

    # StratifiedGroupKFold needs >= n_folds members per stratum; fold rare strata into the most common one.
    # stratify by dominant class (and source dataset, so every fold gets a similar dataset mix)
    y = [f"{r['source']}_{dominant_class(r)}" if stratify_source else str(dominant_class(r)) for r in pool]
    counts = Counter(y)
    common = counts.most_common(1)[0][0]
    y = [c if counts[c] >= n_folds else common for c in y]

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for tr, va in sgkf.split(pool, y, groups=groups):
        folds.append((sorted(pool[i]["path"] for i in tr), sorted(pool[i]["path"] for i in va)))
    out = {"test": test, "folds": folds}
    assert_no_leakage(rows, out)
    return out


def merge_duplicate_groups(rows: list[dict]) -> list[str]:
    """Group per row, with groups that share an identical image (sha1) merged, so exact duplicates
    (e.g. the same photo in two public datasets) can never sit on both sides of a fold."""
    parent: dict[str, str] = {}

    def find(g: str) -> str:
        while parent.setdefault(g, g) != g:
            g = parent[g]
        return g

    first: dict[str, str] = {}
    for r in rows:
        if r["sha1"]:
            if r["sha1"] in first:
                parent[find(r["group"])] = find(first[r["sha1"]])
            else:
                first[r["sha1"]] = r["group"]
    return [find(r["group"]) for r in rows]


def assert_no_leakage(rows: list[dict], split: dict) -> None:
    by_path = {r["path"]: r for r in rows}
    test = set(split["test"])
    test_sha = {by_path[p]["sha1"] for p in test if by_path[p]["sha1"]}
    test_groups = {by_path[p]["group"] for p in test}
    for k, (tr, va) in enumerate(split["folds"]):
        g_tr = {by_path[p]["group"] for p in tr}
        g_va = {by_path[p]["group"] for p in va}
        if g_tr & g_va:
            raise AssertionError(f"fold {k}: groups in train and val: {sorted(g_tr & g_va)[:5]}")
        s_tr = {by_path[p]["sha1"] for p in tr if by_path[p]["sha1"]}
        s_va = {by_path[p]["sha1"] for p in va if by_path[p]["sha1"]}
        if s_tr & s_va:
            raise AssertionError(f"fold {k}: {len(s_tr & s_va)} identical images in train and val")
        fold = set(tr) | set(va)
        if fold & test:
            raise AssertionError(f"fold {k}: test images inside the fold")
        if test_groups & (g_tr | g_va):
            raise AssertionError(f"fold {k}: test groups inside the fold")
        sha = {by_path[p]["sha1"] for p in fold if by_path[p]["sha1"]}
        if sha & test_sha:
            raise AssertionError(f"fold {k}: {len(sha & test_sha)} test images duplicated in the fold (sha1)")
    # every image in exactly one val fold
    vals = [p for _, va in split["folds"] for p in va]
    if len(vals) != len(set(vals)):
        raise AssertionError("an image is in more than one val fold")


def write_splits(split: dict, rows: list[dict], version: str, seed: int, out_root: Path = SPLITS) -> Path:
    out = out_root / version
    if (out / "FROZEN").exists():
        raise FileExistsError(f"{out} is frozen. Create a new version instead.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "test.txt").write_text("".join(p + "\n" for p in split["test"]))
    for k, (tr, va) in enumerate(split["folds"]):
        d = out / f"fold_{k}"
        d.mkdir(exist_ok=True)
        (d / "train.txt").write_text("".join(p + "\n" for p in tr))
        (d / "val.txt").write_text("".join(p + "\n" for p in va))
    (out / "split_report.md").write_text(report(split, rows, version, seed))
    return out


def report(split: dict, rows: list[dict], version: str, seed: int) -> str:
    by_path = {r["path"]: r for r in rows}
    names = class_names()

    def line(label: str, paths: list[str]) -> str:
        src = Counter(by_path[p]["source"] for p in paths)
        inst = [sum(by_path[p][f"n_{c}"] for p in paths) for c in names]
        groups = len({by_path[p]["group"] for p in paths})
        srcs = ", ".join(f"{k} {v}" for k, v in sorted(src.items()))
        return f"| {label} | {len(paths)} | {groups} | " + " | ".join(map(str, inst)) + f" | {srcs} |"

    head = "| part | images | groups | " + " | ".join(names) + " | sources |"
    sep = "|" + "---|" * (len(names) + 4)
    lines = [f"# Split {version}", "", f"seed {seed}, StratifiedGroupKFold, test = own photos only.", "",
             head, sep, line("test", split["test"])]
    for k, (tr, va) in enumerate(split["folds"]):
        lines += [line(f"fold {k} train", tr), line(f"fold {k} val", va)]
    lines += ["", "Leakage checks passed: no group or identical image in train and val of a fold,",
              "no test image in any fold, every public image in exactly one val fold.", ""]
    return "\n".join(lines)


def read_list(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if ln.strip()] if path.exists() else []


def materialize(version: str, part: str, root: Path | None = None) -> Path:
    """Write absolute image lists + data.yaml for one part and return the yaml path.

    part: 'fold_0'..'fold_4' (train/val of that fold), or 'final' (train = all folds, val = test, used ONCE
    by scripts/final_train_test.py).
    """
    root = root or data_root()
    src = SPLITS / version
    if part == "final":
        train = sorted({p for d in sorted(src.glob("fold_*")) for p in read_list(d / "val.txt")})
        val = read_list(src / "test.txt")
        if not val:
            raise ValueError("test.txt is empty: record and label own photos first")
    else:
        train, val = read_list(src / part / "train.txt"), read_list(src / part / "val.txt")
    if not train or not val:
        raise ValueError(f"split {version}/{part} is empty. Run scripts/make_splits.py first.")

    out = root / "_lists" / version / part
    out.mkdir(parents=True, exist_ok=True)
    for name, paths in (("train", train), ("val", val)):
        (out / f"{name}.txt").write_text("".join(str((root / p).resolve()) + "\n" for p in paths))
    names = class_names()
    data = {"path": str(root), "train": str(out / "train.txt"), "val": str(out / "val.txt"),
            "names": dict(enumerate(names))}
    yml = out / "data.yaml"
    yml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return yml


def instances_per_class(rows: list[dict]) -> dict[str, int]:
    tot: dict[str, int] = defaultdict(int)
    for r in rows:
        for c in class_names():
            tot[c] += r[f"n_{c}"]
    return dict(tot)
