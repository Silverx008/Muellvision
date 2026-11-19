import pytest

from mv.data import manifest
from mv.splits import assert_no_leakage, make_splits, materialize, write_splits


def test_split_roles_and_no_leakage(data_store):
    rows = manifest.scan(data_store)
    s = make_splits(rows, n_folds=5, seed=42)
    own = {r["path"] for r in rows if r["source"] == "own"}
    assert set(s["test"]) == own
    pub = {r["path"] for r in rows if r["source"] != "own"}
    vals = [p for _, va in s["folds"] for p in va]
    assert set(vals) == pub and len(vals) == len(pub)
    for tr, va in s["folds"]:
        assert not (set(tr) & own) and not (set(va) & own)


def test_taco_batch_never_split(data_store):
    rows = manifest.scan(data_store)
    s = make_splits(rows, n_folds=5, seed=1)
    for _, va in s["folds"]:
        batches = {p.split("/")[1] for p in va if p.startswith("taco/")}
        for b in batches:
            n_b = sum(1 for p in va if p.startswith(f"taco/{b}/"))
            assert n_b == 6


def test_leakage_detected(data_store):
    rows = manifest.scan(data_store)
    s = make_splits(rows)
    s["folds"][0][0].append(s["test"][0])
    with pytest.raises(AssertionError):
        assert_no_leakage(rows, s)


def test_own_photo_with_cv_role_rejected(data_store):
    rows = manifest.scan(data_store)
    rows[0]["split_role"] = "cv" if rows[0]["source"] == "own" else rows[0]["split_role"]
    if rows[0]["source"] == "own":
        with pytest.raises(AssertionError):
            make_splits(rows)


def test_write_and_materialize(data_store, tmp_path, monkeypatch):
    import mv.splits as sp

    rows = manifest.scan(data_store)
    s = make_splits(rows)
    out = write_splits(s, rows, "vt", 42, out_root=tmp_path / "splits")
    monkeypatch.setattr(sp, "SPLITS", tmp_path / "splits")
    yml = materialize("vt", "fold_0", root=data_store)
    lines = (yml.parent / "train.txt").read_text().splitlines()
    assert lines and all(line.startswith(str(data_store)) for line in lines)
    assert (out / "split_report.md").exists()
    final = materialize("vt", "final", root=data_store)
    assert len((final.parent / "val.txt").read_text().splitlines()) == len(s["test"])
    (out / "FROZEN").write_text("x")
    with pytest.raises(FileExistsError):
        write_splits(s, rows, "vt", 42, out_root=tmp_path / "splits")


def test_identical_images_share_a_fold(data_store):
    rows = manifest.scan(data_store)
    pub = [r for r in rows if r["source"] == "realwaste"]
    pub[0]["sha1"] = pub[-1]["sha1"]  # same photo in two folders
    s = make_splits(rows, seed=3)
    for _, va in s["folds"]:
        assert (pub[0]["path"] in va) == (pub[-1]["path"] in va)
