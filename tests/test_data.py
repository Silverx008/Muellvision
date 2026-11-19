import json

import pytest
from PIL import Image

from mv.config import class_names
from mv.data import manifest
from mv.data.checks import check_label_file
from mv.data.folder import ingest
from mv.data.taco import convert_annotations
from mv.data.yolo import Box, coco_to_yolo, label_path


def test_classes_order():
    assert class_names() == ["pet_bottle", "plastic_film", "beverage_carton", "metal_can", "paper_cardboard"]


def test_label_path_uses_last_images_dir(tmp_path):
    p = tmp_path / "images" / "taco" / "b" / "images" / "x.jpg"
    assert label_path(p) == tmp_path / "images" / "taco" / "b" / "labels" / "x.txt"


def test_coco_to_yolo_clips_and_rejects():
    b = coco_to_yolo([90, 40, 20, 20], 100, 100, 3)
    assert b == Box(3, 0.95, 0.5, 0.1, 0.2)
    assert coco_to_yolo([10, 10, 0.5, 5], 100, 100, 0) is None


def test_taco_mapping_drops_unmapped_and_keeps_background():
    coco = {
        "categories": [{"id": 1, "name": "Drink can"}, {"id": 2, "name": "Cigarette"}],
        "images": [{"id": 10, "file_name": "batch_1/a.jpg", "width": 100, "height": 50},
                   {"id": 11, "file_name": "batch_1/b.jpg", "width": 100, "height": 50}],
        "annotations": [{"image_id": 10, "category_id": 1, "bbox": [0, 0, 50, 50]},
                        {"image_id": 10, "category_id": 2, "bbox": [60, 0, 10, 10]},
                        {"image_id": 11, "category_id": 2, "bbox": [0, 0, 10, 10]}],
    }
    out = convert_annotations(coco, {"Drink can": "metal_can"})
    assert out["batch_1/a.jpg"] == [Box(3, 0.25, 0.5, 0.5, 1.0)]
    assert out["batch_1/b.jpg"] == []  # only unmapped objects -> background image
    json.dumps(coco)


def test_check_label_file(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("0 0.5 0.5 0.2 0.2\n7 0.5 0.5 0.2 0.2\n1 0 0.5 0.2 0.2\n2 0.5 0.5\n")
    errs = check_label_file(f, 5)
    assert len(errs) == 3


def test_manifest_scan_roles_and_groups(data_store):
    rows = manifest.scan(data_store)
    own = [r for r in rows if r["source"] == "own"]
    assert own and all(r["split_role"] == "test" for r in own)
    assert all(r["split_role"] == "cv" for r in rows if r["source"] != "own")
    assert {r["group"] for r in rows if r["source"] == "taco"} == {"taco:batch_0", "taco:batch_1", "taco:batch_2"}
    rw = [r for r in rows if r["source"] == "realwaste"]
    assert len({r["group"] for r in rw}) == len(rw)
    assert own[0]["scenario"] == "band"


def test_manifest_roundtrip(data_store, tmp_path):
    rows = manifest.scan(data_store)
    manifest.write(rows, tmp_path / "m.csv")
    back = manifest.read(tmp_path / "m.csv")
    assert back[0]["n_pet_bottle"] == rows[0]["n_pet_bottle"] and back[0]["labelled"] == 1


@pytest.mark.parametrize("size,name,expected", [((524, 524), "Plastic_1.jpg", "realwaste"),
                                                ((512, 384), "plastic1.jpg", "trashnet")])
def test_folder_ingest_detects_source(tmp_path, size, name, expected):
    src = tmp_path / "src" / "Plastic"
    src.mkdir(parents=True)
    Image.new("RGB", size).save(src / name)
    stats = ingest(tmp_path / "src", tmp_path / "data")
    assert stats[f"{expected}/Plastic"] == 1
    assert len(list((tmp_path / "data" / expected / "Plastic" / "images").glob("*.jpg"))) == 1
