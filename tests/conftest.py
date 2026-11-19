import pytest
from PIL import Image

from mv.config import class_names
from mv.data.yolo import Box, write_labels


@pytest.fixture
def data_store(tmp_path):
    """Tiny data store: 2 own series (test), 3 TACO batches, 30 RealWaste images."""
    root = tmp_path / "data"
    k = 0

    def add(source, series, n, cls_cycle):
        nonlocal k
        for i in range(n):
            img = root / source / series / "images" / f"{source}_{series}_{i:03d}.jpg"
            img.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), (k % 255, (k * 7) % 255, (k * 13) % 255)).save(img)
            k += 1
            c = cls_cycle[i % len(cls_cycle)]
            boxes = [] if c is None else [Box(c, 0.5, 0.5, 0.4, 0.4)]
            write_labels(img.parent.parent / "labels" / f"{img.stem}.txt", boxes)

    n = len(class_names())
    add("own", "20261204_band_tageslicht", 4, [0, 1])
    add("own", "20261205_band_kunstlicht", 4, [2, 3, None])
    for b in range(3):
        add("taco", f"batch_{b}", 6, list(range(n)))
    add("realwaste", "Plastic", 15, [0, 1])
    add("realwaste", "Cardboard", 15, [4])
    return root
