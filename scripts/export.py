"""Export a trained model for the edge. INT8 calibration uses the training images of a fold.

  python scripts/export.py --weights best.pt --format ncnn --imgsz 320 --int8 --split v1 --fold 0
Then compare FP32@640, FP32@320 and INT8@320 on the same val data (scripts/eval_cv.py style) for the Doku.
"""

import argparse

import _bootstrap  # noqa: F401

from mv.splits import materialize

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--weights", required=True)
ap.add_argument("--format", default="ncnn", choices=["ncnn", "tflite", "onnx", "openvino"])
ap.add_argument("--imgsz", type=int, default=320)
ap.add_argument("--int8", action="store_true")
ap.add_argument("--split", default="v1")
ap.add_argument("--fold", type=int, default=0)
a = ap.parse_args()

from ultralytics import YOLO  # noqa: E402

kw = dict(format=a.format, imgsz=a.imgsz)
if a.int8:
    kw.update(int8=True, data=str(materialize(a.split, f"fold_{a.fold}")))
path = YOLO(a.weights).export(**kw)
print(f"exported -> {path}")
