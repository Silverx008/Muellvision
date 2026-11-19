"""Arduino Uno Q, Linux side (own scope, milestone E2/E3).

USB camera -> NCNN INT8 model -> Sortierrest rule -> Fraktion per object -> STM32 via Bridge (or serial fallback).
  python main.py --model best_ncnn_model --imgsz 320 --tau 0.5
Status: skeleton. Verify the Bridge API on the board first (E1) and fill in send_to_mcu().
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import cv2  # noqa: E402

from mv.config import class_names  # noqa: E402
from mv.residual import decide  # noqa: E402


def send_to_mcu(label: str) -> None:
    """TODO(E3): call the Uno Q Bridge RPC (or write to the serial port) with the Fraktion index."""
    print("->", label)


ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="NCNN export dir (*_ncnn_model) or .tflite")
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--imgsz", type=int, default=320)
ap.add_argument("--tau", type=float, default=0.5)
a = ap.parse_args()

from ultralytics import YOLO  # noqa: E402

model = YOLO(a.model, task="detect")
names = class_names()
cap = cv2.VideoCapture(a.camera)
while True:
    ok, frame = cap.read()
    if not ok:
        time.sleep(0.1)
        continue
    t0 = time.perf_counter()
    r = model.predict(frame, imgsz=a.imgsz, conf=0.1, verbose=False)[0]
    for c, p in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        send_to_mcu(decide(names, int(c), p, a.tau).label)
    print(f"{(time.perf_counter() - t0) * 1000:.0f} ms, {len(r.boxes)} objects")
