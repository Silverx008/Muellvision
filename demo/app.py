"""Live demo: webcam or video, boxes with class + Fraktion, Sortierrest rule, FPS overlay.

  python demo/app.py --weights best.pt --tau 0.5 --source 0
Frames where the model is unsure (conf between --conf-min and tau) are saved to DATA_ROOT/residual_queue/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from mv.config import class_names, classes, fraktion  # noqa: E402
from mv.paths import data_root  # noqa: E402
from mv.residual import decide  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--source", default="0", help="camera index or video file")
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--conf-min", type=float, default=0.15)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--device")
ap.add_argument("--no-queue", action="store_true")
a = ap.parse_args()

from ultralytics import YOLO  # noqa: E402

model = YOLO(a.weights)
names = class_names()
src = int(a.source) if a.source.isdigit() else a.source
cap = cv2.VideoCapture(src)
queue = data_root() / "residual_queue"
fps, t_prev = 0.0, time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    r = model.predict(frame, conf=a.conf_min, imgsz=a.imgsz, verbose=False, device=a.device)[0]
    unsure = False
    for box, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        d = decide(names, int(c), p, a.tau)
        x0, y0, x1, y1 = map(int, box)
        if d.label == "residual":
            unsure = True
            color, text = (0, 165, 255), f"{classes()['residual']} ({p:.2f})"
        else:
            color, text = (0, 200, 0), f"{d.label} -> {fraktion(d.label)} ({p:.2f})"
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        cv2.putText(frame, text, (x0, max(15, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    now = time.perf_counter()
    fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
    t_prev = now
    cv2.putText(frame, f"{fps:.1f} FPS  tau={a.tau}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if unsure and not a.no_queue:
        queue.mkdir(parents=True, exist_ok=True)
        stem = time.strftime("%Y%m%d_%H%M%S") + f"_{int(now * 1000) % 1000:03d}"
        cv2.imwrite(str(queue / f"{stem}.jpg"), r.orig_img)
        (queue / f"{stem}.json").write_text(json.dumps({"boxes": r.boxes.xyxy.tolist(), "cls": r.boxes.cls.tolist(),
                                                        "conf": r.boxes.conf.tolist(), "tau": a.tau}))
    cv2.imshow("Muellvision", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
        break
cap.release()
cv2.destroyAllWindows()
