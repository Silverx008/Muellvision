"""Inference timing: batch 1, warm-up, then median/p95 of pre/inference/post (Ultralytics) and end-to-end."""

from __future__ import annotations

import platform
import time

import numpy as np


def benchmark(weights: str, frames: list, imgsz: int = 640, warmup: int = 50, n: int = 500,
              device: str | None = None, half: bool = False) -> dict:
    from ultralytics import YOLO, __version__

    model = YOLO(weights)
    kw = dict(imgsz=imgsz, verbose=False, half=half)
    if device:
        kw["device"] = device
    for i in range(warmup):
        model.predict(frames[i % len(frames)], **kw)
    pre, inf, post, e2e = [], [], [], []
    for i in range(n):
        t0 = time.perf_counter()
        r = model.predict(frames[i % len(frames)], **kw)[0]
        _ = r.plot()  # drawing is part of the demo loop
        e2e.append((time.perf_counter() - t0) * 1000)
        pre.append(r.speed["preprocess"])
        inf.append(r.speed["inference"])
        post.append(r.speed["postprocess"])

    def stats(x):
        return round(float(np.median(x)), 2), round(float(np.percentile(x, 95)), 2)

    out = {"weights": weights, "imgsz": imgsz, "half": half, "n": n, "ultralytics": __version__,
           "host": platform.node(), "cpu": platform.processor() or platform.machine()}
    for k, v in (("pre", pre), ("infer", inf), ("post", post), ("e2e", e2e)):
        out[f"{k}_ms_median"], out[f"{k}_ms_p95"] = stats(v)
    out["fps"] = round(1000 / out["e2e_ms_median"], 1)
    out["meets_33ms"] = out["e2e_ms_median"] < 33
    return out
