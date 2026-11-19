"""Sortierrest rule and tau tuning.

Per detection: conf >= tau -> eject into the Fraktion of the predicted class; otherwise the object stays on the
belt and ends up as Sortierrest. tau is tuned per CV fold to maximise correct ejections while keeping wrong
ejections (contamination of a Fraktion) below a limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Decision:
    label: str        # class name or "residual"
    conf: float


def decide(cls_names: list[str], cls: int | None, conf: float, tau: float) -> Decision:
    if cls is None or conf < tau:
        return Decision("residual", conf)
    return Decision(cls_names[cls], conf)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes a (N,4) and b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_image(gt_xyxy, gt_cls, pr_xyxy, pr_cls, pr_conf, iou_thr: float = 0.5) -> list[tuple]:
    """Greedy match by confidence. Returns events (kind, conf, correct):
    ('gt', conf_of_matched_pred or 0, pred_class == gt_class) for every GT object and
    ('fp', conf, False) for every unmatched prediction (background or unlabelled Störstoff).
    """
    gt_xyxy, pr_xyxy = np.asarray(gt_xyxy, float).reshape(-1, 4), np.asarray(pr_xyxy, float).reshape(-1, 4)
    gt_cls, pr_cls, pr_conf = np.asarray(gt_cls), np.asarray(pr_cls), np.asarray(pr_conf, float)
    m = iou(pr_xyxy, gt_xyxy)
    used_gt: set[int] = set()
    gt_hit: dict[int, tuple[float, bool]] = {}
    events = []
    for p in np.argsort(-pr_conf):
        cand = [g for g in np.argsort(-m[p]) if m[p, g] >= iou_thr and g not in used_gt] if len(gt_cls) else []
        if cand:
            g = int(cand[0])
            used_gt.add(g)
            gt_hit[g] = (float(pr_conf[p]), bool(pr_cls[p] == gt_cls[g]))
        else:
            events.append(("fp", float(pr_conf[p]), False))
    for g in range(len(gt_cls)):
        conf, ok = gt_hit.get(g, (0.0, False))
        events.append(("gt", conf, ok))
    return events


def sweep(events: list[tuple], taus=None) -> list[dict]:
    """Rates per tau. correct = GT objects ejected into the right Fraktion / all GT objects.
    wrong = (GT ejected into a wrong Fraktion + false ejections) / all ejections.
    """
    taus = np.round(np.arange(0.05, 0.96, 0.05), 2) if taus is None else taus
    n_gt = sum(e[0] == "gt" for e in events) or 1
    rows = []
    for t in taus:
        ej = [e for e in events if e[1] >= t and e[1] > 0]
        correct = sum(e[0] == "gt" and e[2] for e in ej)
        wrong = len(ej) - correct
        rows.append({"tau": float(t), "correct_rate": correct / n_gt,
                     "wrong_rate": wrong / len(ej) if ej else 0.0,
                     "residual_rate": 1 - sum(e[0] == "gt" for e in ej) / n_gt, "ejected": len(ej)})
    return rows


def choose_tau(rows: list[dict], max_wrong: float = 0.05) -> dict:
    """Highest correct_rate with wrong_rate <= max_wrong; if none qualifies, the tau with the lowest wrong_rate."""
    ok = [r for r in rows if r["wrong_rate"] <= max_wrong and r["ejected"] > 0]
    if ok:
        return max(ok, key=lambda r: (r["correct_rate"], -r["tau"]))
    return min(rows, key=lambda r: (r["wrong_rate"], -r["correct_rate"]))
