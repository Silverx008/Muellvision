import pytest

from mv.residual import choose_tau, decide, match_image, sweep

NAMES = ["pet_bottle", "plastic_film", "beverage_carton", "metal_can", "paper_cardboard"]


def test_decide_threshold():
    assert decide(NAMES, 3, 0.8, 0.5).label == "metal_can"
    assert decide(NAMES, 3, 0.4, 0.5).label == "residual"
    assert decide(NAMES, None, 0.0, 0.5).label == "residual"


def test_match_correct_wrong_fp_and_miss():
    gt = [[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]]
    gt_cls = [0, 1, 2]
    pr = [[0, 0, 10, 10], [20, 20, 30, 30], [80, 80, 90, 90]]
    ev = match_image(gt, gt_cls, pr, [0, 3, 1], [0.9, 0.7, 0.6])
    gts = [e for e in ev if e[0] == "gt"]
    assert sorted((e[1], e[2]) for e in gts) == [(0.0, False), (0.7, False), (0.9, True)]
    assert [e for e in ev if e[0] == "fp"] == [("fp", 0.6, False)]


def test_sweep_and_choose():
    ev = [("gt", 0.9, True), ("gt", 0.7, False), ("gt", 0.0, False), ("fp", 0.6, False)]
    rows = {r["tau"]: r for r in sweep(ev, taus=[0.5, 0.8])}
    assert rows[0.5]["correct_rate"] == pytest.approx(1 / 3) and rows[0.5]["wrong_rate"] == pytest.approx(2 / 3)
    assert rows[0.8]["wrong_rate"] == 0.0 and rows[0.8]["residual_rate"] == pytest.approx(2 / 3)
    assert choose_tau(list(rows.values()), max_wrong=0.1)["tau"] == 0.8


def test_empty_image_no_crash():
    assert match_image([], [], [], [], []) == []
