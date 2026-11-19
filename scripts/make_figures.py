"""Build Doku figures from results/*.csv (so every number in the Doku comes from code).

  python scripts/make_figures.py
"""

import pandas as pd

import _bootstrap  # noqa: F401
from mv.paths import RESULTS

FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

cv = RESULTS / "cv_summary.csv"
if cv.exists():
    df = pd.read_csv(cv)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(df["model"], df["mAP50_95_mean"], yerr=df["mAP50_95_std"], capsize=4, color="#4a6fa5")
    ax.set_ylabel("mAP50-95 (Mittelwert ± Std, 5 Folds)")
    ax.set_title("Kreuzvalidierung der Modellvarianten")
    fig.tight_layout()
    fig.savefig(FIG / "cv_map.png", dpi=200)
    print("-> figures/cv_map.png")

curves = RESULTS / "tau_curves.csv"
if curves.exists():
    df = pd.read_csv(curves)
    g = df.groupby("tau")[["correct_rate", "wrong_rate", "residual_rate"]].mean()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for col, lbl in (("correct_rate", "korrekt ausgeschleust"), ("wrong_rate", "Fehlausschleusung"),
                     ("residual_rate", "Sortierrest")):
        ax.plot(g.index, g[col], label=lbl)
    ax.set_xlabel("Schwellwert τ")
    ax.set_ylabel("Anteil")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "tau_sweep.png", dpi=200)
    print("-> figures/tau_sweep.png")
