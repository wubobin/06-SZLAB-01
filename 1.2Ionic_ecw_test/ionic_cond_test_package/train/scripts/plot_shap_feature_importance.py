"""
plot_shap_feature_importance.py
==================================================
Standalone script to plot SHAP feature importance bar chart.

Reads  : reports/shap_importance_20feat.csv
Outputs: figures/shap_feature_importance.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
SHAP_CSV = ROOT / "reports" / "shap_importance_20feat.csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = FIG_DIR / "shap_feature_importance.png"

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.dpi": 400,
})

# Dark-to-light blue gradient (top = most important = darkest)
DARK_BLUE = "#1a5276"
LIGHT_BLUE = "#a9cce3"


def main() -> None:
    # 1. Load data
    df = pd.read_csv(SHAP_CSV, encoding="utf-8-sig")
    df = df.sort_values("mean_abs_shap", ascending=True).reset_index(drop=True)

    features = df["feature"].tolist()
    values = df["mean_abs_shap"].tolist()
    n = len(features)

    # 2. Build color gradient (bottom of chart = least important = lightest)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "shap_blue", [LIGHT_BLUE, DARK_BLUE]
    )
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    # 3. Plot
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.barh(
        range(n), values,
        color=colors,
        edgecolor="white",
        linewidth=0.4,
        height=0.75,
    )
    ax.set_yticks(range(n))
    ax.set_yticklabels(features, fontsize=10)

    ax.set_xlabel("mean |Shapley value|", fontsize=12)
    ax.set_xlim(left=0)

    # Subtle x-axis grid only
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    # Remove top/right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] SHAP feature importance plot saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
