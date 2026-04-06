# Ionic Conductivity (log10) - Prediction Visualization (Aligned Axes + Test-only Metrics)

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from pathlib import Path

# Matplotlib settings
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['font.size'] = 22

# -------- Settings --------
SAVE_FIG = True
SCATTER_FNAME = str(Path(r"D:\workspace\test_dagang\test_ionic_0304_final\ionic_cond_20feat\reports\scatter_pred_vs_true_styled1.png"))
DATA_CSV      = Path(r"D:\workspace\test_dagang\test_ionic_0304_final\ionic_cond_20feat\data\processed\features_20feat.csv")
SUMMARY_JSON  = Path(r"D:\workspace\test_dagang\test_ionic_0304_final\ionic_cond_20feat\reports\model_summary.json")

FEAT_20 = [
    "n_elements", "x_li", "wavg_AtomicNumber", "wavg_AtomicWeight",
    "wavg_CovalentRadius", "ratio_CationToAnionIonicPotential", "x_HalideTotal", "anion_MixEntropy",
    "AnionCation_PhiGap", "LiPhi_Coupling", "LiHalide_Coupling",
    "Density", "VolumePerAtom", "SpaceGroupNumber",
    "LatticeA", "LatticeB", "LatticeC",
    "LatticeAlpha", "LatticeBeta", "LatticeGamma"
]
TARGET = "log10_ionic_conductivity"

# -------- Load data & rebuild model --------
with open(SUMMARY_JSON) as f:
    summary = json.load(f)

best_params = summary["best_params"]
split_seed  = summary["split_seed"]

df = pd.read_csv(DATA_CSV)

X = df[FEAT_20].values
y = df[TARGET].values

imputer = SimpleImputer(strategy="median")
X_imp   = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y, test_size=0.2, random_state=split_seed
)

model = XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    random_state=split_seed,
    **best_params
)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)

# Metrics (test only for annotation)
test_r2   = r2_score(y_test, y_test_pred)
test_rmse = float(np.sqrt(np.mean((y_test - y_test_pred) ** 2)))
test_mae  = float(np.mean(np.abs(y_test - y_test_pred)))

print(f"Loaded {len(df)} samples")
print(f"Train samples: {len(y_train)}")
print(f"Test  samples: {len(y_test)}")
print(f"Test  - R2: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

# ---------------------------
# Predicted vs True scatter
# ---------------------------
plt.figure(figsize=(8, 8))

plt.scatter(y_train, y_train_pred,
            alpha=0.6, s=60, label=f'Train (n={len(y_train)})')
plt.scatter(y_test,  y_test_pred,
            alpha=0.7, s=60, label=f'Test (n={len(y_test)})')

ax = plt.gca()

# 固定范围
ax.set_xlim(-5, 2)
ax.set_ylim(-5, 2)

# 刻度
ticks = np.arange(-5, 3, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)

# -5 和 2 留空，其余显示标签
xlabels = [str(t) if -4 <= t <= 1 else "" for t in ticks]
ylabels = [str(t) if -4 <= t <= 1 else "" for t in ticks]
ax.set_xticklabels(xlabels)
ax.set_yticklabels(ylabels)

# 等比例坐标
ax.set_aspect('equal', adjustable='box')

# y=x 参考线
ax.plot([-5, 2], [-5, 2], '--', linewidth=2, alpha=0.8, label='y = x')

ax.set_xlabel(r'True $\log_{10}(\sigma)$',
              fontsize=26, fontweight='bold')
ax.set_ylabel(r'Predicted $\log_{10}(\sigma)$',
              fontsize=26, fontweight='bold')
# 无标题

# 只标注测试集指标（两行，无文本框）
#textstr = f'Test $R^2$ = {test_r2:.3f}\nTest RMSE = {test_rmse:.3f}\nTest MAE = {test_mae:.3f}'
textstr = f'Test $R^2$ = {test_r2:.3f}\nTest MAE = {test_mae:.3f}'

ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=22,
        va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
)

ax.legend(loc='lower right')
ax.grid(True, which='major', alpha=0.3)

plt.tight_layout()
if SAVE_FIG:
    plt.savefig(SCATTER_FNAME, dpi=400, bbox_inches='tight')
    print(f"[SUCCESS] saved: {SCATTER_FNAME}")
    pkg_path = Path(r"D:\workspace\test_dagang\test_ionic_0304_final\ionic_cond_20feat_package\figures\scatter_pred_vs_true_styled.png")
    plt.savefig(pkg_path, dpi=400, bbox_inches='tight')
    print(f"[SUCCESS] saved: {pkg_path}")
