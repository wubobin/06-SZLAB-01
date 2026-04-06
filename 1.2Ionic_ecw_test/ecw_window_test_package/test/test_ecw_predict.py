"""
关联模型测试脚本
输入: feature_ecw_matrix.csv, top_configs.json
输出: result_ecw.csv, result_ecw.json, window_model_rank.joblib, ecw_prediction_result.png
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42

PROTOCOL_FEATURES = [
    "Bandgap",
    "packing_fraction",
    "wavg_phi_Miedema",
    "wavg_ElectronAffinity",
    "wavg_GSestFCClatcnt",
    "VolumePerAtom",
    "wavg_NdUnfilled",
    "wavg_ZungerPP-r_sigma",
    "wavg_GSmagmom",
    "n_elements",
    "wavg_GSbandgap",
    "wavg_NdValence",
    "wavg_IsNonmetal",
    "wavg_CovalentRadius",
    "wavg_FirstIonizationEnergy",
    "wavg_Electronegativity",
    "wavg_NsValence",
    "wavg_MeltingT",
    "wavg_HeatCapacityMolar",
    "wavg_Anion_Electronegativity",
]

# ── 1. 读取描述符矩阵 ─────────────────────────────────────────────────────────
df = pd.read_csv(BASE_DIR / "feature_ecw_matrix.csv")
X = df[PROTOCOL_FEATURES].to_numpy()
y = df["Window Length (V)"].to_numpy()
mp_ids = df["MP ID"].to_numpy()
print(f"Dataset: {len(df)} samples, {len(PROTOCOL_FEATURES)} features")

# ── 2. 读取超参数 ─────────────────────────────────────────────────────────────
with open(BASE_DIR / "top_configs.json", encoding="utf-8") as f:
    cfgs = json.load(f)
rank2 = cfgs[1]
cfg = rank2["cfg"]
best_iter = int(rank2.get("best_iter", 2600))
n_estimators = min(max(best_iter + 100, 320), 2800)

# ── 3. 划分训练集 / 测试集（8:2，random_state=42）──────────────────────────
idx = np.arange(len(df))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

# ── 4. 构建关联模型 ──────────────────────────────────────────────────────────
model = XGBRegressor(
    n_estimators=n_estimators,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    n_jobs=16,
    random_state=RANDOM_STATE,
    **cfg,
)
model.fit(X[train_idx], y[train_idx], verbose=False)

# ── 5. 保存关联模型 ───────────────────────────────────────────────────────────
joblib.dump(model, BASE_DIR / "window_model_rank.joblib")
print("window_model_rank.joblib saved.")

# ── 6. 测试集评估 ─────────────────────────────────────────────────────────────
train_pred = model.predict(X[train_idx])
test_pred = model.predict(X[test_idx])
test_r2 = float(r2_score(y[test_idx], test_pred))
test_rmse = float(root_mean_squared_error(y[test_idx], test_pred))
print(f"Test R2: {test_r2:.4f}, RMSE: {test_rmse:.4f}")

# ── 7. 输出 result_ecw.csv（全部：训练集 + 测试集）─────────────────────────
df_result = pd.concat([
    pd.DataFrame({"MP ID": mp_ids[train_idx], "y_true": y[train_idx], "y_pred": train_pred}),
    pd.DataFrame({"MP ID": mp_ids[test_idx],  "y_true": y[test_idx],  "y_pred": test_pred}),
], ignore_index=True)
df_result.to_csv(BASE_DIR / "result_ecw.csv", index=False)
print("result_ecw.csv saved.")

# ── 8. 输出 result_ecw.json ───────────────────────────────────────────────────
with open(BASE_DIR / "result_ecw.json", "w") as f:
    json.dump({"test_r2": round(test_r2, 4), "test_rmse": round(test_rmse, 4)}, f, indent=2)
print("result_ecw.json saved.")

# ── 9. 绘制预测散点图 ─────────────────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 20

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(
    y[train_idx], train_pred,
    s=20, alpha=0.7, color="#4C92C3", edgecolors="none",
    label=f"Train (n={len(train_idx)})",
)
ax.scatter(
    y[test_idx], test_pred,
    s=20, alpha=0.8, color="#F28E2B", edgecolors="none",
    label=f"Test (n={len(test_idx)})",
)

lower_bound = min(float(y.min()), float(np.concatenate([train_pred, test_pred]).min()), 0.0)
upper_bound = max(float(y.max()), float(np.concatenate([train_pred, test_pred]).max())) * 1.05

ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound],
        "--", linewidth=1.6, color="#4C92C3", label="y = x")
ax.set_xlim(lower_bound, upper_bound)
ax.set_ylim(lower_bound, upper_bound)

ax.set_xlabel("True Window Length (V)", fontsize=18, fontweight="bold")
ax.set_ylabel("Predicted Window Length (V)", fontsize=18, fontweight="bold")
ax.tick_params(axis="both", labelsize=15)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")

metric_text = f"Test $R^2$ = {test_r2:.3f}\nTest RMSE = {test_rmse:.3f} V"
ax.text(
    0.05, 0.97, metric_text, transform=ax.transAxes, fontsize=20,
    va="top",
    bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "none", "alpha": 0.95},
)

ax.legend(loc="lower right", fontsize=20, frameon=True)
ax.grid(False)

fig.tight_layout()
fig.savefig(BASE_DIR / "ecw_prediction_result.png", dpi=400, bbox_inches="tight")
plt.close(fig)
print("[SUCCESS] ecw_prediction_result.png saved.")
