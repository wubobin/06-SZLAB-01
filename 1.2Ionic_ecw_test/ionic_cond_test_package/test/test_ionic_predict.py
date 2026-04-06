"""
关联模型测试脚本
输入: data_ionic_conductivity.json, feature_ionic_matrix.csv, top_configs.json
输出: result_ionic.csv, result_ionic.json, ionic_rank.joblib, ionic_prediction_result.png
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 55

FEAT_20 = [
    "n_elements", "x_li", "wavg_AtomicNumber", "wavg_AtomicWeight",
    "wavg_CovalentRadius", "ratio_CationToAnionIonicPotential", "x_HalideTotal", "anion_MixEntropy",
    "AnionCation_PhiGap", "LiPhi_Coupling", "LiHalide_Coupling",
    "Density", "VolumePerAtom", "SpaceGroupNumber",
    "LatticeA", "LatticeB", "LatticeC", "LatticeAlpha", "LatticeBeta", "LatticeGamma"
]

# ── 1. 读取原始实验数据 ──────────────────────────────────────────────────────
with open(BASE_DIR / "data_ionic_conductivity.json", encoding="utf-8") as f:
    raw_records = json.load(f)

df_label = pd.DataFrame(raw_records)[["record_id", "log10_ionic_conductivity"]]

# ── 2. 读取描述符矩阵 ─────────────────────────────────────────────────────────
df_feat = pd.read_csv(BASE_DIR / "feature_ionic_matrix.csv")

# ── 3. 按 record_id 合并 ─────────────────────────────────────────────────────
df = df_label.merge(df_feat, on="record_id", how="inner")
print(f"Merged dataset: {len(df)} samples, {len(FEAT_20)} features")

# ── 4. 读取超参数 ─────────────────────────────────────────────────────────────
with open(BASE_DIR / "top_configs.json") as f:
    best_params = json.load(f)

# ── 5. 划分训练集 / 测试集（8:2，random_state=55）──────────────────────────
X = df[FEAT_20].values
y = df["log10_ionic_conductivity"].values
record_ids = df["record_id"].values

imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X)

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X_imp, y, record_ids, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Train: {len(y_train)}, Test: {len(y_test)}")

# ── 6. 构建关联模型 ──────────────────────────────────────────────────────────
model = XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    random_state=RANDOM_STATE,
    **best_params
)
model.fit(X_train, y_train)

# ── 7. 保存关联模型 ───────────────────────────────────────────────────────────
joblib.dump(model, BASE_DIR / "ionic_rank.joblib")
print("ionic_rank.joblib saved.")

# ── 8. 测试集评估 ─────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
test_r2  = float(r2_score(y_test, y_pred))
test_mae = float(np.mean(np.abs(y_test - y_pred)))
print(f"Test R2: {test_r2:.4f}, MAE: {test_mae:.4f}")

# ── 9. 输出 result_ionic.csv（全部 171 条：训练集 + 测试集）─────────────────
y_train_pred = model.predict(X_train)
df_result = pd.concat([
    pd.DataFrame({"record_id": id_train, "y_true": y_train, "y_pred": y_train_pred}),
    pd.DataFrame({"record_id": id_test,  "y_true": y_test,  "y_pred": y_pred}),
], ignore_index=True)
df_result.to_csv(BASE_DIR / "result_ionic.csv", index=False)
print("result_ionic.csv saved.")

# ── 10. 输出 result_ionic.json ───────────────────────────────────────────────
with open(BASE_DIR / "result_ionic.json", "w") as f:
    json.dump({"test_r2": round(test_r2, 4), "test_mae": round(test_mae, 4)}, f, indent=2)
print("result_ionic.json saved.")

# ── 11. 绘制预测散点图 ───────────────────────────────────────────────────────

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 20

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(
    y_train, y_train_pred,
    s=60, alpha=0.6, color="#4C92C3", edgecolors="none",
    label=f"Train (n={len(y_train)})",
)
ax.scatter(
    y_test, y_pred,
    s=60, alpha=0.7, color="#F28E2B", edgecolors="none",
    label=f"Test (n={len(y_test)})",
)

ax.set_xlim(-5, 2)
ax.set_ylim(-5, 2)
ticks = np.arange(-5, 3, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
xlabels = [str(t) if -4 <= t <= 1 else "" for t in ticks]
ylabels = [str(t) if -4 <= t <= 1 else "" for t in ticks]
ax.set_xticklabels(xlabels)
ax.set_yticklabels(ylabels)
ax.set_aspect("equal", adjustable="box")

ax.plot([-5, 2], [-5, 2], "--", linewidth=1.6, color="#4C92C3", label="y = x")

ax.set_xlabel(r"True $\log_{10}(\sigma)$", fontsize=22, fontweight="bold")
ax.set_ylabel(r"Predicted $\log_{10}(\sigma)$", fontsize=22, fontweight="bold")
ax.tick_params(axis="both", labelsize=16)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")

metric_text = f"Test $R^2$ = {test_r2:.3f}\nTest MAE = {test_mae:.3f}"
ax.text(
    0.05, 0.95, metric_text, transform=ax.transAxes, fontsize=20,
    va="top",
    bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "gray", "alpha": 0.8},
)

ax.legend(loc="lower right", fontsize=16, frameon=True)
ax.grid(True, which="major", alpha=0.3)

fig.tight_layout()
fig.savefig(BASE_DIR / "ionic_prediction_result.png", dpi=400, bbox_inches="tight")
plt.close(fig)
print("[SUCCESS] ionic_prediction_result.png saved.")
