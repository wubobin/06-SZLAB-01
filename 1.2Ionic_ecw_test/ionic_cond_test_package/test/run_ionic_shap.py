"""
SHAP 关键影响因素分析脚本
输入: feature_ionic_matrix.csv, ionic_rank.joblib
输出: key_factors_ionic_ranking.csv, shap_feature_importance.png
"""

import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).parent

FEAT_20 = [
    "n_elements", "x_li", "wavg_AtomicNumber", "wavg_AtomicWeight",
    "wavg_CovalentRadius", "ratio_CationToAnionIonicPotential", "x_HalideTotal", "anion_MixEntropy",
    "AnionCation_PhiGap", "LiPhi_Coupling", "LiHalide_Coupling",
    "Density", "VolumePerAtom", "SpaceGroupNumber",
    "LatticeA", "LatticeB", "LatticeC", "LatticeAlpha", "LatticeBeta", "LatticeGamma"
]

# ── 1. 读取描述符矩阵（全部171条样本）────────────────────────────────────────
df_feat = pd.read_csv(BASE_DIR / "feature_ionic_matrix.csv")
X = df_feat[FEAT_20].values

imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X)
print(f"Loaded feature matrix: {X_imp.shape}")

# ── 2. 加载训练好的模型权重 ───────────────────────────────────────────────────
model = joblib.load(BASE_DIR / "ionic_rank.joblib")
print("ionic_rank.joblib loaded.")

# XGBoost 3.x + SHAP 0.49.x 兼容补丁
# SHAP 0.49.x 使用 save_raw + decode_ubjson_buffer 解析模型（不调用 save_config）
# XGBoost 3.x 在 UBJSON 中将 base_score 存为 '[-4.58321E-1]'（带方括号）
# 拦截 decode_ubjson_buffer 返回值，去掉方括号即可
import shap.explainers._tree as _shap_tree

_orig_decode_ubjson = _shap_tree.decode_ubjson_buffer

def _fixed_decode_ubjson(fd):
    result = _orig_decode_ubjson(fd)
    try:
        bs = result['learner']['learner_model_param']['base_score']
        if isinstance(bs, str) and bs.startswith('[') and bs.endswith(']'):
            result['learner']['learner_model_param']['base_score'] = bs[1:-1]
    except (KeyError, TypeError):
        pass
    return result

_shap_tree.decode_ubjson_buffer = _fixed_decode_ubjson

# ── 3. 计算 SHAP 值 ───────────────────────────────────────────────────────────
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_imp)   # shape: (N, 20)

# ── 4. mean_abs_shap = (1/N) * Σ|φ_{i,j}| ───────────────────────────────────
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# ── 5. 排序并输出 CSV ───────────────────────────────────────────────────────────
df_ranking = pd.DataFrame({
    "描述符":              FEAT_20,
    "mean |Shapley value|": np.round(mean_abs_shap, 4)
}).sort_values("mean |Shapley value|", ascending=False).reset_index(drop=True)

df_ranking.to_csv(BASE_DIR / "key_factors_ionic_ranking.csv", index=False)
print("key_factors_ionic_ranking.csv saved.")

print("\nTop-5 Key Factors:")
print(df_ranking.head(5).to_string(index=False))

# ── 6. 绘制 SHAP 特征重要性条形图 ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.dpi": 400,
})

DARK_BLUE = "#1a5276"
LIGHT_BLUE = "#a9cce3"

sorted_idx = np.argsort(mean_abs_shap)
sorted_features = [FEAT_20[i] for i in sorted_idx]
sorted_values = [float(mean_abs_shap[i]) for i in sorted_idx]

cmap = mcolors.LinearSegmentedColormap.from_list("shap_blue", [LIGHT_BLUE, DARK_BLUE])
colors = [cmap(i / max(len(sorted_features) - 1, 1)) for i in range(len(sorted_features))]

fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(range(len(sorted_features)), sorted_values, color=colors, edgecolor="white", linewidth=0.4, height=0.75)
ax.set_yticks(range(len(sorted_features)))
ax.set_yticklabels(sorted_features, fontsize=10)
ax.set_xlabel("Mean |SHAP value|", fontsize=12)
ax.set_xlim(left=0)
ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig_path = BASE_DIR / "shap_feature_importance.png"
fig.savefig(fig_path, dpi=400, bbox_inches="tight")
plt.close(fig)
print(f"\n[SUCCESS] SHAP bar chart saved: {fig_path}")
