#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -N 1
#SBATCH --time=01:00:00
#SBATCH --mem=20G
#SBATCH --job-name shap-ecw
#SBATCH -o Job-%j.out
#SBATCH -e Job-%j.err
#SBATCH -p xhacnormalc

module purge

source /work/share/acitw40es7/wubb/miniconda3/etc/profile.d/conda.sh
conda activate test_ecw

cd $SLURM_SUBMIT_DIR

date
echo ">>> Starting SHAP analysis: $(pwd)"
echo ">>> CPUs allocated: $SLURM_CPUS_PER_TASK"

python - << 'EOF'
"""
SHAP 关键影响因素分析脚本
输入: feature_ecw_matrix.csv, window_model_rank.joblib
输出: key_factors_window_ranking.csv, ecw_shap_feature_importance.png
"""

import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# 内嵌模式下 __file__ 不可用，用 cwd() 代替（sbatch 已 cd 到提交目录）
BASE_DIR = Path.cwd()

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
df_feat = pd.read_csv(BASE_DIR / "feature_ecw_matrix.csv")
X = df_feat[PROTOCOL_FEATURES]
print(f"Loaded feature matrix: {X.shape}")

# ── 2. 加载关联模型 ───────────────────────────────────────────────────────────
model = joblib.load(BASE_DIR / "window_model_rank.joblib")
print("window_model_rank.joblib loaded.")

# ── XGBoost 3.x + SHAP 0.49.x 兼容补丁 ───────────────────────────────────────
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

# ── 3. 计算 SHAP 值（8进程并行）────────────────────────────────────────────────
from joblib import Parallel, delayed

_MODEL_PATH = str(BASE_DIR / "window_model_rank.joblib")

def _shap_chunk(model_path, X_chunk_values):
    """子进程入口：独立加载模型，避免跨进程共享对象冲突"""
    import joblib, shap, numpy as np
    import shap.explainers._tree as _shap_tree

    _orig = _shap_tree.decode_ubjson_buffer
    def _fixed(fd):
        result = _orig(fd)
        try:
            bs = result['learner']['learner_model_param']['base_score']
            if isinstance(bs, str) and bs.startswith('[') and bs.endswith(']'):
                result['learner']['learner_model_param']['base_score'] = bs[1:-1]
        except (KeyError, TypeError):
            pass
        return result
    _shap_tree.decode_ubjson_buffer = _fixed

    m = joblib.load(model_path)
    exp = shap.TreeExplainer(m)
    return exp.shap_values(X_chunk_values, check_additivity=False)

N_JOBS = 8
X_chunks = np.array_split(X.values, N_JOBS)
print(f"Computing SHAP values with {N_JOBS} parallel processes ({len(X_chunks[0])} samples/process)...")
results = Parallel(n_jobs=N_JOBS, backend="loky")(
    delayed(_shap_chunk)(_MODEL_PATH, chunk) for chunk in X_chunks
)
shap_values = np.vstack(results)   # shape: (N, 20)

# ── 4. 汇总 mean_abs_shap 并输出 CSV ────────────────────────────────────────
mean_abs_shap = np.abs(shap_values).mean(axis=0)

df_ranking = pd.DataFrame({
    "描述符": PROTOCOL_FEATURES,
    "mean |Shapley value|": np.round(mean_abs_shap, 4),
}).sort_values("mean |Shapley value|", ascending=False).reset_index(drop=True)
df_ranking.to_csv(BASE_DIR / "key_factors_window_ranking.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
print("key_factors_window_ranking.csv saved.")

print("\nTop-5 Key Factors:")
print(df_ranking.head(5).to_string(index=False))

# ── 5. 绘制 SHAP 特征重要性条形图 ─────────────────────────────────────────────
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
features = [PROTOCOL_FEATURES[i] for i in sorted_idx]
values = [float(mean_abs_shap[i]) for i in sorted_idx]

cmap = mcolors.LinearSegmentedColormap.from_list("shap_blue", [LIGHT_BLUE, DARK_BLUE])
colors = [cmap(i / max(len(features) - 1, 1)) for i in range(len(features))]

fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(range(len(features)), values, color=colors, edgecolor="white", linewidth=0.4, height=0.75)
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features, fontsize=10)
ax.set_xlabel("Mean |SHAP value|", fontsize=12)
ax.set_xlim(left=0)
ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(BASE_DIR / "ecw_shap_feature_importance.png", dpi=400, bbox_inches="tight")
plt.close(fig)
print(f"\n[SUCCESS] ecw_shap_feature_importance.png saved.")
EOF

echo ">>> SHAP analysis done."
date

rm -f "${SLURM_SUBMIT_DIR}/Job-${SLURM_JOB_ID}.out" \
      "${SLURM_SUBMIT_DIR}/Job-${SLURM_JOB_ID}.err"
