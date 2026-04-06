"""
compute_shap_all_splits.py
==================================================
重新计算 SHAP 值在三个数据集上的分布：
1. 全样本（train + test, N=171）
2. 训练集（N=136）
3. 测试集（N=35）

对比 mean_abs_shap 在不同样本集上的差异。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
PROC_CSV = ROOT / "data" / "processed" / "features_20feat.csv"
TRAIN_CSV = ROOT / "data" / "train" / "train_split.csv"
TEST_CSV = ROOT / "data" / "test" / "test_split.csv"
SUMMARY_JSON = ROOT / "reports" / "model_summary.json"
OUT_DIR = ROOT / "reports"

FEAT_20 = [
    "n_elements", "x_li", "wavg_AtomicNumber", "wavg_AtomicWeight",
    "wavg_CovalentRadius", "ratio_CationToAnionIonicPotential", "x_HalideTotal", "anion_MixEntropy",
    "AnionCation_PhiGap", "LiPhi_Coupling", "LiHalide_Coupling",
    "Density", "VolumePerAtom", "SpaceGroupNumber",
    "LatticeA", "LatticeB", "LatticeC",
    "LatticeAlpha", "LatticeBeta", "LatticeGamma",
]
TARGET = "log10_ionic_conductivity"


def main() -> None:
    print("=" * 70)
    print("  SHAP 值计算 — 全样本 / 训练集 / 测试集对比")
    print("=" * 70)

    # 1. 加载数据
    df_all = pd.read_csv(PROC_CSV, encoding="utf-8-sig")
    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    df_test = pd.read_csv(TEST_CSV, encoding="utf-8-sig")

    print(f"\n[1/4] 数据加载完成")
    print(f"      全样本: {len(df_all)} 条")
    print(f"      训练集: {len(df_train)} 条")
    print(f"      测试集: {len(df_test)} 条")

    # 2. 加载模型参数并重新训练（因为模型对象未保存）
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary = json.load(f)
    best_params = summary["best_params"]
    seed = summary["split_seed"]

    X_train = df_train[FEAT_20].to_numpy(dtype=float)
    y_train = df_train[TARGET].to_numpy(dtype=float)

    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        **best_params,
    )
    model.fit(X_train_imp, y_train)
    print(f"\n[2/4] 模型重新训练完成（使用保存的最优超参数）")

    # 3. 计算三组 SHAP 值
    explainer = shap.TreeExplainer(model)

    def compute_shap(df: pd.DataFrame, name: str) -> pd.DataFrame:
        X = df[FEAT_20].to_numpy(dtype=float)
        X_imp = imp.transform(X)
        shap_vals = explainer.shap_values(X_imp)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        return pd.DataFrame({
            "feature": FEAT_20,
            f"mean_abs_shap_{name}": mean_abs,
        })

    shap_all = compute_shap(df_all, "all")
    shap_train = compute_shap(df_train, "train")
    shap_test = compute_shap(df_test, "test")

    print(f"[3/4] SHAP 值计算完成")

    # 4. 合并对比
    merged = shap_all.merge(shap_train, on="feature").merge(shap_test, on="feature")
    merged = merged.sort_values("mean_abs_shap_all", ascending=False).reset_index(drop=True)
    merged["rank_all"] = range(1, len(merged) + 1)

    # 保存
    out_path = OUT_DIR / "shap_comparison_all_train_test.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[4/4] 对比表已保存: {out_path}")

    # 打印前10
    print("\n" + "=" * 70)
    print("  SHAP 重要性对比 (Top 10)")
    print("=" * 70)
    print(merged.head(10).to_string(index=False))

    # 统计说明
    print("\n" + "=" * 70)
    print("  计算公式说明")
    print("=" * 70)
    print("""
对于数据集 D（全样本/训练集/测试集），每个特征 j 的 mean_abs_shap 计算为：

    mean_abs_shap_j = (1/N) * Σ_{i=1}^{N} |φ_{i,j}|

其中：
  - N = 数据集样本数（全样本171 / 训练136 / 测试35）
  - φ_{i,j} = 样本 i 在特征 j 上的 SHAP 值（单位：log10σ）
  - |·| = 取绝对值（忽略正负方向，只看贡献量级）
  - Σ / N = 对所有样本求平均

**关键点：**
1. mean_abs_shap 已经是"平均值"，不是"总和"
2. 不同数据集的 N 不同，所以 mean_abs_shap 的数值可直接对比
3. 测试集样本少（N=35），个别极端样本会显著影响均值
4. 训练集样本多（N=136），均值更稳定
5. 全样本（N=171）反映模型在整个数据分布上的特征重要性

**观察：**
- 如果 mean_abs_shap_test >> mean_abs_shap_train，说明该特征在测试集上
  的贡献波动更大（可能测试集包含该特征的极端样本）
- 如果 mean_abs_shap_all ≈ mean_abs_shap_train，说明训练集主导了全局分布
    """)


if __name__ == "__main__":
    main()
