"""
tune_hyperparams.py — Optuna 贝叶斯超参数搜索 + 8:2 Holdout 评估
================================================================
输入:
  - ../feature_ionic_matrix.csv    (20 描述符)
  - ../data_ionic_conductivity.json (标签)
输出:
  - ../top_configs.json            (最优超参数)
  - reports/tuning_summary.json    (搜索过程摘要)

流程:
  1. 合并描述符 + 标签
  2. 8:2 划分训练/测试集 (random_state=55)
  3. Optuna TPE 采样, 100 trials, 5-fold CV
  4. 输出最优超参数 → top_configs.json
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── 路径 ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent       # training_process/
ROOT_DIR   = BASE_DIR.parent                    # ionic_cond_standard_test/
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_CSV   = ROOT_DIR / "feature_ionic_matrix.csv"
LABEL_JSON = ROOT_DIR / "data_ionic_conductivity.json"
OUTPUT_PARAMS = ROOT_DIR / "top_configs.json"

# ── 配置 ──────────────────────────────────────────────────────────────────────
FEAT_20 = [
    "n_elements", "x_li",
    "wavg_AtomicNumber", "wavg_AtomicWeight", "wavg_CovalentRadius",
    "ratio_CationToAnionIonicPotential", "x_HalideTotal", "anion_MixEntropy",
    "AnionCation_PhiGap", "LiPhi_Coupling", "LiHalide_Coupling",
    "Density", "VolumePerAtom", "SpaceGroupNumber",
    "LatticeA", "LatticeB", "LatticeC",
    "LatticeAlpha", "LatticeBeta", "LatticeGamma",
]
RANDOM_STATE = 55
TEST_SIZE    = 0.2
N_TRIALS     = 100
CV_FOLDS     = 5


# ── Optuna 目标函数 ──────────────────────────────────────────────────────────

def make_objective(X_train: np.ndarray, y_train: np.ndarray):
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 150, 1800),
            "max_depth":        trial.suggest_int("max_depth", 2, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 15.0),
            "gamma":            trial.suggest_float("gamma", 0.0, 8.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 20.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-6, 60.0, log=True),
        }
        fold_maes: list[float] = []
        for tr_idx, va_idx in kf.split(X_train):
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(X_train[tr_idx])
            X_va = imp.transform(X_train[va_idx])
            model = XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=1,
                **params,
            )
            model.fit(X_tr, y_train[tr_idx])
            pred = model.predict(X_va)
            fold_maes.append(float(np.mean(np.abs(y_train[va_idx] - pred))))
        return float(np.mean(fold_maes))

    return objective


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Optuna 超参数搜索")
    print("=" * 60)

    # 1. 加载数据
    with open(LABEL_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    df_label = pd.DataFrame(raw)[["record_id", "log10_ionic_conductivity"]]
    df_feat = pd.read_csv(FEAT_CSV)
    df = df_label.merge(df_feat, on="record_id", how="inner")
    print(f"[1/4] 数据: {len(df)} 样本, {len(FEAT_20)} 特征")

    # 2. 划分
    X = df[FEAT_20].values
    y = df["log10_ionic_conductivity"].values
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[2/4] 划分: 训练 {len(y_train)}, 测试 {len(y_test)}")

    # 3. Optuna 搜索
    print(f"[3/4] Optuna TPE: {N_TRIALS} trials, {CV_FOLDS}-fold CV ...")
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(make_objective(X_train, y_train), n_trials=N_TRIALS)

    best_params = dict(study.best_params)
    print(f"      最优 CV MAE: {study.best_value:.4f}")

    # 4. 用最优参数在测试集上验证
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        **best_params,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_r2  = float(r2_score(y_test, y_pred))
    test_mae = float(np.mean(np.abs(y_test - y_pred)))
    print(f"[4/4] 测试集: R² = {test_r2:.4f}, MAE = {test_mae:.4f}")

    # 输出
    with open(OUTPUT_PARAMS, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"      已保存: {OUTPUT_PARAMS}")

    summary = {
        "random_state": RANDOM_STATE,
        "n_trials": N_TRIALS,
        "cv_folds": CV_FOLDS,
        "best_cv_mae": round(study.best_value, 4),
        "test_r2": round(test_r2, 4),
        "test_mae": round(test_mae, 4),
        "best_params": best_params,
    }
    summary_path = REPORT_DIR / "tuning_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"      搜索摘要: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
