#!/usr/bin/env python
"""Reproduce the final 20-descriptor ECW training/test workflow."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from build_final_20_feature_matrix import ID_COL, TARGET_COL, build_feature_dataframe


ROOT = Path(__file__).resolve().parent
TRAIN_ROOT = ROOT.parent
CONFIG_PATH = TRAIN_ROOT / 'configs' / 'top_configs.json'
OUTPUT_DIR = TRAIN_ROOT / 'outputs'
MODEL_DIR = TRAIN_ROOT / 'models'
IMAGE_DIR = TRAIN_ROOT / 'figures'
MODEL_FEATURE_ORDER = [
    'Bandgap',
    'packing_fraction',
    'wavg_phi_Miedema',
    'wavg_ElectronAffinity',
    'wavg_GSestFCClatcnt',
    'VolumePerAtom',
    'wavg_NdUnfilled',
    'wavg_ZungerPP-r_sigma',
    'wavg_GSmagmom',
    'n_elements',
    'wavg_GSbandgap',
    'wavg_NdValence',
    'wavg_IsNonmetal',
    'wavg_CovalentRadius',
    'wavg_FirstIonizationEnergy',
    'wavg_Electronegativity',
    'wavg_NsValence',
    'wavg_MeltingT',
    'wavg_HeatCapacityMolar',
    'wavg_Anion_Electronegativity',
]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        'r2': float(r2_score(y_true, y_pred)),
        'rmse': float(root_mean_squared_error(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
    }


def _save_scatter_plot(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    test_metrics: dict[str, float],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(9, 8))

    ax.scatter(y_train_true, y_train_pred, s=30, alpha=0.65, label=f'Train (n={len(y_train_true)})')
    ax.scatter(y_test_true, y_test_pred, s=30, alpha=0.75, label=f'Test (n={len(y_test_true)})')

    lo = min(
        float(np.min(y_train_true)),
        float(np.min(y_train_pred)),
        float(np.min(y_test_true)),
        float(np.min(y_test_pred)),
    )
    hi = max(
        float(np.max(y_train_true)),
        float(np.max(y_train_pred)),
        float(np.max(y_test_true)),
        float(np.max(y_test_pred)),
    )
    ax.plot([lo, hi], [lo, hi], linestyle='--', linewidth=2.2, label='y = x')

    ax.set_xlabel('True Value (V)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Predicted Value (V)', fontsize=20, fontweight='bold')
    ax.set_title('ECW Window Length (V) - Prediction Performance', fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=15)

    metric_text = (
        f"Test: RMSE = {test_metrics['rmse']:.3f} V\n"
        f"Test: R2 = {test_metrics['r2']:.3f}"
    )
    ax.text(
        0.04,
        0.95,
        metric_text,
        transform=ax.transAxes,
        fontsize=16,
        va='top',
        bbox={'boxstyle': 'round', 'facecolor': '#f7e6bf', 'edgecolor': '#333333', 'alpha': 0.95},
    )

    ax.legend(loc='lower right', fontsize=14, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_training(output_dir: Path | None = None) -> dict[str, object]:
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    data_df, selected_features = build_feature_dataframe()

    idx = np.arange(len(data_df))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
    train_df = data_df.iloc[train_idx].reset_index(drop=True)
    test_df = data_df.iloc[test_idx].reset_index(drop=True)

    train_csv = output_dir / 'train_data.csv'
    test_csv = output_dir / 'test_data.csv'
    metrics_json = output_dir / 'metrics.json'
    predictions_csv = output_dir / 'test_predictions.csv'
    feature_list_json = output_dir / 'feature_list_final_20.json'
    summary_md = output_dir / 'run_summary.md'
    model_path = MODEL_DIR / 'window_model_single_xgb.joblib'
    image_path = IMAGE_DIR / 'test_pred_vs_true.png'

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    x_train = train_df[MODEL_FEATURE_ORDER].to_numpy()
    y_train = train_df[TARGET_COL].to_numpy()
    x_test = test_df[MODEL_FEATURE_ORDER].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    cfgs = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    rank2 = cfgs[1]
    cfg = rank2['cfg']
    best_iter = int(rank2.get('best_iter', 2600))
    n_estimators = min(max(best_iter + 100, 320), 2800)

    model = XGBRegressor(
        n_estimators=n_estimators,
        objective='reg:squarederror',
        eval_metric='rmse',
        tree_method='hist',
        n_jobs=16,
        random_state=42,
        **cfg,
    )
    model.fit(x_train, y_train, verbose=False)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    metric_values = metrics(y_test, y_test_pred)
    threshold = {
        'rmse_lt_0p4': bool(metric_values['rmse'] < 0.4),
        'r2_gt_0p85': bool(metric_values['r2'] > 0.85),
    }

    dump(model, model_path)
    pd.DataFrame({
        ID_COL: test_df[ID_COL].to_numpy(),
        'y_true': y_test,
        'y_pred': y_test_pred,
        'abs_err': np.abs(y_test_pred - y_test),
    }).to_csv(predictions_csv, index=False)

    feature_list_json.write_text(
        json.dumps({'selected_features': selected_features}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    _save_scatter_plot(
        y_train_true=y_train,
        y_train_pred=y_train_pred,
        y_test_true=y_test,
        y_test_pred=y_test_pred,
        test_metrics=metric_values,
        out_path=image_path,
    )

    result = {
        'project': TRAIN_ROOT.name,
        'target': TARGET_COL,
        'model': 'XGBRegressor(single)',
        'split': '8:2 random_state=42',
        'n_samples': int(len(data_df)),
        'n_features': int(len(selected_features)),
        'features': selected_features,
        'model_feature_order': MODEL_FEATURE_ORDER,
        'metrics_test': metric_values,
        'threshold': threshold,
        'artifacts': {
            'train_data': str(train_csv),
            'test_data': str(test_csv),
            'model': str(model_path),
            'predictions': str(predictions_csv),
            'feature_list': str(feature_list_json),
            'plot': str(image_path),
        },
    }
    metrics_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    summary_lines = [
        '# ECW Single XGBoost (20 descriptors) Training Reproduction',
        '',
        f"- Test R2: `{metric_values['r2']:.10f}`",
        f"- Test RMSE: `{metric_values['rmse']:.10f} V`",
        f"- Test MAE: `{metric_values['mae']:.10f} V`",
        f"- Threshold RMSE<0.4: `{threshold['rmse_lt_0p4']}`",
        f"- Threshold R2>0.85: `{threshold['r2_gt_0p85']}`",
    ]
    summary_md.write_text('\n'.join(summary_lines), encoding='utf-8')
    return result


def main() -> None:
    result = run_training()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
