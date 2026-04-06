# 20-Feature XGBoost Model — Results Report

## 1. Model Configuration

| Item | Value |
|------|-------|
| Algorithm | XGBoost (single model, reg:squarederror) |
| Features | 20 |
| Target | log10(ionic conductivity / mS·cm⁻¹) |
| Train/Test Split | 8:2  (seed=55) |
| n_train | 136 |
| n_test  | 35 |
| Optuna trials | 100 |
| CV folds (tuning) | 5 |

## 2. Performance Metrics

| Metric | Train | Test |
|--------|-------|------|
| R² | 0.9146 | **0.8234** |
| MAE (log10σ) | 0.1952 | **0.2566** |

Target (R² > 0.8 & MAE < 0.3): **MET**

## 3. Best Hyperparameters

```json
{
  "n_estimators": 727,
  "max_depth": 9,
  "learning_rate": 0.16043159941695556,
  "subsample": 0.600379835500918,
  "colsample_bytree": 0.9476438345593615,
  "min_child_weight": 2.3405174661277037,
  "gamma": 0.6450549035667705,
  "reg_alpha": 2.1072274507455447e-05,
  "reg_lambda": 4.1570142768446117e-05
}
```

## 4. SHAP Top-5 Feature Importance (test set)

| Rank | Feature | Mean |SHAP| |
|------|---------|-----------|
| 1 | `LiPhi_Coupling` | 0.1241 |
| 2 | `n_elements` | 0.1236 |
| 3 | `LatticeC` | 0.1203 |
| 4 | `LatticeB` | 0.0935 |
| 5 | `LiHalide_Coupling` | 0.0629 |

## 5. Feature List (20 features)

| # | Symbol | Category |
|---|--------|----------|
| 1 | `n_elements` | Composition |
| 2 | `x_li` | Composition |
| 3 | `wavg_AtomicNumber` | Composition |
| 4 | `wavg_AtomicWeight` | Composition |
| 5 | `wavg_CovalentRadius` | Composition |
| 6 | `ratio_CationToAnionIonicPotential` | Composition |
| 7 | `x_HalideTotal` | Composition |
| 8 | `anion_MixEntropy` | Composition |
| 9 | `AnionCation_PhiGap` | Coupling |
| 10 | `LiPhi_Coupling` | Coupling |
| 11 | `LiHalide_Coupling` | Coupling |
| 12 | `Density` | Structure |
| 13 | `VolumePerAtom` | Structure |
| 14 | `SpaceGroupNumber` | Structure |
| 15 | `LatticeA` | Structure |
| 16 | `LatticeB` | Structure |
| 17 | `LatticeC` | Structure |
| 18 | `LatticeAlpha` | Structure (angle) |
| 19 | `LatticeBeta` | Structure (angle) |
| 20 | `LatticeGamma` | Structure (angle) |

## 6. Output Files

| File | Description |
|------|-------------|
| `data/processed/features_20feat.csv` | Feature-engineered dataset |
| `data/train/train_split.csv` | Training split |
| `data/test/test_split.csv` | Test split |
| `data/test/test_predictions.csv` | Test set predictions |
| `figures/scatter_pred_vs_true.png` | Pred vs True scatter |
| `figures/shap_feature_importance.png` | SHAP importance bar chart |
| `reports/model_summary.json` | Machine-readable summary |
