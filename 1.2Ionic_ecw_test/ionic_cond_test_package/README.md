# 卤化物固态电解质离子电导率预测模型 — 测试包

> **服务器路径**：`/work/share/acitw40es7/ND_test/1.2Ionic_ecw_test/ionic_cond_test_package/`
> **环境名称**：`test_ionic`
> **验证指标**：测试集 R² = 0.8234，MAE = 0.2566 log₁₀(σ / mS·cm⁻¹)

---

## 目录结构

```
ionic_cond_test_package/
├── test/                                ← 测试（运行2个脚本即完成）
│   ├── data_ionic_conductivity.json     ← 171条实验记录
│   ├── feature_ionic_matrix.csv         ← record_id + 20描述符
│   ├── top_configs.json                 ← XGBoost最优超参数（9个）
│   ├── test_ionic_predict.py            ← 脚本1: 关联模型测试 + 预测散点图
│   └── run_ionic_shap.py                ← 脚本2: SHAP分析 + 特征重要性图
│
├── train/                               ← 训练过程完整记录
│   ├── data/
│   │   ├── cif/                         ← 171个CIF晶体结构文件
│   │   └── oxidation_state_default_map.json
│   ├── scripts/                         ← 训练脚本（描述符构建、超参数搜索等）
│   ├── reports/                         ← 模型报告与SHAP分析
│   └── figures/                         ← 训练过程可视化图
│
├── env/                                 ← 环境安装
│   ├── requirements.txt                 ← Python依赖列表
│   └── setup_env.sh                     ← 一键安装脚本
│
└── README.md                            ← 本文件
```

---

## 一、环境安装（一键）

```bash
cd /work/share/acitw40es7/ND_test/1.2Ionic_ecw_test/ionic_cond_test_package/env
bash setup_env.sh
```

安装完成后激活环境：
```bash
conda activate test_ionic
```

---

## 二、运行测试（两个脚本即完成全部测试 + 图片输出）

```bash
cd /work/share/acitw40es7/ND_test/1.2Ionic_ecw_test/ionic_cond_test_package/test

# 脚本1: 关联模型测试（训练 + 评估 + 散点图）
python test_ionic_predict.py

# 脚本2: SHAP关键影响因素分析（SHAP值计算 + 条形图）
python run_ionic_shap.py
```

### 运行后 test/ 目录产出

| 输出文件 | 说明 | 来源脚本 |
|---------|------|---------|
| `result_ionic.csv` | 测试集预测结果（record_id \| y_true \| y_pred） | test_ionic_predict.py |
| `result_ionic.json` | 测试集指标 {"test_r2": 0.8234, "test_mae": 0.2566} | test_ionic_predict.py |
| `ionic_rank.joblib` | 训练好的XGBoost模型权重 | test_ionic_predict.py |
| `ionic_prediction_result.png` | 预测值 vs 真实值散点图（Train+Test） | test_ionic_predict.py |
| `key_factors_ionic_ranking.csv` | 20个描述符SHAP重要性排序 | run_ionic_shap.py |
| `shap_feature_importance.png` | SHAP特征重要性水平条形图 | run_ionic_shap.py |

---

## 三、测试流程说明

### 3.1 关联模型测试 — `test_ionic_predict.py`

1. 读取 `data_ionic_conductivity.json` → 提取 record_id + log₁₀(σ) 目标值
2. 读取 `feature_ionic_matrix.csv` → 20个描述符
3. 按 record_id 合并 → 完整数据集（N=171）
4. 读取 `top_configs.json` → XGBoost最优超参数
5. 8:2 随机划分（random_state=55）→ 训练集136 + 测试集35
6. SimpleImputer(strategy='median') 填充缺失值
7. 训练 XGBRegressor → 保存 `ionic_rank.joblib`
8. 测试集评估 → 输出 `result_ionic.csv` + `result_ionic.json`
9. 绘制 Train/Test 散点图 → 输出 `ionic_prediction_result.png`

### 3.2 SHAP关键影响因素分析 — `run_ionic_shap.py`

1. 读取 `feature_ionic_matrix.csv` → 全部171条样本
2. 加载 `ionic_rank.joblib`（由脚本1生成）
3. SHAP TreeExplainer → 计算每个样本在每个描述符上的Shapley值
4. 计算 mean|SHAP|：mean_abs_shap_j = (1/N) × Σ|φ_{i,j}|
5. 按均值降序排列 → 输出 `key_factors_ionic_ranking.csv`
6. 绘制水平条形图 → 输出 `shap_feature_importance.png`

> **注意**：`run_ionic_shap.py` 依赖 `test_ionic_predict.py` 生成的 `ionic_rank.joblib`，必须先运行脚本1。

---

## 四、预期结果

| 测试项 | 预期值 |
|--------|--------|
| 测试集 R² | **0.8234** |
| 测试集 MAE | **0.2566** log₁₀(σ) |
| SHAP 第1位 | `LiPhi_Coupling` (0.1529) |
| SHAP 第2位 | `n_elements` (0.1262) |
| SHAP 第3位 | `LatticeC` (0.1159) |

---

## 五、输入数据说明

### 5.1 实验数据 — `data_ionic_conductivity.json`

171条卤化物固态电解质实验记录，每条含：
- `record_id`：结构唯一编号
- `formula`：化学式
- `space_group`：空间群
- `ionic_conductivity_mS_cm`：实验离子电导率（mS/cm）
- `log10_ionic_conductivity`：电导率对数值

### 5.2 特征矩阵 — `feature_ionic_matrix.csv`

21列（record_id + 20描述符），171行。

### 5.3 超参数 — `top_configs.json`

Optuna贝叶斯优化（100次试验，5折交叉验证）得到的9个最优超参数。

---

## 六、20个描述符分类

| 类别 | 描述符 | 说明 |
|------|--------|------|
| 组分 (8) | `n_elements` | 元素种类数 |
| | `x_li` | Li原子摩尔分数 |
| | `wavg_AtomicNumber` | 组分加权平均原子序数 |
| | `wavg_AtomicWeight` | 组分加权平均原子量 |
| | `wavg_CovalentRadius` | 组分加权平均共价半径 |
| | `ratio_CationToAnionIonicPotential` | 阳离子/阴离子离子势之比 |
| | `x_HalideTotal` | 卤素总摩尔分数 |
| | `anion_MixEntropy` | 阴离子混合熵 |
| 结构 (9) | `Density` | 晶体密度 (g/cm³) |
| | `VolumePerAtom` | 每原子体积 (Å³) |
| | `SpaceGroupNumber` | 空间群编号 |
| | `LatticeA` / `LatticeB` / `LatticeC` | 晶格常数 a, b, c (Å) |
| | `LatticeAlpha` / `LatticeBeta` / `LatticeGamma` | 晶格角度 α, β, γ (°) |
| 耦合 (3) | `AnionCation_PhiGap` | \|φ_anion − φ_cation\| |
| | `LiPhi_Coupling` | x_li × φ_cation |
| | `LiHalide_Coupling` | x_li × x_halide_total |

---

## 七、训练过程（train/）

`train/` 文件夹保存了模型从原始数据到最终结果的完整训练流程：

| 子目录 | 内容 |
|--------|------|
| `data/cif/` | 171个CIF晶体结构文件（与JSON记录一一对应） |
| `data/oxidation_state_default_map.json` | 38种元素氧化态映射 |
| `scripts/build_descriptors.py` | 描述符构建：CIF + 化学式 → 20描述符 |
| `scripts/tune_hyperparams.py` | Optuna超参数搜索（100次试验，5折CV） |
| `scripts/compute_shap_all_splits.py` | 全集/训练/测试集SHAP对比分析 |
| `scripts/plot_*.py` | 可视化脚本（散点图、分布图、SHAP图） |
| `reports/model_summary.json` | 模型配置与指标（机器可读） |
| `reports/model_summary.md` | 模型报告（人类可读） |
| `figures/` | 训练过程可视化图（6张） |

---

*模型版本：20-Feature XGBoost (seed=55)*
