# 电化学窗口 ECW 测试/训练总包

> 建议 Linux 上传目标路径：`/work/share/acitw40es7/ND_test/1.2Ionic_ecw_test/ecw_window_test_package/`
> 建议环境名：`test_ecw`

---

## 目录结构

```text
ecw_window_test_package/
├─ test/     交付测试包：直接运行脚本即可输出结果文件和图片
├─ train/    训练复现包：包含原始数据、CIF、20 描述符显式构建与训练复现
├─ env/      Linux 环境安装文件
└─ README.md
```

## 一、环境安装

```bash
cd /work/share/acitw40es7/ND_test/1.2Ionic_ecw_test/ecw_window_test_package/env
bash setup_env.sh
conda activate test_ecw
```

## 二、测试流程

进入：

```bash
cd /work/share/acitw40es7/ND_test/1.2Ionic_ecw_test/ecw_window_test_package/test
```

### 1）预测测试 + 预测图

```bash
python test_ecw_predict.py
```

输出：
- `result_ecw.csv`
- `result_ecw.json`
- `window_model_rank.joblib`
- `ecw_prediction_result.png`

### 2）SHAP 排名 + SHAP 图

```bash
python run_ecw_shap.py
```

输出：
- `key_factors_window_ranking.csv`
- `key_factors_window_ranking.json`
- `ecw_shap_feature_importance.png`

### 3）如需仅根据输出文件重绘图片

```bash
python plot_ecw_prediction_result.py
python plot_ecw_shap_feature_importance.py
```

## 三、训练复现流程

进入：

```bash
cd /work/share/acitw40es7/ND_test/1.2Ionic_ecw_test/ecw_window_test_package/train/scripts
```

### 1）显式构建最终 20 描述符

```bash
python build_final_20_feature_matrix.py
```

### 2）对比重建结果与测试矩阵

```bash
python compare_rebuilt_descriptors.py
```

### 3）复现最终训练/测试指标

```bash
python reproduce_final_20_model.py
```

训练侧关键内容：
- `train/data/source/`：原始 CSV 数据
- `train/data/cif/`：全部 CIF/结构文件
- `train/scripts/`：显式 20 描述符构建、对比、训练复现
- `train/configs/`：最终模型超参数
- `train/models/`：训练复现模型
- `train/outputs/`：特征矩阵、对比结果、训练输出
- `train/figures/`：训练图像输出

## 四、最终 20 描述符名称

- `Bandgap`
- `packing_fraction`
- `wavg_GSbandgap`
- `wavg_IsNonmetal`
- `wavg_ZungerPP-r_sigma`
- `wavg_phi_Miedema`
- `n_elements`
- `wavg_NdValence`
- `wavg_NsValence`
- `VolumePerAtom`
- `wavg_Electronegativity`
- `wavg_Anion_Electronegativity`
- `wavg_NdUnfilled`
- `wavg_GSestFCClatcnt`
- `wavg_HeatCapacityMolar`
- `wavg_FirstIonizationEnergy`
- `wavg_GSmagmom`
- `wavg_MeltingT`
- `wavg_ElectronAffinity`
- `wavg_CovalentRadius`

## 五、说明

- `test/` 面向交付测试，脚本内部已直接串联绘图。
- `train/` 面向训练复现，描述符构建使用原始数据 + CIF/结构文件。
- 所有路径均已改为包内相对路径，便于后续上传到 Linux 后直接运行。
