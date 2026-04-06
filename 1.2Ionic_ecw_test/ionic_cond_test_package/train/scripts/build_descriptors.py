"""
build_descriptors.py — 从 CIF 晶体结构文件 + 化学式计算 20 个描述符
=======================================================================
输入:
  - data_ionic_conductivity.json   (171 条实验记录)
  - data/cif/                      (171 个 CIF 文件，与 JSON 记录一一对应)
  - data/oxidation_state_default_map.json

  数据清洗说明:
    原始文献中收集了若干 CIF 晶体结构文件。经过以下筛选后保留 171 条有效记录:
    1. 从 JSON 实验记录中提取出 171 条含有效离子电导率数据的结构
    2. 删除无对应 JSON 记录的多余 CIF 文件（无实验电导率数据，无法建模）
    3. 删除无效的 CIF 结构（解析失败或数据不完整）
    最终保证 JSON 记录与 CIF 文件严格一一对应，均为 171 条。

输出:
  - feature_ionic_matrix.csv       (record_id + 20 列描述符)

描述符分三类 (共 20 个):
  A. 组分描述符 (8 个)  — 由化学式 + matminer/pymatgen 计算
  B. 结构描述符 (9 个)  — 由 CIF 晶体结构解析
  C. 耦合描述符 (3 个)  — 由阳/阴离子离子势 (phi_cation/phi_anion) 派生

依赖: pymatgen, matminer, numpy, pandas
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.structure.symmetry import GlobalSymmetryFeatures
from pymatgen.core import Composition, Element, Species, Structure

warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF*")

# ── 路径 ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent          # training_process/
ROOT_DIR = BASE_DIR.parent                       # ionic_cond_standard_test/
DATA_DIR = BASE_DIR / "data"
CIF_DIR  = DATA_DIR / "cif"
OXI_MAP  = DATA_DIR / "oxidation_state_default_map.json"
INPUT_JSON = ROOT_DIR / "data_ionic_conductivity.json"
OUTPUT_CSV = ROOT_DIR / "feature_ionic_matrix.csv"

ANION_ELEMENTS = {"F", "Cl", "Br", "I", "O", "S", "Se", "Te", "N"}

# 最终 20 描述符列名（标准测试包命名）
FEAT_20_COLUMNS = [
    "n_elements", "x_li",
    "wavg_AtomicNumber", "wavg_AtomicWeight", "wavg_CovalentRadius",
    "ratio_CationToAnionIonicPotential", "x_HalideTotal", "anion_MixEntropy",
    "AnionCation_PhiGap", "LiPhi_Coupling", "LiHalide_Coupling",
    "Density", "VolumePerAtom", "SpaceGroupNumber",
    "LatticeA", "LatticeB", "LatticeC",
    "LatticeAlpha", "LatticeBeta", "LatticeGamma",
]


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def read_oxi_map(path: Path) -> dict[str, int]:
    """读取氧化态默认映射表"""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in payload.get("chosen_oxidation_states", {}).items()}


def safe_element_symbol(raw: str) -> str:
    """从物种字符串中提取纯元素符号"""
    match = re.match(r"([A-Z][a-z]?)", str(raw))
    return match.group(1) if match else str(raw)


def infer_oxidation_state(element: str) -> int:
    """根据 pymatgen 常见氧化态推断默认值"""
    el = Element(element)
    common = [int(x) for x in el.common_oxidation_states]
    if common:
        if element in ANION_ELEMENTS:
            negatives = [x for x in common if x < 0]
            if negatives:
                return min(negatives)
        positives = [x for x in common if x > 0]
        if positives:
            return positives[0]
        return common[0]
    if element in ANION_ELEMENTS:
        return -1 if element in {"F", "Cl", "Br", "I"} else -2
    return 1


def get_ionic_radius(element: str, oxidation_state: int) -> float:
    """获取离子半径 (Å)，回退到原子半径"""
    symbol = safe_element_symbol(element)
    try:
        spec = Species(symbol, oxidation_state)
        ir = spec.ionic_radius
        if ir is not None:
            v = float(ir)
            if np.isfinite(v) and v > 0:
                return v
    except Exception:
        pass
    try:
        el = Element(symbol)
        for candidate in (el.atomic_radius_calculated, el.atomic_radius):
            if candidate is not None:
                v = float(candidate)
                if np.isfinite(v) and v > 0:
                    return v
    except Exception:
        pass
    return float("nan")


def safe_entropy(probs: list[float]) -> float:
    """计算香农混合熵 -Σ p·ln(p)"""
    vals = [float(p) for p in probs if p > 1e-15]
    if not vals:
        return float("nan")
    return float(-sum(p * math.log(p) for p in vals))


# ── matminer 特征提取器 ──────────────────────────────────────────────────────

def make_featurizers() -> tuple[ElementProperty, ElementProperty]:
    """
    Magpie 数据源: 原子序数、原子量、共价半径 (组分加权平均)
    Deml  数据源: (备用，本脚本未直接使用，但保留接口)
    """
    magpie = ElementProperty(
        data_source="magpie",
        features=["Number", "AtomicWeight", "CovalentRadius"],
        stats=["mean"],
    )
    deml = ElementProperty(
        data_source="deml",
        features=["first_ioniz", "electron_affin"],
        stats=["mean"],
    )
    return magpie, deml


# ── 单条记录描述符计算 ────────────────────────────────────────────────────────

def compute_one_record(
    record_id: str,
    formula: str,
    cif_path: Path,
    oxi_map: dict[str, int],
    magpie: ElementProperty,
    gsf: GlobalSymmetryFeatures,
) -> dict[str, float]:
    """
    从 CIF + 化学式计算全部 20 个描述符 + 2 个中间变量。

    计算流程:
      Step 1: 解析化学式 → 组分摩尔分数
      Step 2: matminer Magpie → 原子序数/原子量/共价半径 加权平均
      Step 3: 氧化态 + 离子半径 → 离子势 φ = |oxi|/r
      Step 4: 解析 CIF → 密度、体积/原子、空间群、晶格参数
      Step 5: 阴离子分数 → 卤素总分数、阴离子混合熵
      Step 6: 耦合特征 = 组分 × 电位 交叉项
    """
    structure = Structure.from_file(str(cif_path))
    comp = Composition(formula).fractional_composition
    frac = {safe_element_symbol(k): float(v) for k, v in comp.as_dict().items()}

    # ── Step 2: matminer 组分加权平均 ──
    mag_vals = magpie.featurize(comp)
    mag_map = dict(zip(magpie.feature_labels(), mag_vals))

    # ── Step 3: 离子势 φ = |氧化态| / 离子半径 ──
    cat_num, cat_den = 0.0, 0.0
    an_num, an_den = 0.0, 0.0

    for el, x in frac.items():
        oxi = float(oxi_map.get(el, infer_oxidation_state(el)))
        r = get_ionic_radius(el, int(oxi))
        if np.isfinite(r) and r > 0:
            phi = abs(oxi) / r
            if oxi > 0:
                cat_num += x * phi
                cat_den += x
            elif oxi < 0:
                an_num += x * phi
                an_den += x

    phi_cation = cat_num / cat_den if cat_den > 0 else float("nan")
    phi_anion = an_num / an_den if an_den > 0 else float("nan")
    phi_ratio = (
        phi_cation / phi_anion
        if np.isfinite(phi_cation) and np.isfinite(phi_anion) and phi_anion > 0
        else float("nan")
    )

    # ── Step 4: CIF 结构属性 ──
    sym_vals = gsf.featurize(structure)
    sym_map = dict(zip(gsf.feature_labels(), sym_vals))

    density = float(structure.density)
    n_atoms = structure.composition.num_atoms
    vpa = float(structure.volume / n_atoms) if n_atoms > 0 else float("nan")

    # ── Step 5: 阴离子分数 + 混合熵 ──
    x_halide = sum(float(frac.get(h, 0.0)) for h in ("F", "Cl", "Br", "I"))
    anion_probs = [float(v) for k, v in frac.items() if k in ANION_ELEMENTS]
    anion_entropy = safe_entropy(anion_probs)

    # ── Step 6: 耦合特征 ──
    x_li = float(frac.get("Li", 0.0))
    li_phi_coupling = x_li * phi_cation if np.isfinite(phi_cation) else float("nan")
    li_halide_coupling = x_li * x_halide
    phi_gap = (
        abs(phi_anion - phi_cation)
        if np.isfinite(phi_anion) and np.isfinite(phi_cation)
        else float("nan")
    )

    return {
        "record_id": record_id,
        # A. 组分描述符 (8)
        "n_elements": float(len(frac)),
        "x_li": x_li,
        "wavg_AtomicNumber": float(mag_map.get("MagpieData mean Number", np.nan)),
        "wavg_AtomicWeight": float(mag_map.get("MagpieData mean AtomicWeight", np.nan)),
        "wavg_CovalentRadius": float(mag_map.get("MagpieData mean CovalentRadius", np.nan)),
        "ratio_CationToAnionIonicPotential": phi_ratio,
        "x_HalideTotal": x_halide,
        "anion_MixEntropy": anion_entropy,
        # B. 结构描述符 (9)
        "Density": density,
        "VolumePerAtom": vpa,
        "SpaceGroupNumber": float(sym_map.get("spacegroup_num", np.nan)),
        "LatticeA": float(structure.lattice.a),
        "LatticeB": float(structure.lattice.b),
        "LatticeC": float(structure.lattice.c),
        "LatticeAlpha": float(structure.lattice.alpha),
        "LatticeBeta": float(structure.lattice.beta),
        "LatticeGamma": float(structure.lattice.gamma),
        # C. 耦合描述符 (3)
        "AnionCation_PhiGap": phi_gap,
        "LiPhi_Coupling": li_phi_coupling,
        "LiHalide_Coupling": li_halide_coupling,
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  描述符构建: CIF + 化学式 → 20 描述符")
    print("=" * 60)

    # 1. 读取实验记录
    with open(INPUT_JSON, encoding="utf-8") as f:
        records = json.load(f)
    print(f"[1/4] 读取 {len(records)} 条实验记录")

    # 2. 初始化
    oxi_map = read_oxi_map(OXI_MAP)
    magpie, _ = make_featurizers()
    gsf = GlobalSymmetryFeatures()
    print(f"[2/4] 氧化态映射: {len(oxi_map)} 种元素, CIF 目录: {CIF_DIR}")

    # 3. 逐条计算
    rows: list[dict] = []
    failures: list[dict] = []

    for i, rec in enumerate(records):
        rid = str(rec["record_id"])
        formula = str(rec["formula"])

        # 匹配 CIF 文件: record_id 即为文件名前缀
        cif_candidates = list(CIF_DIR.glob(f"{rid}*.cif"))
        if not cif_candidates:
            cif_candidates = list(CIF_DIR.glob(f"{rid}.cif"))
        if not cif_candidates:
            failures.append({"record_id": rid, "error": "CIF not found"})
            continue

        cif_path = cif_candidates[0]
        try:
            row = compute_one_record(rid, formula, cif_path, oxi_map, magpie, gsf)
            rows.append(row)
        except Exception as exc:
            failures.append({"record_id": rid, "error": str(exc)})

        if (i + 1) % 20 == 0:
            print(f"      ... {i + 1}/{len(records)} 完成")

    print(f"[3/4] 计算完成: 成功 {len(rows)}, 失败 {len(failures)}")

    # 4. 输出
    df = pd.DataFrame(rows)
    output_cols = ["record_id"] + FEAT_20_COLUMNS
    df = df[[c for c in output_cols if c in df.columns]]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[4/4] 已保存: {OUTPUT_CSV}")
    print(f"      形状: {df.shape[0]} 行 × {df.shape[1]} 列")

    if failures:
        fail_csv = BASE_DIR / "reports" / "build_failures.csv"
        pd.DataFrame(failures).to_csv(fail_csv, index=False, encoding="utf-8-sig")
        print(f"      失败记录: {fail_csv}")

    print("=" * 60)


if __name__ == "__main__":
    main()
