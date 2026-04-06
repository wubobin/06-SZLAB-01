#!/usr/bin/env python
"""Explicitly rebuild the final 20 ECW descriptors from raw data plus CIF files."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matminer.featurizers.structure import DensityFeatures
from matminer.utils.data import MagpieData
from pymatgen.core import Composition, Element, Structure


warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF*")

ID_COL = "MP ID"
FORMULA_COL = "Formula"
BANDGAP_COL = "Band Gap (eV)"
TARGET_COL = "Window Length (V)"
ANION_FALLBACK_SET = {"F", "Cl", "Br", "I", "O", "S", "Se", "Te", "N"}

PROTOCOL_FEATURES = [
    "Bandgap",
    "packing_fraction",
    "wavg_GSbandgap",
    "wavg_IsNonmetal",
    "wavg_ZungerPP-r_sigma",
    "wavg_phi_Miedema",
    "n_elements",
    "wavg_NdValence",
    "wavg_NsValence",
    "VolumePerAtom",
    "wavg_Electronegativity",
    "wavg_Anion_Electronegativity",
    "wavg_NdUnfilled",
    "wavg_GSestFCClatcnt",
    "wavg_HeatCapacityMolar",
    "wavg_FirstIonizationEnergy",
    "wavg_GSmagmom",
    "wavg_MeltingT",
    "wavg_ElectronAffinity",
    "wavg_CovalentRadius",
]

ROOT = Path(__file__).resolve().parent
TRAIN_ROOT = ROOT.parent
PACKAGE_ROOT = TRAIN_ROOT.parent
SOURCE_DIR = TRAIN_ROOT / "data" / "source"
STRUCTURE_DIR = TRAIN_ROOT / "data" / "cif"
OUTPUT_DIR = TRAIN_ROOT / "outputs"
DEFAULT_N_JOBS = max(1, min(8, os.cpu_count() or 1))

_WORKER_STATE: dict[str, Any] = {}


def _get_worker_state() -> dict[str, Any]:
    state = _WORKER_STATE.get("state")
    if state is None:
        state = {
            "magpie": MagpieData(),
            "density": DensityFeatures(),
        }
        _WORKER_STATE["state"] = state
    return state


def to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def weighted_mean(values: Iterable[float], weights: np.ndarray) -> float:
    arr = np.asarray(list(values), dtype=float)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return float("nan")
    valid_values = arr[mask]
    valid_weights = np.asarray(weights, dtype=float)[mask]
    weight_sum = float(valid_weights.sum())
    if weight_sum <= 0:
        return float(valid_values.mean())
    return float(np.dot(valid_values, valid_weights / weight_sum))


def _select_anions_by_oxidation_state(comp: Composition) -> list[str]:
    try:
        guesses = comp.oxi_state_guesses(max_sites=-1)
    except Exception:
        guesses = []
    if not guesses:
        return []
    anions: list[str] = []
    for element, oxidation_state in guesses[0].items():
        symbol = element.symbol if hasattr(element, "symbol") else str(element)
        if to_float(oxidation_state) < 0:
            anions.append(symbol)
    return anions


def _fallback_anion_symbols(comp: Composition) -> list[str]:
    symbols = [element.symbol for element in comp.elements]
    preferred = [symbol for symbol in symbols if symbol in ANION_FALLBACK_SET]
    if preferred:
        return preferred

    electronegativities: list[tuple[str, float]] = []
    for symbol in symbols:
        value = Element(symbol).X
        if value is not None:
            electronegativities.append((symbol, float(value)))
    if not electronegativities:
        return []

    max_value = max(value for _, value in electronegativities)
    return [symbol for symbol, value in electronegativities if abs(value - max_value) < 1e-12]


def compute_anion_weighted_electronegativity(formula: str) -> float:
    comp = Composition(formula)
    stoichiometry = comp.get_el_amt_dict()
    anion_symbols = _select_anions_by_oxidation_state(comp)
    if not anion_symbols:
        anion_symbols = _fallback_anion_symbols(comp)

    numerator = 0.0
    denominator = 0.0
    for symbol in anion_symbols:
        value = Element(symbol).X
        amount = to_float(stoichiometry.get(symbol, 0.0))
        if value is None or not np.isfinite(amount) or amount <= 0:
            continue
        numerator += amount * float(value)
        denominator += amount

    if denominator > 0:
        return float(numerator / denominator)

    fallback_num = 0.0
    fallback_den = 0.0
    for symbol, amount in stoichiometry.items():
        value = Element(symbol).X
        amount_float = to_float(amount)
        if value is None or not np.isfinite(amount_float) or amount_float <= 0:
            continue
        fallback_num += amount_float * float(value)
        fallback_den += amount_float
    return float(fallback_num / fallback_den) if fallback_den > 0 else float("nan")


def compute_structure_descriptors(cif_path: Path) -> dict[str, float]:
    structure = Structure.from_file(cif_path)
    density_featurizer = _get_worker_state()["density"]
    density_values = density_featurizer.featurize(structure)
    density_map = dict(zip(density_featurizer.feature_labels(), density_values))
    return {
        "packing_fraction": to_float(density_map.get("packing fraction")),
        "VolumePerAtom": to_float(density_map.get("vpa")),
    }


def compute_one_record(record: dict[str, Any], cif_dir: Path | None = None) -> dict[str, float]:
    if cif_dir is None:
        cif_dir = STRUCTURE_DIR

    worker_state = _get_worker_state()
    magpie: MagpieData = worker_state["magpie"]

    mp_id = str(record[ID_COL])
    formula = str(record[FORMULA_COL])
    cif_path = Path(cif_dir) / f"{mp_id}.cif"
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF not found for {mp_id}: {cif_path}")

    composition = Composition(formula).fractional_composition
    elements = list(composition.elements)
    weights = np.asarray([float(composition[element]) for element in elements], dtype=float)

    bandgap = to_float(record[BANDGAP_COL])
    structure_features = compute_structure_descriptors(cif_path)

    wavg_gsbandgap = weighted_mean(
        [to_float(magpie.get_elemental_property(element, "GSbandgap")) for element in elements],
        weights,
    )
    wavg_is_nonmetal = weighted_mean(
        [to_float(magpie.get_elemental_property(element, "IsNonmetal")) for element in elements],
        weights,
    )
    wavg_zunger_pp_r_sigma = weighted_mean(
        [to_float(magpie.get_elemental_property(element, "ZungerPP-r_sigma")) for element in elements],
        weights,
    )
    wavg_phi_miedema = weighted_mean(
        [to_float(magpie.get_elemental_property(element, "phi")) for element in elements],
        weights,
    )
    n_elements = float(len(elements))
    wavg_nd_valence = weighted_mean(
        [to_float(magpie.get_elemental_property(element, "NdValence")) for element in elements],
        weights,
    )
    wavg_ns_valence = weighted_mean(
        [to_float(magpie.get_elemental_property(element, "NsValence")) for element in elements],
        weights,
    )
    wavg_electronegativity = weighted_mean(
        [to_float(magpie.get_elemental_property(element, "Electronegativity")) for element in elements],
        weights,
    )
    wavg_anion_electronegativity = compute_anion_weighted_electronegativity(formula)
    wavg_nd_unfilled = weighted_mean(
        [to_float(magpie.get_elemental_property(element, "NdUnfilled")) for element in elements],
        weights,
    )
    wavg_gsest_fcclatcnt = weighted_mean(
        [to_float(magpie.get_elemental_property(element, "GSestFCClatcnt")) for element in elements],
        weights,
    )
    wavg_heat_capacity_molar = weighted_mean(
        [to_float(magpie.get_elemental_property(element, "HeatCapacityMolar")) for element in elements],
        weights,
    )
    wavg_first_ionization_energy = weighted_mean(
        [to_float(magpie.get_elemental_property(element, "FirstIonizationEnergy")) for element in elements],
        weights,
    )
    wavg_gsmagmom = weighted_mean(
        [to_float(magpie.get_elemental_property(element, "GSmagmom")) for element in elements],
        weights,
    )
    wavg_melting_t = weighted_mean(
        [to_float(magpie.get_elemental_property(element, "MeltingT")) for element in elements],
        weights,
    )
    wavg_electron_affinity = weighted_mean(
        [to_float(magpie.get_elemental_property(element, "ElectronAffinity")) for element in elements],
        weights,
    )
    wavg_covalent_radius = weighted_mean(
        [to_float(magpie.get_elemental_property(element, "CovalentRadius")) for element in elements],
        weights,
    )

    return {
        ID_COL: mp_id,
        TARGET_COL: to_float(record[TARGET_COL]),
        "Bandgap": bandgap,
        "packing_fraction": structure_features["packing_fraction"],
        "wavg_GSbandgap": wavg_gsbandgap,
        "wavg_IsNonmetal": wavg_is_nonmetal,
        "wavg_ZungerPP-r_sigma": wavg_zunger_pp_r_sigma,
        "wavg_phi_Miedema": wavg_phi_miedema,
        "n_elements": n_elements,
        "wavg_NdValence": wavg_nd_valence,
        "wavg_NsValence": wavg_ns_valence,
        "VolumePerAtom": structure_features["VolumePerAtom"],
        "wavg_Electronegativity": wavg_electronegativity,
        "wavg_Anion_Electronegativity": wavg_anion_electronegativity,
        "wavg_NdUnfilled": wavg_nd_unfilled,
        "wavg_GSestFCClatcnt": wavg_gsest_fcclatcnt,
        "wavg_HeatCapacityMolar": wavg_heat_capacity_molar,
        "wavg_FirstIonizationEnergy": wavg_first_ionization_energy,
        "wavg_GSmagmom": wavg_gsmagmom,
        "wavg_MeltingT": wavg_melting_t,
        "wavg_ElectronAffinity": wavg_electron_affinity,
        "wavg_CovalentRadius": wavg_covalent_radius,
    }


def _build_records(raw_df: pd.DataFrame, cif_dir: Path, n_jobs: int) -> list[dict[str, float]]:
    records = raw_df[[ID_COL, FORMULA_COL, BANDGAP_COL, TARGET_COL]].to_dict("records")
    if n_jobs == 1:
        return [compute_one_record(record, cif_dir=cif_dir) for record in records]
    return Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(compute_one_record)(record, cif_dir=cif_dir) for record in records
    )


def build_feature_dataframe(cif_dir: Path | None = None, n_jobs: int = DEFAULT_N_JOBS) -> tuple[pd.DataFrame, list[str]]:
    if cif_dir is None:
        cif_dir = STRUCTURE_DIR

    raw_path = SOURCE_DIR / "ecw_mp_16k_final.csv"
    raw_df = pd.read_csv(raw_path)
    raw_df = raw_df[raw_df[TARGET_COL].notna()].copy()
    rows = _build_records(raw_df=raw_df, cif_dir=Path(cif_dir), n_jobs=max(1, int(n_jobs)))
    feature_df = pd.DataFrame(rows)
    ordered_columns = [ID_COL, TARGET_COL, *PROTOCOL_FEATURES]
    feature_df = feature_df[ordered_columns]
    return feature_df, PROTOCOL_FEATURES.copy()


def build_feature_matrix(output_dir: Path | None = None, n_jobs: int = DEFAULT_N_JOBS) -> dict[str, object]:
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_df, selected_features = build_feature_dataframe(n_jobs=n_jobs)
    feature_matrix_path = output_dir / "feature_matrix_final_20.csv"
    feature_list_path = output_dir / "feature_list_final_20.json"

    feature_df.to_csv(feature_matrix_path, index=False)
    feature_list_path.write_text(
        json.dumps({"selected_features": selected_features}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "feature_matrix_path": feature_matrix_path,
        "feature_list_path": feature_list_path,
        "selected_features": selected_features,
        "n_samples": int(len(feature_df)),
    }


def main() -> None:
    result = build_feature_matrix()
    print(
        json.dumps(
            {
                "feature_matrix_path": str(result["feature_matrix_path"]),
                "feature_list_path": str(result["feature_list_path"]),
                "selected_features": result["selected_features"],
                "n_samples": result["n_samples"],
                "source_raw_csv": str(SOURCE_DIR / "ecw_mp_16k_final.csv"),
                "source_cif_dir": str(STRUCTURE_DIR),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
