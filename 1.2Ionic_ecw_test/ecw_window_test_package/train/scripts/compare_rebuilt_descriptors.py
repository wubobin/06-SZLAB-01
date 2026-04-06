#!/usr/bin/env python
"""Compare explicitly rebuilt ECW descriptors against the packaged feature matrix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_final_20_feature_matrix import ID_COL, PROTOCOL_FEATURES, TARGET_COL, build_feature_matrix


ROOT = Path(__file__).resolve().parent
TRAIN_ROOT = ROOT.parent
PACKAGE_ROOT = TRAIN_ROOT.parent
OUTPUT_DIR = TRAIN_ROOT / "outputs"
REFERENCE_MATRIX_PATH = PACKAGE_ROOT / "test" / "feature_ecw_matrix.csv"


def _column_summary(rebuilt: pd.Series, reference: pd.Series) -> dict[str, object]:
    rebuilt_num = pd.to_numeric(rebuilt, errors="coerce")
    reference_num = pd.to_numeric(reference, errors="coerce")

    exact_mask = rebuilt_num.eq(reference_num) | (rebuilt.isna() & reference.isna())
    close_mask = np.isclose(
        rebuilt_num.to_numpy(dtype=float),
        reference_num.to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    abs_diff = np.abs(rebuilt_num - reference_num)

    return {
        "exact_equal": bool(np.all(exact_mask.to_numpy())),
        "allclose_at_1e-12": bool(np.all(close_mask)),
        "max_abs_diff": float(abs_diff.max(skipna=True)) if abs_diff.notna().any() else 0.0,
        "mean_abs_diff": float(abs_diff.mean(skipna=True)) if abs_diff.notna().any() else 0.0,
        "mismatch_count_exact": int((~exact_mask).sum()),
        "mismatch_count_allclose": int((~close_mask).sum()),
    }


def compare_rebuilt_descriptors(
    rebuilt_path: Path | None = None,
    reference_path: Path = REFERENCE_MATRIX_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if rebuilt_path is None:
        build_result = build_feature_matrix(output_dir=output_dir)
        rebuilt_path = Path(build_result["feature_matrix_path"])
    else:
        rebuilt_path = Path(rebuilt_path)

    rebuilt_df = pd.read_csv(rebuilt_path)
    reference_df = pd.read_csv(reference_path)

    compare_cols = [TARGET_COL, *PROTOCOL_FEATURES]
    rebuilt_cmp = rebuilt_df[[ID_COL, *compare_cols]].copy()
    reference_cmp = reference_df[[ID_COL, *compare_cols]].copy()
    merged = rebuilt_cmp.merge(reference_cmp, on=ID_COL, suffixes=("_rebuilt", "_reference"), how="inner")

    if len(merged) != len(reference_df):
        raise ValueError(
            f"Row count mismatch after merge: merged={len(merged)}, reference={len(reference_df)}"
        )

    summary_rows: list[dict[str, object]] = []
    for column in compare_cols:
        stats = _column_summary(
            rebuilt=merged[f"{column}_rebuilt"],
            reference=merged[f"{column}_reference"],
        )
        summary_rows.append({"column": column, **stats})

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "descriptor_comparison_summary.csv"
    summary_json = output_dir / "descriptor_comparison_summary.json"
    summary_df.to_csv(summary_csv, index=False)

    result = {
        "rebuilt_path": str(rebuilt_path),
        "reference_path": str(reference_path),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "row_count": int(len(merged)),
        "all_exact_equal": bool(summary_df["exact_equal"].all()),
        "all_allclose_at_1e-12": bool(summary_df["allclose_at_1e-12"].all()),
        "columns": summary_rows,
    }
    summary_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    result = compare_rebuilt_descriptors()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
