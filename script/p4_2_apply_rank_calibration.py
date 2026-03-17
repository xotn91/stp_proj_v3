# -*- coding: utf-8 -*-
"""
Apply size-bin logistic rank calibration to p4_0 prediction logs.

The calibrated value produced here is a rank-based hit-rate proxy derived from
query-level empirical hit curves. It is therefore not a direct per-target
posterior probability, and the output column is named accordingly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = DEFAULT_PROJECT_ROOT / "data" / "cv_predictions.parquet"
DEFAULT_ANALYSIS_DIR = DEFAULT_PROJECT_ROOT / "results" / "p4_1_cv_analysis_20260313"
DEFAULT_OUTPUT = DEFAULT_PROJECT_ROOT / "data" / "cv_predictions_rank_calibrated.parquet"


def logistic_rank_curve(rank: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    """Return the monotonic logistic curve used in p4_1 analysis."""
    safe_rank = np.maximum(np.asarray(rank, dtype=np.float64), 1.0)
    return 1.0 / (1.0 + np.exp(-(intercept + slope * np.log(safe_rank))))


def load_fit_params(analysis_dir: str | Path) -> pd.DataFrame:
    """Load size-bin logistic parameters exported by p4_1 analysis."""
    path = Path(analysis_dir) / "size_bin_logistic_params.csv"
    fit_params = pd.read_csv(path)
    required = {"size_bin", "intercept", "slope"}
    missing = required.difference(fit_params.columns)
    if missing:
        raise KeyError(f"Missing fit parameter columns: {sorted(missing)}")
    return fit_params


def apply_rank_calibration(predictions: pd.DataFrame, fit_params: pd.DataFrame) -> pd.DataFrame:
    """Attach calibrated rank-based hit-rate proxies to each prediction row."""
    work = predictions.copy()
    work["rank"] = pd.to_numeric(work["rank"], errors="coerce").fillna(0).astype(np.int64)

    params = fit_params.set_index("size_bin")[["intercept", "slope"]].to_dict(orient="index")
    calibrated = np.zeros(len(work), dtype=np.float64)

    for size_bin, group_index in work.groupby("size_bin", sort=False).groups.items():
        if size_bin not in params:
            continue
        intercept = float(params[size_bin]["intercept"])
        slope = float(params[size_bin]["slope"])
        calibrated[group_index] = logistic_rank_curve(work.loc[group_index, "rank"].to_numpy(), intercept, slope)

    work["calibrated_rank_hit_rate"] = calibrated.astype(np.float32)
    return work


def build_summary(calibrated_df: pd.DataFrame, analysis_dir: str | Path) -> dict:
    """Build a compact summary of calibrated values."""
    out = {
        "rows": int(len(calibrated_df)),
        "queries": int(calibrated_df["query_id"].nunique()),
        "calibrated_rank_hit_rate_min": float(calibrated_df["calibrated_rank_hit_rate"].min()),
        "calibrated_rank_hit_rate_p10": float(calibrated_df["calibrated_rank_hit_rate"].quantile(0.1)),
        "calibrated_rank_hit_rate_p50": float(calibrated_df["calibrated_rank_hit_rate"].quantile(0.5)),
        "calibrated_rank_hit_rate_p90": float(calibrated_df["calibrated_rank_hit_rate"].quantile(0.9)),
        "calibrated_rank_hit_rate_max": float(calibrated_df["calibrated_rank_hit_rate"].max()),
        "analysis_dir": str(analysis_dir),
    }
    return out


def main() -> None:
    """Parse arguments and write calibrated parquet plus a short JSON summary."""
    parser = argparse.ArgumentParser(description="Apply size-bin logistic rank calibration.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    predictions = pd.read_parquet(args.input)
    fit_params = load_fit_params(args.analysis_dir)
    calibrated_df = apply_rank_calibration(predictions, fit_params)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibrated_df.to_parquet(output_path, index=False)

    summary = build_summary(calibrated_df, args.analysis_dir)
    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f">> Saved calibrated parquet: {output_path}")
    print(f">> Saved calibration summary: {summary_path}")


if __name__ == "__main__":
    main()

