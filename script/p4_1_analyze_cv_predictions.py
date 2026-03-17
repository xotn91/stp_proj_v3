# -*- coding: utf-8 -*-
"""
Analyze p4_0 leave-one-out outputs and fit size-bin logistic hit curves.

This script reads the parquet and sidecar summary files produced by
p4_0_generate_cv_predictions.py, exports top-k evaluation tables, and fits a
monotonic logistic curve to the empirical query-level hit-rate trajectory for
each size bin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_STEM = DEFAULT_PROJECT_ROOT / "data" / "cv_predictions"
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT_ROOT / "results" / "p4_1_cv_analysis"


def logistic_rank_curve(rank: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    """Return a monotonic logistic curve on log-rank."""
    safe_rank = np.maximum(np.asarray(rank, dtype=np.float64), 1.0)
    return 1.0 / (1.0 + np.exp(-(intercept + slope * np.log(safe_rank))))


def load_outputs(data_stem: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load p4_0 parquet sidecars using a common filename stem."""
    stem = Path(data_stem)
    query_summary = pd.read_csv(f"{stem}.query_summary.csv")
    size_bin_precision = pd.read_csv(f"{stem}.size_bin_precision.csv")
    with open(f"{stem}.run_summary.json", "r", encoding="utf-8") as handle:
        run_summary = json.load(handle)
    return query_summary, size_bin_precision, run_summary


def build_topk_tables(query_summary: pd.DataFrame, cutoffs: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build overall and size-bin query-level top-k hit tables."""
    work = query_summary.copy()
    work["best_true_rank"] = pd.to_numeric(work["best_true_rank"], errors="coerce")
    cutoffs = [int(k) for k in cutoffs]

    overall_rows: List[dict] = []
    by_size_rows: List[dict] = []
    for cutoff in cutoffs:
        overall_rows.append(
            {
                "cutoff": cutoff,
                "query_hit_rate": float(work["best_true_rank"].le(cutoff).mean()),
            }
        )
        for size_bin, group in work.groupby("size_bin", sort=True):
            by_size_rows.append(
                {
                    "size_bin": size_bin,
                    "cutoff": cutoff,
                    "query_hit_rate": float(group["best_true_rank"].le(cutoff).mean()),
                }
            )

    return pd.DataFrame(overall_rows), pd.DataFrame(by_size_rows)


def fit_size_bin_logistic_curves(
    query_summary: pd.DataFrame,
    max_cutoff: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit monotonic logistic curves to empirical query-level hit trajectories."""
    work = query_summary.copy()
    work["best_true_rank"] = pd.to_numeric(work["best_true_rank"], errors="coerce")

    fit_rows: List[dict] = []
    curve_rows: List[dict] = []
    cutoffs = np.arange(1, int(max_cutoff) + 1, dtype=np.float64)

    for size_bin, group in work.groupby("size_bin", sort=True):
        empirical = np.array([group["best_true_rank"].le(int(k)).mean() for k in cutoffs], dtype=np.float64)
        clipped = np.clip(empirical, 1.0e-6, 1.0 - 1.0e-6)
        initial_intercept = float(np.log(clipped[0] / (1.0 - clipped[0])))
        initial_slope = 0.5

        try:
            params, _ = curve_fit(
                logistic_rank_curve,
                cutoffs,
                empirical,
                p0=(initial_intercept, initial_slope),
                bounds=([-20.0, 0.0], [20.0, 20.0]),
                maxfev=20000,
            )
            intercept, slope = float(params[0]), float(params[1])
        except Exception:
            intercept, slope = initial_intercept, max(initial_slope, 1.0e-6)

        fitted = logistic_rank_curve(cutoffs, intercept, slope)
        rmse = float(np.sqrt(np.mean((fitted - empirical) ** 2)))

        fit_rows.append(
            {
                "size_bin": size_bin,
                "intercept": intercept,
                "slope": slope,
                "rmse": rmse,
                "empirical_top1": float(empirical[0]),
                "empirical_top5": float(empirical[4]) if len(empirical) >= 5 else np.nan,
                "empirical_top15": float(empirical[14]) if len(empirical) >= 15 else np.nan,
                "empirical_top100": float(empirical[99]) if len(empirical) >= 100 else np.nan,
                "fitted_top1": float(fitted[0]),
                "fitted_top5": float(fitted[4]) if len(fitted) >= 5 else np.nan,
                "fitted_top15": float(fitted[14]) if len(fitted) >= 15 else np.nan,
                "fitted_top100": float(fitted[99]) if len(fitted) >= 100 else np.nan,
            }
        )

        curve_rows.extend(
            {
                "size_bin": size_bin,
                "cutoff": int(rank),
                "empirical_hit_rate": float(emp),
                "fitted_hit_rate": float(fit_val),
            }
            for rank, emp, fit_val in zip(cutoffs.astype(int), empirical, fitted)
        )

    return pd.DataFrame(fit_rows), pd.DataFrame(curve_rows)


def save_plots(
    topk_overall: pd.DataFrame,
    curve_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save compact plots for top-k hit rates and logistic fitted curves."""
    output_dir = Path(output_dir)

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(topk_overall["cutoff"], topk_overall["query_hit_rate"], marker="o", color="#2563eb")
    plt.xscale("log")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Rank Cutoff (log scale)")
    plt.ylabel("Query-Level Hit Rate")
    plt.title("Overall Top-K Hit Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "01_overall_topk_curve.png", dpi=160)
    plt.close()

    unique_bins = curve_df["size_bin"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(unique_bins), 1, figsize=(8.0, 2.6 * len(unique_bins)), sharex=True)
    if len(unique_bins) == 1:
        axes = [axes]
    for axis, size_bin in zip(axes, unique_bins):
        sub = curve_df[curve_df["size_bin"] == size_bin]
        axis.plot(sub["cutoff"], sub["empirical_hit_rate"], color="#9ca3af", linewidth=1.2, label="Empirical")
        axis.plot(sub["cutoff"], sub["fitted_hit_rate"], color="#dc2626", linewidth=1.5, label="Logistic fit")
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel(size_bin)
    axes[0].legend(loc="lower right")
    axes[-1].set_xlabel("Rank Cutoff")
    fig.suptitle("Size-Bin Logistic Hit-Curve Fits", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "02_size_bin_logistic_fits.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """Parse arguments and export analysis outputs."""
    parser = argparse.ArgumentParser(description="Analyze p4_0 cv_predictions outputs.")
    parser.add_argument("--data-stem", default=str(DEFAULT_DATA_STEM))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-cutoff", type=int, default=300)
    parser.add_argument("--cutoffs", nargs="*", type=int, default=[1, 5, 10, 15, 50, 100, 250, 500])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_summary, size_bin_precision, run_summary = load_outputs(args.data_stem)
    topk_overall, topk_by_size = build_topk_tables(query_summary, args.cutoffs)
    fit_params, fit_curves = fit_size_bin_logistic_curves(query_summary, args.max_cutoff)

    topk_overall.to_csv(output_dir / "topk_overall.csv", index=False)
    topk_by_size.to_csv(output_dir / "topk_by_size_bin.csv", index=False)
    fit_params.to_csv(output_dir / "size_bin_logistic_params.csv", index=False)
    fit_curves.to_csv(output_dir / "size_bin_logistic_curves.csv", index=False)
    size_bin_precision.to_csv(output_dir / "size_bin_precision_copy.csv", index=False)
    with open(output_dir / "run_summary_copy.json", "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)

    save_plots(topk_overall, fit_curves, output_dir)
    print(f">> Saved analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()

