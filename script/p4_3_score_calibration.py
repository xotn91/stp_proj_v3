# -*- coding: utf-8 -*-
"""
Fit and apply size-bin score-to-precision calibration for p4_0 outputs.

The calibrated value produced here is a size-bin-specific empirical precision
proxy derived from combined_score. It is intended for interpretation and
thresholding support, not as a guaranteed posterior probability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = DEFAULT_PROJECT_ROOT / "data" / "cv_predictions.parquet"
DEFAULT_OUTPUT = DEFAULT_PROJECT_ROOT / "data" / "cv_predictions_score_calibrated.parquet"
DEFAULT_REPORT_DIR = DEFAULT_PROJECT_ROOT / "results" / "p4_3_score_calibration_20260313"


def logistic_score_curve(score: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    """Return a monotonic logistic curve on raw combined_score."""
    score = np.asarray(score, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-(intercept + slope * score)))


def build_empirical_bins(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    n_bins: int,
) -> pd.DataFrame:
    """Build size-bin-specific empirical precision tables on score quantiles."""
    rows: List[dict] = []
    work = df.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work[label_col] = work[label_col].astype(bool)

    for size_bin, group in work.groupby("size_bin", sort=True):
        sub = group.dropna(subset=[score_col]).copy()
        if sub.empty:
            continue

        local_bins = min(int(n_bins), max(2, sub[score_col].nunique()))
        try:
            sub["score_bin_id"] = pd.qcut(sub[score_col], q=local_bins, labels=False, duplicates="drop")
        except ValueError:
            sub["score_bin_id"] = 0

        grouped = (
            sub.groupby("score_bin_id", sort=True)
            .agg(
                score_min=(score_col, "min"),
                score_max=(score_col, "max"),
                score_mean=(score_col, "mean"),
                n_rows=(label_col, "size"),
                n_true=(label_col, "sum"),
            )
            .reset_index(drop=True)
        )
        grouped["empirical_precision"] = np.where(
            grouped["n_rows"] > 0,
            grouped["n_true"] / grouped["n_rows"],
            0.0,
        )

        rows.extend(
            {
                "size_bin": size_bin,
                "score_min": float(row.score_min),
                "score_max": float(row.score_max),
                "score_mean": float(row.score_mean),
                "n_rows": int(row.n_rows),
                "n_true": int(row.n_true),
                "empirical_precision": float(row.empirical_precision),
            }
            for row in grouped.itertuples(index=False)
        )

    return pd.DataFrame(rows)


def fit_score_calibration(empirical_bins: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit size-bin-specific logistic score calibration curves."""
    param_rows: List[dict] = []
    curve_rows: List[dict] = []

    for size_bin, group in empirical_bins.groupby("size_bin", sort=True):
        x = group["score_mean"].to_numpy(dtype=np.float64)
        y = group["empirical_precision"].to_numpy(dtype=np.float64)
        w = group["n_rows"].to_numpy(dtype=np.float64)

        y_clip = np.clip(y, 1.0e-8, 1.0 - 1.0e-8)
        initial_intercept = float(np.log(y_clip.mean() / (1.0 - y_clip.mean())))
        initial_slope = 5.0

        try:
            params, _ = curve_fit(
                logistic_score_curve,
                x,
                y,
                p0=(initial_intercept, initial_slope),
                sigma=np.where(w > 0, 1.0 / np.sqrt(w), 1.0),
                absolute_sigma=False,
                bounds=([-30.0, 0.0], [5.0, 100.0]),
                maxfev=20000,
            )
            intercept, slope = float(params[0]), float(params[1])
        except Exception:
            intercept, slope = initial_intercept, max(initial_slope, 1.0e-6)

        fitted = logistic_score_curve(x, intercept, slope)
        rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))

        param_rows.append(
            {
                "size_bin": size_bin,
                "intercept": intercept,
                "slope": slope,
                "rmse": rmse,
                "empirical_precision_min": float(y.min()),
                "empirical_precision_max": float(y.max()),
                "fitted_at_score_p10": float(logistic_score_curve(np.array([0.10]), intercept, slope)[0]),
                "fitted_at_score_p50": float(logistic_score_curve(np.array([0.50]), intercept, slope)[0]),
                "fitted_at_score_p90": float(logistic_score_curve(np.array([0.90]), intercept, slope)[0]),
            }
        )

        score_grid = np.linspace(float(group["score_min"].min()), float(group["score_max"].max()), 200)
        curve_rows.extend(
            {
                "size_bin": size_bin,
                "score": float(score),
                "fitted_precision": float(fit),
            }
            for score, fit in zip(score_grid, logistic_score_curve(score_grid, intercept, slope))
        )

    return pd.DataFrame(param_rows), pd.DataFrame(curve_rows)


def apply_score_calibration(predictions: pd.DataFrame, param_df: pd.DataFrame) -> pd.DataFrame:
    """Attach size-bin-specific calibrated score precision to each row."""
    work = predictions.copy()
    work["combined_score"] = pd.to_numeric(work["combined_score"], errors="coerce")
    params = param_df.set_index("size_bin")[["intercept", "slope"]].to_dict(orient="index")

    calibrated = np.zeros(len(work), dtype=np.float64)
    for size_bin, group_index in work.groupby("size_bin", sort=False).groups.items():
        if size_bin not in params:
            continue
        intercept = float(params[size_bin]["intercept"])
        slope = float(params[size_bin]["slope"])
        calibrated[group_index] = logistic_score_curve(
            work.loc[group_index, "combined_score"].to_numpy(),
            intercept,
            slope,
        )

    work["calibrated_score_precision"] = calibrated.astype(np.float32)
    return work


def save_plots(empirical_bins: pd.DataFrame, curve_df: pd.DataFrame, output_dir: str | Path) -> None:
    """Save size-bin calibration plots."""
    output_dir = Path(output_dir)
    bins = empirical_bins["size_bin"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(bins), 1, figsize=(8.0, 2.6 * len(bins)), sharex=False)
    if len(bins) == 1:
        axes = [axes]

    for axis, size_bin in zip(axes, bins):
        empirical = empirical_bins[empirical_bins["size_bin"] == size_bin]
        fitted = curve_df[curve_df["size_bin"] == size_bin]
        axis.scatter(empirical["score_mean"], empirical["empirical_precision"], s=18, color="#2563eb", label="Empirical")
        axis.plot(fitted["score"], fitted["fitted_precision"], color="#dc2626", linewidth=1.5, label="Logistic fit")
        axis.set_ylabel(size_bin)
    axes[0].legend(loc="upper left")
    axes[-1].set_xlabel("Combined Score")
    fig.suptitle("Size-Bin Score Calibration Curves", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "01_size_bin_score_calibration.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """Parse arguments, fit score calibration, and export outputs."""
    parser = argparse.ArgumentParser(description="Fit and apply size-bin score calibration.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--score-col", default="combined_score")
    parser.add_argument("--label-col", default="is_true_target")
    parser.add_argument("--n-bins", type=int, default=20)
    args = parser.parse_args()

    output_path = Path(args.output)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(args.input)
    empirical_bins = build_empirical_bins(predictions, args.score_col, args.label_col, args.n_bins)
    param_df, curve_df = fit_score_calibration(empirical_bins)
    calibrated = apply_score_calibration(predictions, param_df)

    calibrated.to_parquet(output_path, index=False)
    empirical_bins.to_csv(report_dir / "empirical_score_bins.csv", index=False)
    param_df.to_csv(report_dir / "score_calibration_params.csv", index=False)
    curve_df.to_csv(report_dir / "score_calibration_curves.csv", index=False)
    save_plots(empirical_bins, curve_df, report_dir)

    summary = {
        "rows": int(len(calibrated)),
        "queries": int(calibrated["query_id"].nunique()),
        "calibrated_score_precision_p10": float(calibrated["calibrated_score_precision"].quantile(0.1)),
        "calibrated_score_precision_p50": float(calibrated["calibrated_score_precision"].quantile(0.5)),
        "calibrated_score_precision_p90": float(calibrated["calibrated_score_precision"].quantile(0.9)),
        "report_dir": str(report_dir),
    }
    with open(output_path.with_suffix(".summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f">> Saved score-calibrated parquet: {output_path}")
    print(f">> Saved score calibration report: {report_dir}")


if __name__ == "__main__":
    main()

