# -*- coding: utf-8 -*-
"""
Validate subset quality against full meta and save evidence artifacts:
- CSV: distribution comparison, target coverage, stratified ratio summary
- JSON/TXT: key statistics
- PNG: comparison plots
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def _ensure_time_split(df, cutoff_year):
    if "publication_year" in df.columns:
        return np.where(df["publication_year"] > cutoff_year, "future", "past")
    return np.array(["all"] * len(df))


def _dist_table(df, col):
    s = df[col].astype("string")
    t = s.value_counts(dropna=False).rename_axis(col).reset_index(name="count")
    total = int(t["count"].sum())
    t["ratio"] = t["count"] / max(total, 1)
    return t


def _js_divergence(p, q):
    eps = 1e-12
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(np.sqrt(0.5 * (kl_pm + kl_qm)))


def _categorical_js(full_df, subset_df, col):
    f = _dist_table(full_df, col)[[col, "ratio"]].rename(columns={"ratio": "full_ratio"})
    s = _dist_table(subset_df, col)[[col, "ratio"]].rename(columns={"ratio": "subset_ratio"})
    m = pd.merge(f, s, on=col, how="outer").fillna(0.0)
    return _js_divergence(m["full_ratio"].values, m["subset_ratio"].values)


def _try_ks_test(full_vals, subset_vals):
    try:
        from scipy.stats import ks_2samp

        stat, pval = ks_2samp(full_vals, subset_vals)
        return {"ks_stat": float(stat), "ks_pvalue": float(pval), "ks_available": True}
    except Exception:
        return {"ks_stat": None, "ks_pvalue": None, "ks_available": False}


def _pair_quality(df, min_neg_per_pair=10, max_neg_per_pair=10):
    pair_sizes = df.groupby("pair_id").size()
    pos_sizes = (
        df[df["set_type"] == "Positive"]
        .groupby("pair_id")
        .size()
        .reindex(pair_sizes.index, fill_value=0)
    )
    neg_sizes = (
        df[df["set_type"] == "Negative"]
        .groupby("pair_id")
        .size()
        .reindex(pair_sizes.index, fill_value=0)
    )
    target_nunique = df.groupby("pair_id")["target_chembl_id"].nunique()
    cv_nunique = (
        df.groupby("pair_id")["cv_fold"].nunique()
        if "cv_fold" in df.columns
        else pd.Series(1, index=pair_sizes.index)
    )
    min_rows = 1 + int(min_neg_per_pair)
    max_rows = 1 + int(max_neg_per_pair)
    return {
        "pairs_total": int(pair_sizes.shape[0]),
        "pair_rows_below_min_count": int((pair_sizes < min_rows).sum()),
        "pair_rows_above_max_count": int((pair_sizes > max_rows).sum()),
        "pair_pos_not_1_count": int((pos_sizes != 1).sum()),
        "pair_neg_below_min_count": int((neg_sizes < int(min_neg_per_pair)).sum()),
        "pair_neg_above_max_count": int((neg_sizes > int(max_neg_per_pair)).sum()),
        "pair_target_mismatch_count": int((target_nunique != 1).sum()),
        "pair_cv_leakage_count": int((cv_nunique != 1).sum()),
    }


def _save_plots(out_dir, full_df, subset_df):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {"plot_available": False, "plot_files": []}

    plot_files = []
    for col in ["set_type", "time_split", "cv_fold"]:
        f = _dist_table(full_df, col).rename(columns={"ratio": "full_ratio"})
        s = _dist_table(subset_df, col).rename(columns={"ratio": "subset_ratio"})
        m = pd.merge(f[[col, "full_ratio"]], s[[col, "subset_ratio"]], on=col, how="outer").fillna(0.0)
        m = m.sort_values(col, key=lambda x: x.astype("string"))

        x = np.arange(len(m))
        w = 0.4
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x - w / 2, m["full_ratio"], width=w, label="full")
        ax.bar(x + w / 2, m["subset_ratio"], width=w, label="subset")
        ax.set_title(f"Distribution Compare: {col}")
        ax.set_xticks(x)
        ax.set_xticklabels(m[col].astype(str), rotation=45, ha="right")
        ax.set_ylabel("ratio")
        ax.legend()
        fig.tight_layout()
        p = os.path.join(out_dir, f"dist_compare_{col}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        plot_files.append(p)

    # heavy_atoms histogram
    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.arange(0, 121, 1)
    ax.hist(full_df["heavy_atoms"].dropna().values, bins=bins, alpha=0.5, density=True, label="full")
    ax.hist(subset_df["heavy_atoms"].dropna().values, bins=bins, alpha=0.5, density=True, label="subset")
    ax.set_title("heavy_atoms density")
    ax.set_xlabel("heavy_atoms")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "dist_compare_heavy_atoms.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    plot_files.append(p)
    return {"plot_available": True, "plot_files": plot_files}


def main():
    parser = argparse.ArgumentParser(description="Validate subset representativeness against full meta")
    parser.add_argument("--full_meta", default="features_store/final_training_meta.parquet")
    parser.add_argument("--subset_meta", required=True)
    parser.add_argument("--out_dir", default="features_store")
    parser.add_argument("--cutoff_year", type=int, default=2023)
    parser.add_argument("--min_neg_per_pair", type=int, default=7)
    parser.add_argument("--max_neg_per_pair", type=int, default=10)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"p3_0_subset_validation_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    full_df = pd.read_parquet(args.full_meta)
    subset_df = pd.read_parquet(args.subset_meta)

    required = ["memmap_idx", "target_chembl_id", "set_type", "heavy_atoms", "pair_id"]
    for col in required:
        if col not in full_df.columns or col not in subset_df.columns:
            raise KeyError(f"required column missing in full/subset: {col}")

    for df in [full_df, subset_df]:
        if "ha_bin" not in df.columns:
            df["ha_bin"] = np.clip(df["heavy_atoms"].values, 10, 60)
        if "cv_fold" not in df.columns:
            df["cv_fold"] = -1
        df["time_split"] = _ensure_time_split(df, args.cutoff_year)

    # 1) Distribution comparison CSV
    cat_cols = ["set_type", "time_split", "cv_fold", "ha_bin"]
    dist_rows = []
    for col in cat_cols:
        f = _dist_table(full_df, col).rename(columns={"count": "full_count", "ratio": "full_ratio"})
        s = _dist_table(subset_df, col).rename(columns={"count": "subset_count", "ratio": "subset_ratio"})
        m = pd.merge(f, s, on=col, how="outer").fillna(0)
        m["column"] = col
        m["ratio_diff_abs"] = (m["subset_ratio"] - m["full_ratio"]).abs()
        dist_rows.append(m[["column", col, "full_count", "subset_count", "full_ratio", "subset_ratio", "ratio_diff_abs"]])
    dist_df = pd.concat(dist_rows, axis=0, ignore_index=True)
    dist_csv = os.path.join(run_dir, "subset_distribution_compare.csv")
    dist_df.to_csv(dist_csv, index=False)

    # 2) target coverage CSV
    t_full = full_df.groupby("target_chembl_id").size().rename("full_count").reset_index()
    t_sub = subset_df.groupby("target_chembl_id").size().rename("subset_count").reset_index()
    t_cov = pd.merge(t_full, t_sub, on="target_chembl_id", how="left").fillna(0)
    t_cov["subset_count"] = t_cov["subset_count"].astype(int)
    t_cov["subset_ratio"] = t_cov.apply(lambda r: _safe_div(r["subset_count"], r["full_count"]), axis=1)
    target_csv = os.path.join(run_dir, "subset_target_coverage.csv")
    t_cov.to_csv(target_csv, index=False)

    # 3) stratified ratio CSV
    strat_cols = ["target_chembl_id", "set_type", "ha_bin", "cv_fold", "time_split"]
    g_full = full_df.groupby(strat_cols).size().rename("full_count").reset_index()
    g_sub = subset_df.groupby(strat_cols).size().rename("subset_count").reset_index()
    g = pd.merge(g_full, g_sub, on=strat_cols, how="left").fillna(0)
    g["subset_count"] = g["subset_count"].astype(int)
    g["subset_ratio"] = g.apply(lambda r: _safe_div(r["subset_count"], r["full_count"]), axis=1)
    strat_csv = os.path.join(run_dir, "subset_stratified_ratio.csv")
    g.to_csv(strat_csv, index=False)

    # 4) Summary JSON/TXT
    pair_quality_full = _pair_quality(
        full_df, min_neg_per_pair=args.min_neg_per_pair, max_neg_per_pair=args.max_neg_per_pair
    )
    pair_quality_subset = _pair_quality(
        subset_df, min_neg_per_pair=args.min_neg_per_pair, max_neg_per_pair=args.max_neg_per_pair
    )
    summary = {
        "full_meta": args.full_meta,
        "subset_meta": args.subset_meta,
        "cutoff_year": args.cutoff_year,
        "rows_full": int(len(full_df)),
        "rows_subset": int(len(subset_df)),
        "subset_fraction_realized": float(len(subset_df) / max(len(full_df), 1)),
        "targets_full": int(full_df["target_chembl_id"].nunique()),
        "targets_subset": int(subset_df["target_chembl_id"].nunique()),
        "target_coverage_ratio": float(
            subset_df["target_chembl_id"].nunique() / max(full_df["target_chembl_id"].nunique(), 1)
        ),
        "pair_coverage_ratio": float(
            subset_df["pair_id"].nunique() / max(full_df["pair_id"].nunique(), 1)
        ),
        "pair_quality_full": pair_quality_full,
        "pair_quality_subset": pair_quality_subset,
        "js_divergence": {
            "set_type": _categorical_js(full_df, subset_df, "set_type"),
            "time_split": _categorical_js(full_df, subset_df, "time_split"),
            "cv_fold": _categorical_js(full_df, subset_df, "cv_fold"),
            "ha_bin": _categorical_js(full_df, subset_df, "ha_bin"),
        },
        "ks_heavy_atoms": _try_ks_test(
            full_df["heavy_atoms"].dropna().values, subset_df["heavy_atoms"].dropna().values
        ),
        "stratified_ratio_summary": {
            "mean": float(g["subset_ratio"].mean()),
            "std": float(g["subset_ratio"].std(ddof=0)),
            "min": float(g["subset_ratio"].min()),
            "max": float(g["subset_ratio"].max()),
            "p05": float(g["subset_ratio"].quantile(0.05)),
            "p50": float(g["subset_ratio"].quantile(0.50)),
            "p95": float(g["subset_ratio"].quantile(0.95)),
        },
        "artifacts": {
            "distribution_csv": dist_csv,
            "target_coverage_csv": target_csv,
            "stratified_ratio_csv": strat_csv,
        },
    }

    plot_meta = _save_plots(run_dir, full_df, subset_df)
    summary["plots"] = plot_meta

    json_path = os.path.join(run_dir, "subset_validation_summary.json")
    txt_path = os.path.join(run_dir, "subset_validation_summary.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"rows_full={summary['rows_full']}\n")
        f.write(f"rows_subset={summary['rows_subset']}\n")
        f.write(f"subset_fraction_realized={summary['subset_fraction_realized']:.6f}\n")
        f.write(f"targets_full={summary['targets_full']}\n")
        f.write(f"targets_subset={summary['targets_subset']}\n")
        f.write(f"target_coverage_ratio={summary['target_coverage_ratio']:.6f}\n")
        f.write(f"pair_coverage_ratio={summary['pair_coverage_ratio']:.6f}\n")
        f.write(f"pair_rows_below_min_full={summary['pair_quality_full']['pair_rows_below_min_count']}\n")
        f.write(f"pair_rows_above_max_full={summary['pair_quality_full']['pair_rows_above_max_count']}\n")
        f.write(f"pair_pos_not_1_full={summary['pair_quality_full']['pair_pos_not_1_count']}\n")
        f.write(f"pair_neg_below_min_full={summary['pair_quality_full']['pair_neg_below_min_count']}\n")
        f.write(f"pair_neg_above_max_full={summary['pair_quality_full']['pair_neg_above_max_count']}\n")
        f.write(f"pair_rows_below_min_subset={summary['pair_quality_subset']['pair_rows_below_min_count']}\n")
        f.write(f"pair_rows_above_max_subset={summary['pair_quality_subset']['pair_rows_above_max_count']}\n")
        f.write(f"pair_pos_not_1_subset={summary['pair_quality_subset']['pair_pos_not_1_count']}\n")
        f.write(f"pair_neg_below_min_subset={summary['pair_quality_subset']['pair_neg_below_min_count']}\n")
        f.write(f"pair_neg_above_max_subset={summary['pair_quality_subset']['pair_neg_above_max_count']}\n")
        f.write(f"js_set_type={summary['js_divergence']['set_type']:.6f}\n")
        f.write(f"js_time_split={summary['js_divergence']['time_split']:.6f}\n")
        f.write(f"js_cv_fold={summary['js_divergence']['cv_fold']:.6f}\n")
        f.write(f"js_ha_bin={summary['js_divergence']['ha_bin']:.6f}\n")
        f.write(f"ks_available={summary['ks_heavy_atoms']['ks_available']}\n")
        if summary["ks_heavy_atoms"]["ks_available"]:
            f.write(f"ks_stat={summary['ks_heavy_atoms']['ks_stat']:.6f}\n")
            f.write(f"ks_pvalue={summary['ks_heavy_atoms']['ks_pvalue']:.6g}\n")
        f.write(f"distribution_csv={dist_csv}\n")
        f.write(f"target_coverage_csv={target_csv}\n")
        f.write(f"stratified_ratio_csv={strat_csv}\n")
        f.write(f"plots_available={plot_meta['plot_available']}\n")
        f.write(f"plot_count={len(plot_meta['plot_files'])}\n")

    print(">> subset validation completed")
    print(f"   - run dir: {run_dir}")
    print(f"   - summary json: {json_path}")
    print(f"   - summary txt: {txt_path}")
    print(f"   - distribution csv: {dist_csv}")
    print(f"   - target coverage csv: {target_csv}")
    print(f"   - stratified ratio csv: {strat_csv}")
    if plot_meta["plot_available"]:
        print(f"   - plots: {len(plot_meta['plot_files'])} files")
    else:
        print("   - plots: skipped (matplotlib unavailable)")


if __name__ == "__main__":
    main()
