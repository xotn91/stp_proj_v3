# -*- coding: utf-8 -*-
"""
P3 helper: build a reproducible subset meta parquet for faster option sweeps.

- Stratified sampling by pair_id anchor (positive rows) using target/ha_bin/cv_fold/time_split
- Keeps temporal split semantics with cutoff_year
- Writes subset parquet + run log(json/txt)
"""

import argparse
import json
import os
from datetime import datetime
from zlib import crc32

import numpy as np
import pandas as pd


def _safe_counts(series, topn=None):
    vc = series.value_counts(dropna=False)
    if topn is not None:
        vc = vc.head(topn)
    out = {}
    for k, v in vc.items():
        out[str(k)] = int(v)
    return out


def _sample_group(group, ratio, min_per_group, seed):
    n_total = len(group)
    if n_total == 0:
        return group
    n_keep = int(round(n_total * ratio))
    n_keep = max(min_per_group, n_keep)
    n_keep = min(n_total, n_keep)
    key = str(group.name).encode("utf-8")
    rs = (seed + crc32(key)) % (2 ** 32)
    return group.sample(n=n_keep, random_state=rs)


def _validate_pair_rule(df, min_neg_per_pair=10, max_neg_per_pair=10):
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


def main():
    parser = argparse.ArgumentParser(description="Build 20% (or custom ratio) subset meta for P3 runs")
    parser.add_argument("--meta_file", default="features_store/final_training_meta.parquet")
    parser.add_argument("--out_dir", default="features_store")
    parser.add_argument("--ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260301)
    parser.add_argument("--cutoff_year", type=int, default=2023)
    parser.add_argument("--min_per_group", type=int, default=1)
    parser.add_argument("--min_neg_per_pair", type=int, default=7)
    parser.add_argument("--max_neg_per_pair", type=int, default=10)
    args = parser.parse_args()

    assert 0.0 < args.ratio <= 1.0, "ratio must be in (0, 1]"
    assert args.min_per_group >= 1, "min_per_group must be >= 1"

    started_at = datetime.now()
    ratio_tag = int(round(args.ratio * 100))
    run_dir = os.path.join(
        args.out_dir,
        f"p3_0_subset_r{ratio_tag}_s{args.seed}_y{args.cutoff_year}_{started_at.strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(run_dir, exist_ok=True)

    print(f">> Loading meta parquet: {args.meta_file}")
    df = pd.read_parquet(args.meta_file)

    required_cols = ["memmap_idx", "target_chembl_id", "set_type", "heavy_atoms", "pair_id"]
    for col in required_cols:
        if col not in df.columns:
            raise AssertionError(f"Required column missing: {col}")

    if "ha_bin" not in df.columns:
        df["ha_bin"] = np.clip(df["heavy_atoms"].values, 10, 60)

    has_publication_year = "publication_year" in df.columns
    if has_publication_year:
        df["time_split"] = np.where(df["publication_year"] > args.cutoff_year, "future", "past")
    else:
        df["time_split"] = "all"

    if "cv_fold" not in df.columns:
        df["cv_fold"] = -1

    pair_quality_total = _validate_pair_rule(
        df,
        min_neg_per_pair=args.min_neg_per_pair,
        max_neg_per_pair=args.max_neg_per_pair,
    )
    if pair_quality_total["pair_rows_below_min_count"] > 0 or pair_quality_total["pair_rows_above_max_count"] > 0:
        raise AssertionError(
            "Input meta violates pair row bounds: "
            f"pair_rows_below_min_count={pair_quality_total['pair_rows_below_min_count']}, "
            f"pair_rows_above_max_count={pair_quality_total['pair_rows_above_max_count']}"
        )
    if pair_quality_total["pair_pos_not_1_count"] > 0:
        raise AssertionError(
            f"Input meta violates pair rule: pair_pos_not_1_count={pair_quality_total['pair_pos_not_1_count']}"
        )
    if pair_quality_total["pair_neg_below_min_count"] > 0 or pair_quality_total["pair_neg_above_max_count"] > 0:
        raise AssertionError(
            "Input meta violates neg bounds: "
            f"pair_neg_below_min_count={pair_quality_total['pair_neg_below_min_count']}, "
            f"pair_neg_above_max_count={pair_quality_total['pair_neg_above_max_count']}"
        )

    pair_anchor = (
        df[df["set_type"] == "Positive"][
            ["pair_id", "target_chembl_id", "ha_bin", "cv_fold", "time_split"]
        ]
        .drop_duplicates(subset=["pair_id"])
        .copy()
    )
    if pair_anchor["pair_id"].nunique() != df["pair_id"].nunique():
        raise AssertionError("pair anchor build failed: each pair_id must have exactly one positive row.")

    pair_strat_cols = ["target_chembl_id", "ha_bin", "cv_fold", "time_split"]
    print(f">> Pair-level stratified sampling by: {pair_strat_cols}")
    grouped = pair_anchor.groupby(pair_strat_cols, dropna=False, group_keys=False)
    try:
        sampled_pair_df = grouped.apply(
            _sample_group, args.ratio, args.min_per_group, args.seed, include_groups=False
        ).copy()
    except TypeError:
        sampled_pair_df = grouped.apply(
            _sample_group, args.ratio, args.min_per_group, args.seed
        ).copy()
    sampled_pair_ids = set(sampled_pair_df["pair_id"].tolist())
    subset_df = df[df["pair_id"].isin(sampled_pair_ids)].copy()

    # Stable ordering for downstream reproducibility.
    if "memmap_idx" in subset_df.columns:
        subset_df = subset_df.sort_values(["memmap_idx"]).reset_index(drop=True)
    else:
        subset_df = subset_df.reset_index(drop=True)

    out_parquet = os.path.join(run_dir, "final_training_meta_subset.parquet")
    subset_df.to_parquet(out_parquet, index=False)

    log = {
        "script": os.path.abspath(__file__),
        "start_time": started_at.isoformat(),
        "end_time": datetime.now().isoformat(),
        "meta_file": args.meta_file,
        "out_dir": run_dir,
        "out_parquet": out_parquet,
        "ratio": args.ratio,
        "seed": args.seed,
        "cutoff_year": args.cutoff_year,
        "min_per_group": args.min_per_group,
        "pair_strat_cols": pair_strat_cols,
        "rows_total": int(len(df)),
        "rows_subset": int(len(subset_df)),
        "subset_fraction_realized": float(len(subset_df) / max(len(df), 1)),
        "pairs_total": int(df["pair_id"].nunique()),
        "pairs_subset": int(subset_df["pair_id"].nunique()),
        "pair_fraction_realized": float(subset_df["pair_id"].nunique() / max(df["pair_id"].nunique(), 1)),
        "has_publication_year": bool(has_publication_year),
        "pair_quality_total": pair_quality_total,
        "pair_quality_subset": _validate_pair_rule(
            subset_df,
            min_neg_per_pair=args.min_neg_per_pair,
            max_neg_per_pair=args.max_neg_per_pair,
        ),
        "summary_total": {
            "set_type": _safe_counts(df["set_type"]),
            "time_split": _safe_counts(df["time_split"]),
            "cv_fold": _safe_counts(df["cv_fold"]),
            "ha_bin_top20": _safe_counts(df["ha_bin"], topn=20),
            "n_targets": int(df["target_chembl_id"].nunique()),
        },
        "summary_subset": {
            "set_type": _safe_counts(subset_df["set_type"]),
            "time_split": _safe_counts(subset_df["time_split"]),
            "cv_fold": _safe_counts(subset_df["cv_fold"]),
            "ha_bin_top20": _safe_counts(subset_df["ha_bin"], topn=20),
            "n_targets": int(subset_df["target_chembl_id"].nunique()),
        },
    }

    log_json = os.path.join(run_dir, "subset_build_log.json")
    with open(log_json, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    log_txt = os.path.join(run_dir, "subset_build_log.txt")
    with open(log_txt, "w", encoding="utf-8") as f:
        f.write(f"start_time={log['start_time']}\n")
        f.write(f"end_time={log['end_time']}\n")
        f.write(f"meta_file={log['meta_file']}\n")
        f.write(f"out_parquet={log['out_parquet']}\n")
        f.write(f"ratio={log['ratio']}\n")
        f.write(f"seed={log['seed']}\n")
        f.write(f"cutoff_year={log['cutoff_year']}\n")
        f.write(f"min_neg_per_pair={args.min_neg_per_pair}\n")
        f.write(f"max_neg_per_pair={args.max_neg_per_pair}\n")
        f.write(f"rows_total={log['rows_total']}\n")
        f.write(f"rows_subset={log['rows_subset']}\n")
        f.write(f"subset_fraction_realized={log['subset_fraction_realized']:.6f}\n")
        f.write(f"pairs_total={log['pairs_total']}\n")
        f.write(f"pairs_subset={log['pairs_subset']}\n")
        f.write(f"pair_fraction_realized={log['pair_fraction_realized']:.6f}\n")
        f.write(f"pair_strat_cols={','.join(pair_strat_cols)}\n")
        f.write(f"pair_rows_below_min_count_total={log['pair_quality_total']['pair_rows_below_min_count']}\n")
        f.write(f"pair_rows_above_max_count_total={log['pair_quality_total']['pair_rows_above_max_count']}\n")
        f.write(f"pair_pos_not_1_count_total={log['pair_quality_total']['pair_pos_not_1_count']}\n")
        f.write(f"pair_neg_below_min_count_total={log['pair_quality_total']['pair_neg_below_min_count']}\n")
        f.write(f"pair_neg_above_max_count_total={log['pair_quality_total']['pair_neg_above_max_count']}\n")
        f.write(f"pair_rows_below_min_count_subset={log['pair_quality_subset']['pair_rows_below_min_count']}\n")
        f.write(f"pair_rows_above_max_count_subset={log['pair_quality_subset']['pair_rows_above_max_count']}\n")
        f.write(f"pair_pos_not_1_count_subset={log['pair_quality_subset']['pair_pos_not_1_count']}\n")
        f.write(f"pair_neg_below_min_count_subset={log['pair_quality_subset']['pair_neg_below_min_count']}\n")
        f.write(f"pair_neg_above_max_count_subset={log['pair_quality_subset']['pair_neg_above_max_count']}\n")
        f.write(f"n_targets_total={log['summary_total']['n_targets']}\n")
        f.write(f"n_targets_subset={log['summary_subset']['n_targets']}\n")

    print(">> Subset build completed")
    print(f"   - subset parquet: {out_parquet}")
    print(f"   - log json: {log_json}")
    print(f"   - log txt: {log_txt}")


if __name__ == "__main__":
    main()
