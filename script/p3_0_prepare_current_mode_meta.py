#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _neg_pick_order(df):
    out = df.copy()
    out["has_measured_activity"] = out["activity_uM"].notna().astype(np.int8)
    out["activity_rank"] = out["activity_uM"].fillna(-1.0)
    return out.sort_values(
        ["has_measured_activity", "activity_rank", "publication_year", "molregno"],
        ascending=[False, False, True, True],
        na_position="last",
    )


def main():
    ap = argparse.ArgumentParser(
        description="Prepare a current-mode P3 meta set from a <10uM-positive training parquet using existing ES5D/FP2 alignment."
    )
    ap.add_argument(
        "--master_parquet",
        default="/mnt/d/stp_proj_v3/features_store/chembl36_stp_training_set.neg_ge100_aug.parquet",
    )
    ap.add_argument(
        "--es5d_meta",
        default="/mnt/d/stp_proj_v3/features_store/es5d_meta_db.parquet",
    )
    ap.add_argument(
        "--out_dir",
        default="/mnt/d/stp_proj_v3/features_store/p3_ready_pos_lt10_neg7plus_20260316",
    )
    ap.add_argument("--min_neg_per_pair", type=int, default=7)
    ap.add_argument("--max_neg_per_pair", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master_cols = [
        "molregno",
        "mol_chembl_id",
        "target_chembl_id",
        "heavy_atoms",
        "set_type",
        "pair_id",
        "cv_fold",
        "scaffold_smiles",
        "publication_year",
        "assay_id",
        "assay_chembl_id",
        "activity_uM",
    ]

    master = pd.read_parquet(args.master_parquet)
    missing = [c for c in master_cols if c not in master.columns]
    if missing:
        raise KeyError(f"master parquet missing required columns: {missing}")
    master = master.loc[:, master_cols].copy()

    es5d = pd.read_parquet(args.es5d_meta, columns=["molregno", "memmap_idx"])
    merged = pd.merge(master, es5d, on="molregno", how="inner")

    pair_pos_count = merged[merged["set_type"] == "Positive"].groupby("pair_id").size()
    pair_neg_count = merged[merged["set_type"] == "Negative"].groupby("pair_id").size()
    all_pairs = pd.Index(sorted(set(merged["pair_id"].tolist())))
    pair_pos_count = pair_pos_count.reindex(all_pairs, fill_value=0)
    pair_neg_count = pair_neg_count.reindex(all_pairs, fill_value=0)

    drop_pos_bad = set(all_pairs[pair_pos_count != 1])
    drop_neg_too_few = set(all_pairs[pair_neg_count < int(args.min_neg_per_pair)])

    kept_parts = []
    pair_neg_not10_rows = []
    pair_dropped_rows = []

    for pair_id, group in merged.groupby("pair_id", sort=False):
        pos_rows = group[group["set_type"] == "Positive"].copy()
        neg_rows = group[group["set_type"] == "Negative"].copy()
        n_pos = len(pos_rows)
        n_neg = len(neg_rows)

        if pair_id in drop_pos_bad:
            pair_dropped_rows.append(
                {
                    "pair_id": pair_id,
                    "reason": "pos_not_1_after_es5d_merge",
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                }
            )
            continue

        if pair_id in drop_neg_too_few:
            pair_dropped_rows.append(
                {
                    "pair_id": pair_id,
                    "reason": "neg_below_min_after_es5d_merge",
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                }
            )
            continue

        neg_kept = neg_rows.copy()
        trim_reason = None
        if n_neg > int(args.max_neg_per_pair):
            neg_kept = _neg_pick_order(neg_rows).head(int(args.max_neg_per_pair)).copy()
            trim_reason = "trimmed_to_max_neg"

        neg_final = len(neg_kept)
        if neg_final != int(args.max_neg_per_pair):
            pair_neg_not10_rows.append(
                {
                    "pair_id": pair_id,
                    "status": "kept_underfilled",
                    "n_pos": n_pos,
                    "n_neg": neg_final,
                }
            )
        elif trim_reason is not None:
            pair_neg_not10_rows.append(
                {
                    "pair_id": pair_id,
                    "status": trim_reason,
                    "n_pos": n_pos,
                    "n_neg_before": n_neg,
                    "n_neg_after": neg_final,
                }
            )

        kept_parts.append(pos_rows)
        kept_parts.append(neg_kept)

    final_df = pd.concat(kept_parts, ignore_index=True)
    final_df = final_df.sort_values(["memmap_idx", "pair_id", "set_type", "molregno"]).reset_index(drop=True)

    neg_out = final_df[final_df["set_type"] == "Negative"][["pair_id", "molregno"]].copy()
    dup_in_pair = int(neg_out.duplicated(["pair_id", "molregno"]).sum())
    if dup_in_pair > 0:
        raise ValueError(f"Duplicate negatives found within pair after preparation: {dup_in_pair}")

    pair_sizes = final_df.groupby("pair_id").size()
    pos_sizes = (
        final_df[final_df["set_type"] == "Positive"]
        .groupby("pair_id")
        .size()
        .reindex(pair_sizes.index, fill_value=0)
    )
    neg_sizes = (
        final_df[final_df["set_type"] == "Negative"]
        .groupby("pair_id")
        .size()
        .reindex(pair_sizes.index, fill_value=0)
    )

    out_meta = out_dir / "final_training_meta.parquet"
    out_pair_not10 = out_dir / "pair_neg_not10_log.csv"
    out_pair_drop = out_dir / "pair_dropped_log.csv"
    out_manifest = out_dir / "build_manifest.json"

    final_df.to_parquet(out_meta, index=False, compression="zstd")
    pd.DataFrame(pair_neg_not10_rows).to_csv(out_pair_not10, index=False)
    pd.DataFrame(pair_dropped_rows).to_csv(out_pair_drop, index=False)

    manifest = {
        "inputs": {
            "master_parquet": str(args.master_parquet),
            "es5d_meta": str(args.es5d_meta),
            "shared_fp2_aligned_memmap": "/mnt/d/stp_proj_v3/features_store/fp2_aligned.memmap",
            "shared_es5d_memmap": "/mnt/d/stp_proj_v3/features_store/es5d_db_k20.memmap",
        },
        "outputs": {
            "out_dir": str(out_dir),
            "final_training_meta": str(out_meta),
            "pair_neg_not10_log": str(out_pair_not10),
            "pair_dropped_log": str(out_pair_drop),
        },
        "policy": {
            "positive_rule": "activity_uM < 10 (from master parquet)",
            "negative_rule": ">=100uM priority already reflected in master parquet",
            "min_neg_per_pair": int(args.min_neg_per_pair),
            "max_neg_per_pair": int(args.max_neg_per_pair),
            "keep_7_to_9": True,
            "drop_below_min": True,
            "trim_above_max": True,
            "trim_strategy": "prefer measured >=100uM, then higher activity_uM, then stable key order",
        },
        "counts": {
            "merged_rows": int(len(merged)),
            "merged_pairs": int(merged["pair_id"].nunique()),
            "final_rows": int(len(final_df)),
            "final_pairs": int(final_df["pair_id"].nunique()),
            "pair_pos_not_1_after_merge": int((pair_pos_count != 1).sum()),
            "pair_neg_below_min_after_merge": int((pair_neg_count < int(args.min_neg_per_pair)).sum()),
            "pair_neg_7_to_9_kept": int(((neg_sizes >= int(args.min_neg_per_pair)) & (neg_sizes < int(args.max_neg_per_pair))).sum()),
            "pair_neg_eq10": int((neg_sizes == int(args.max_neg_per_pair)).sum()),
            "pair_neg_gt10_after_trim": int((neg_sizes > int(args.max_neg_per_pair)).sum()),
            "pairs_trimmed_from_gt10": int(sum(r.get("status") == "trimmed_to_max_neg" for r in pair_neg_not10_rows)),
            "pairs_dropped_total": int(len(pair_dropped_rows)),
            "dup_negative_within_pair": dup_in_pair,
        },
    }
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(str(out_manifest))


if __name__ == "__main__":
    main()
