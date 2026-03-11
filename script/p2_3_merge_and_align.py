# -*- coding: utf-8 -*-
"""
Phase 2.3+2.4 integrated:
1) Build final_training_meta.parquet by merging master + ES5D meta on molregno
2) Rebuild fp2_aligned.memmap to match ES5D memmap_idx order
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def build_final_training_meta(master_file, es5d_meta_file, out_meta_path):
    print(">> [1/2] Building final_training_meta.parquet")
    df_master = pd.read_parquet(master_file)
    df_es5d = pd.read_parquet(es5d_meta_file)

    required_es5d = ["molregno", "memmap_idx"]
    for col in required_es5d:
        if col not in df_es5d.columns:
            raise KeyError(f"es5d meta missing required column: {col}")

    cols_to_bring = [
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
    ]
    cols_to_bring = [c for c in cols_to_bring if c in df_master.columns]

    df_merged = pd.merge(df_es5d, df_master[cols_to_bring], on="molregno", how="inner")
    df_merged = df_merged.sort_values("memmap_idx").reset_index(drop=True)

    os.makedirs(os.path.dirname(out_meta_path), exist_ok=True)
    df_merged.to_parquet(out_meta_path, index=False, compression="zstd")
    print(f"   - saved: {out_meta_path}")
    print(f"   - rows: {len(df_merged):,}")
    return df_merged


def rebuild_fp2_aligned_memmap(es5d_meta_file, fp2_meta_file, fp2_memmap_file, out_memmap_file):
    print(">> [2/2] Rebuilding fp2_aligned.memmap")
    df_es5d = pd.read_parquet(es5d_meta_file)
    df_fp2 = pd.read_parquet(fp2_meta_file)

    if "molregno" not in df_es5d.columns or "memmap_idx" not in df_es5d.columns:
        raise KeyError("es5d meta must contain molregno and memmap_idx")
    if "molregno" not in df_fp2.columns or "memmap_idx" not in df_fp2.columns:
        raise KeyError("fp2 meta must contain molregno and memmap_idx")

    df_fp2 = df_fp2.rename(columns={"memmap_idx": "fp2_idx"})
    df_align = pd.merge(
        df_es5d[["molregno", "memmap_idx"]],
        df_fp2[["molregno", "fp2_idx"]],
        on="molregno",
        how="inner",
    )
    df_align = df_align.sort_values("memmap_idx").reset_index(drop=True)

    n_final = len(df_align)
    n_fp2_meta = len(df_fp2)
    row_bytes_fp2 = 16 * np.dtype(np.uint64).itemsize
    n_fp2_memmap_rows = os.path.getsize(fp2_memmap_file) // row_bytes_fp2
    if n_fp2_memmap_rows != n_fp2_meta:
        raise ValueError(
            f"fp2 memmap rows ({n_fp2_memmap_rows}) != fp2 meta rows ({n_fp2_meta})"
        )

    fp2_src = np.memmap(
        fp2_memmap_file, dtype=np.uint64, mode="r", shape=(n_fp2_memmap_rows, 16)
    )
    fp2_out = np.memmap(out_memmap_file, dtype=np.uint64, mode="w+", shape=(n_final, 16))
    fp2_out[:] = fp2_src[df_align["fp2_idx"].values]
    fp2_out.flush()

    print(f"   - saved: {out_memmap_file}")
    print(f"   - aligned rows: {n_final:,}")
    return {
        "n_final": int(n_final),
        "n_fp2_meta": int(n_fp2_meta),
        "n_fp2_memmap_rows": int(n_fp2_memmap_rows),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Integrated P2.3+P2.4: merge metadata and align FP2 memmap"
    )
    parser.add_argument(
        "--master_file", default="chembl36_stp_training_set_final_v2.parquet"
    )
    parser.add_argument("--es5d_meta_file", default="features_store/es5d_meta_db.parquet")
    parser.add_argument("--fp2_meta_file", default="features_store/fp2_meta.parquet")
    parser.add_argument("--fp2_memmap_file", default="features_store/fp2_uint64.memmap")
    parser.add_argument(
        "--out_meta_file", default="features_store/final_training_meta.parquet"
    )
    parser.add_argument("--out_fp2_aligned", default="features_store/fp2_aligned.memmap")
    parser.add_argument(
        "--out_manifest", default="features_store/p2_3_4_merge_align_manifest.json"
    )
    args = parser.parse_args()

    merged_df = build_final_training_meta(
        master_file=args.master_file,
        es5d_meta_file=args.es5d_meta_file,
        out_meta_path=args.out_meta_file,
    )

    align_stats = rebuild_fp2_aligned_memmap(
        es5d_meta_file=args.es5d_meta_file,
        fp2_meta_file=args.fp2_meta_file,
        fp2_memmap_file=args.fp2_memmap_file,
        out_memmap_file=args.out_fp2_aligned,
    )

    manifest = {
        "master_file": args.master_file,
        "es5d_meta_file": args.es5d_meta_file,
        "fp2_meta_file": args.fp2_meta_file,
        "fp2_memmap_file": args.fp2_memmap_file,
        "out_meta_file": args.out_meta_file,
        "out_fp2_aligned": args.out_fp2_aligned,
        "merged_rows": int(len(merged_df)),
        **align_stats,
    }
    os.makedirs(os.path.dirname(args.out_manifest), exist_ok=True)
    with open(args.out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f">> manifest saved: {args.out_manifest}")


if __name__ == "__main__":
    main()
