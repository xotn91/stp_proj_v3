#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_df(path: Path, cols=None):
    return pd.read_parquet(path, columns=cols)


def _pick_candidates(cands_ge100, cands_fallback, existing_set, need, rng):
    picked = []

    if need <= 0:
        return picked

    ge100 = [m for m in cands_ge100 if m not in existing_set]
    rng.shuffle(ge100)
    take_ge100 = ge100[:need]
    picked.extend(take_ge100)
    existing_set.update(take_ge100)
    need -= len(take_ge100)

    if need <= 0:
        return picked

    fb = [m for m in cands_fallback if m not in existing_set]
    rng.shuffle(fb)
    take_fb = fb[:need]
    picked.extend(take_fb)
    existing_set.update(take_fb)
    return picked


def main():
    ap = argparse.ArgumentParser(description="Augment negatives in current operating mode (>=100uM priority, keep underfilled pairs).")
    ap.add_argument("--base_meta", required=True)
    ap.add_argument("--train_parquet", required=True)
    ap.add_argument("--es5d_meta", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_neg_per_pair", type=int, default=10)
    ap.add_argument("--prefer_ge_uM", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_meta = Path(args.base_meta)
    train_parquet = Path(args.train_parquet)
    es5d_meta = Path(args.es5d_meta)

    need_cols = [
        "memmap_idx",
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
        "activity_uM",
    ]
    df = _load_df(base_meta)
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"base_meta missing columns: {missing}")

    df = df[need_cols].copy()

    # ES5D-available molecule universe (required to keep memmap compatibility)
    es5d = _load_df(es5d_meta, cols=["memmap_idx", "molregno"]).drop_duplicates("molregno")
    mol2idx = dict(zip(es5d["molregno"].astype(int), es5d["memmap_idx"].astype(int)))

    train = _load_df(
        train_parquet,
        cols=[
            "molregno",
            "mol_chembl_id",
            "target_chembl_id",
            "heavy_atoms",
            "set_type",
            "scaffold_smiles",
            "publication_year",
            "assay_id",
            "activity_uM",
        ],
    )
    train = train[train["set_type"] == "Negative"].copy()
    train["molregno"] = train["molregno"].astype("int64")

    # Keep only molecules that already have ES5D row.
    train = train[train["molregno"].isin(mol2idx.keys())].copy()

    # Dedup candidate keys
    cand = train.sort_values(["target_chembl_id", "molregno"]).drop_duplicates(["target_chembl_id", "molregno"])

    cand_ge100 = cand[cand["activity_uM"].notna() & (cand["activity_uM"] >= float(args.prefer_ge_uM))]
    cand_fallback = cand[cand["activity_uM"].isna()]

    pool_ge100 = cand_ge100.groupby("target_chembl_id")["molregno"].agg(list).to_dict()
    pool_fallback = cand_fallback.groupby("target_chembl_id")["molregno"].agg(list).to_dict()

    # Metadata for appended rows
    mol_meta = (
        cand.sort_values(["molregno", "target_chembl_id", "publication_year"], na_position="last")
        .drop_duplicates(["molregno", "target_chembl_id"], keep="first")
        [["molregno", "mol_chembl_id", "target_chembl_id", "heavy_atoms", "scaffold_smiles", "publication_year"]]
    )

    mol_meta_t = {
        (int(r.molregno), str(r.target_chembl_id)): {
            "mol_chembl_id": r.mol_chembl_id,
            "heavy_atoms": r.heavy_atoms,
            "scaffold_smiles": r.scaffold_smiles,
            "publication_year": r.publication_year,
        }
        for r in mol_meta.itertuples(index=False)
    }

    pos = df[df["set_type"] == "Positive"].copy()
    neg = df[df["set_type"] == "Negative"].copy()

    pos_per_pair = pos.groupby("pair_id").size()
    if (pos_per_pair != 1).any():
        raise ValueError("All pairs must have exactly 1 positive row in base_meta.")

    pair_pos = pos[["pair_id", "target_chembl_id", "cv_fold", "assay_id"]].drop_duplicates("pair_id")
    pair_target = dict(zip(pair_pos["pair_id"], pair_pos["target_chembl_id"]))
    pair_cv = dict(zip(pair_pos["pair_id"], pair_pos["cv_fold"]))

    pair_neg_count = neg.groupby("pair_id").size().to_dict()
    pair_existing = neg.groupby("pair_id")["molregno"].agg(set).to_dict()

    rng = np.random.default_rng(args.seed)
    appended_rows = []
    shortage_rows = []

    for pid in pos["pair_id"].drop_duplicates().tolist():
        tgt = pair_target[pid]
        have = int(pair_neg_count.get(pid, 0))
        need = int(args.target_neg_per_pair) - have
        if need <= 0:
            continue

        existing = set(pair_existing.get(pid, set()))

        c_ge = list(pool_ge100.get(tgt, []))
        c_fb = list(pool_fallback.get(tgt, []))
        picked = _pick_candidates(c_ge, c_fb, existing, need, rng)

        if len(picked) < need:
            shortage_rows.append(
                {
                    "pair_id": pid,
                    "target_chembl_id": tgt,
                    "have_neg": have,
                    "need_to_10": need,
                    "added": len(picked),
                    "shortage": need - len(picked),
                }
            )

        for molregno in picked:
            key = (int(molregno), str(tgt))
            md = mol_meta_t.get(key)
            if md is None:
                # should be rare; skip and log as shortage equivalent
                shortage_rows.append(
                    {
                        "pair_id": pid,
                        "target_chembl_id": tgt,
                        "have_neg": have,
                        "need_to_10": need,
                        "added": 0,
                        "shortage": 1,
                        "reason": "missing_mol_meta",
                        "molregno": int(molregno),
                    }
                )
                continue
            appended_rows.append(
                {
                    "memmap_idx": int(mol2idx[int(molregno)]),
                    "molregno": int(molregno),
                    "mol_chembl_id": md["mol_chembl_id"],
                    "target_chembl_id": tgt,
                    "heavy_atoms": md["heavy_atoms"],
                    "set_type": "Negative",
                    "pair_id": pid,
                    "cv_fold": pair_cv[pid],
                    "scaffold_smiles": md["scaffold_smiles"],
                    "publication_year": md["publication_year"],
                    "assay_id": pd.NA,
                    "activity_uM": np.nan,
                }
            )

    add_df = pd.DataFrame(appended_rows, columns=need_cols)
    out_df = pd.concat([df, add_df], ignore_index=True)

    # pair-level duplicate negative guard
    neg_out = out_df[out_df["set_type"] == "Negative"][['pair_id', 'molregno']].copy()
    dup_in_pair = int(neg_out.duplicated(['pair_id', 'molregno']).sum())
    if dup_in_pair > 0:
        raise ValueError(f"Duplicate negatives found within pair after augmentation: {dup_in_pair}")

    out_meta = out_dir / "final_training_meta.parquet"
    out_add = out_dir / "delta_added_negatives.parquet"
    out_short = out_dir / "shortage_pairs_to_fill10.csv"
    out_manifest = out_dir / "augment_manifest.json"

    out_df.to_parquet(out_meta, index=False, compression="zstd")
    add_df.to_parquet(out_add, index=False, compression="zstd")

    short_df = pd.DataFrame(shortage_rows)
    if not short_df.empty:
        short_df = short_df.sort_values(["shortage", "pair_id"], ascending=[False, True])
    short_df.to_csv(out_short, index=False)

    pair_counts = (
        out_df.assign(pos=(out_df["set_type"] == "Positive").astype(int), neg=(out_df["set_type"] == "Negative").astype(int))
        .groupby("pair_id")[['pos', 'neg']]
        .sum()
    )

    manifest = {
        "inputs": {
            "base_meta": str(base_meta),
            "train_parquet": str(train_parquet),
            "es5d_meta": str(es5d_meta),
        },
        "policy": {
            "mode": "current_operation",
            "target_neg_per_pair": int(args.target_neg_per_pair),
            "prefer_ge_uM": float(args.prefer_ge_uM),
            "keep_underfilled_pairs": True,
            "fallback_to_nan_negative": True,
            "drop_underfilled": False,
        },
        "counts": {
            "base_rows": int(len(df)),
            "base_pairs": int(df["pair_id"].nunique()),
            "added_rows": int(len(add_df)),
            "final_rows": int(len(out_df)),
            "final_pairs": int(out_df["pair_id"].nunique()),
            "pairs_neg_lt10": int((pair_counts["neg"] < int(args.target_neg_per_pair)).sum()),
            "pairs_neg_eq10": int((pair_counts["neg"] == int(args.target_neg_per_pair)).sum()),
            "pairs_neg_gt10": int((pair_counts["neg"] > int(args.target_neg_per_pair)).sum()),
            "dup_negative_within_pair": dup_in_pair,
            "shortage_pairs_logged": int(len(short_df)),
        },
        "outputs": {
            "final_meta": str(out_meta),
            "delta_added": str(out_add),
            "shortage_log": str(out_short),
        },
    }
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(str(out_manifest))


if __name__ == "__main__":
    main()
