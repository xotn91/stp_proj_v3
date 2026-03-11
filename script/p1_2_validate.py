# -*- coding: utf-8 -*-
"""
P1 Validation
- Validate P1 dataset against project rules
"""

import os
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "features_store" / "chembl36_stp_training_set.parquet"
REPORT = ROOT / "results" / "p1_validation_report.json"

REQUIRED_COLS = [
    "molregno", "inchikey", "mol_chembl_id", "canonical_smiles", "heavy_atoms", "scaffold_smiles",
    "target_chembl_id", "target_name", "target_type", "organism", "uniprot_id", "sequence",
    "standard_value", "standard_type", "standard_units", "activity_uM", "label",
    "publication_year", "confidence_score", "source_version", "set_type", "cv_fold", "pair_id"
]


def main():
    df = pd.read_parquet(PARQUET)
    report = {}

    # Column check
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    report["missing_columns"] = missing_cols

    # Organism
    allowed_org = {"Homo sapiens", "Mus musculus", "Rattus norvegicus"}
    report["invalid_organism_count"] = int((~df["organism"].isin(allowed_org)).sum())

    # Target type
    allowed_tt = {"SINGLE PROTEIN", "PROTEIN COMPLEX"}
    report["invalid_target_type_count"] = int((~df["target_type"].isin(allowed_tt)).sum())

    # Heavy atoms
    report["max_heavy_atoms"] = float(df["heavy_atoms"].max())
    report["heavy_atoms_over_80"] = int((df["heavy_atoms"] > 80).sum())

    # Pair size
    pair_counts = df.groupby("pair_id").size()
    report["pair_not_11_count"] = int((pair_counts != 11).sum())

    # Target consistency within pair
    target_nunique = df.groupby("pair_id")["target_chembl_id"].nunique()
    report["pair_target_mismatch_count"] = int((target_nunique != 1).sum())

    # Positive/Negative overlap
    pos_pairs = set(zip(df[df["set_type"] == "Positive"]["mol_chembl_id"],
                        df[df["set_type"] == "Positive"]["target_chembl_id"]))
    neg_pairs = set(zip(df[df["set_type"] == "Negative"]["mol_chembl_id"],
                        df[df["set_type"] == "Negative"]["target_chembl_id"]))
    report["pos_neg_overlap_count"] = int(len(pos_pairs.intersection(neg_pairs)))

    # CV leakage
    cv_leakage = df.groupby("pair_id")["cv_fold"].nunique()
    report["cv_leakage_count"] = int((cv_leakage != 1).sum())

    # Activity thresholds
    # Positive should be < 10 uM, Negative should be >= 500 uM or NaN (Assumed)
    pos_bad = df[(df["set_type"] == "Positive") & (df["activity_uM"] >= 10)]
    neg_bad = df[(df["set_type"] == "Negative") & (df["activity_uM"].notna()) & (df["activity_uM"] < 500)]
    report["positive_threshold_violations"] = int(len(pos_bad))
    report["negative_threshold_violations"] = int(len(neg_bad))

    # Save report
    os.makedirs(os.path.dirname(str(REPORT)), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
