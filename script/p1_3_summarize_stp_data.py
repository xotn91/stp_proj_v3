# -*- coding: utf-8 -*-
"""
P1 Summary
- Summarize P1_1 dataset for species-level table, heavy atom distribution, and year distribution.

Default input:
- /mnt/d/stp_proj_v3/features_store/chembl36_stp_training_set.parquet

Default outputs:
- /mnt/d/stp_proj_v3/results/p1_summary_table.csv
- /mnt/d/stp_proj_v3/results/p1_heavy_atoms_distribution.png
- /mnt/d/stp_proj_v3/results/p1_publication_year_distribution.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "features_store" / "chembl36_stp_training_set.parquet"
DEFAULT_OUT_DIR = ROOT / "results"


def build_species_summary(pos_df: pd.DataFrame) -> pd.DataFrame:
    organisms = ["Homo sapiens", "Rattus norvegicus", "Mus musculus"]
    rows = []
    for org in organisms:
        sub = pos_df[pos_df["organism"] == org]
        rows.append(
            {
                "Species": org,
                "Number of targets": int(sub["target_chembl_id"].nunique()),
                "Number of active compounds": int(sub["mol_chembl_id"].nunique()),
                "Number of interactions": int(len(sub)),
            }
        )

    rows.append(
        {
            "Species": "All (Total)",
            "Number of targets": int(pos_df["target_chembl_id"].nunique()),
            "Number of active compounds": int(pos_df["mol_chembl_id"].nunique()),
            "Number of interactions": int(len(pos_df)),
        }
    )
    return pd.DataFrame(rows)


def save_heavy_atoms_plot(pos_df: pd.DataFrame, out_png: Path) -> dict:
    unique_mols = pos_df.drop_duplicates(subset=["mol_chembl_id"])
    median_val = float(unique_mols["heavy_atoms"].median())

    plt.figure(figsize=(10, 6))
    plt.hist(unique_mols["heavy_atoms"], bins=list(range(0, 85, 2)), color="gray", edgecolor="none")
    plt.axvline(median_val, color="black", linestyle="dashed", linewidth=2, label=f"Median ({median_val:.1f})")
    plt.title("Distribution of the number of heavy atoms (ChEMBL 36 Actives)")
    plt.xlabel("Number of heavy atoms")
    plt.ylabel("Distribution (Counts)")
    plt.xlim(0, 80)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    return {
        "unique_active_molecules": int(unique_mols["mol_chembl_id"].nunique()),
        "heavy_atoms_median": median_val,
        "heavy_atoms_max": float(unique_mols["heavy_atoms"].max()),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize P1_1 dataset outputs.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input parquet path from P1_1.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory to save summary outputs.")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    df = pd.read_parquet(input_path)
    if "set_type" not in df.columns:
        raise KeyError("Required column missing: set_type")

    pos_df = df[df["set_type"] == "Positive"].copy()
    if pos_df.empty:
        raise ValueError("No Positive rows found in input dataset.")

    summary_table = build_species_summary(pos_df)
    summary_csv = out_dir / "p1_summary_table.csv"
    summary_table.to_csv(summary_csv, index=False)

    heavy_atoms_png = out_dir / "p1_heavy_atoms_distribution.png"
    heavy_stats = save_heavy_atoms_plot(pos_df, heavy_atoms_png)

    year_csv = out_dir / "p1_publication_year_distribution.csv"
    if "publication_year" in pos_df.columns:
        year_dist = (
            pd.to_numeric(pos_df["publication_year"], errors="coerce")
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )
        year_out = year_dist.rename_axis("publication_year").reset_index(name="count")
    else:
        year_out = pd.DataFrame(columns=["publication_year", "count"])
    year_out.to_csv(year_csv, index=False)

    report = {
        "input_file": str(input_path),
        "total_rows": int(len(df)),
        "positive_rows": int(len(pos_df)),
        "output_summary_table": str(summary_csv),
        "output_heavy_atoms_plot": str(heavy_atoms_png),
        "output_year_distribution": str(year_csv),
    }
    report.update(heavy_stats)

    print(summary_table.to_string(index=False))
    print("\nLatest 10 publication years (Positive only):")
    if len(year_out) > 0:
        print(year_out.tail(10).to_string(index=False))
    else:
        print("No publication_year data.")

    print("\nSummary report:")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
