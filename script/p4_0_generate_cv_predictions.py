# -*- coding: utf-8 -*-
"""
Phase 4.0: Generate grouped-CV out-of-fold prediction rows.

- Loads compound / target / activity tables from data/
- Builds grouped CV splits with GroupKFold on compound-level group_id
- Excludes validation compounds entirely from the fold reference set
- Uses a placeholder prediction adapter so the scoring backend can be replaced later
- Saves filtered prediction rows to data/cv_predictions.parquet
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = DEFAULT_PROJECT_ROOT / "data"


def infer_size_bin(heavy_atom_count):
    """Map heavy atom count to a compact size bin label."""
    if pd.isna(heavy_atom_count):
        return "unknown"
    ha = int(heavy_atom_count)
    if ha <= 10:
        return "le_10"
    if ha <= 20:
        return "11_20"
    if ha <= 30:
        return "21_30"
    if ha <= 40:
        return "31_40"
    if ha <= 50:
        return "41_50"
    return "gt_50"


def load_inputs(compound_path, target_path, activity_path):
    """Load and validate the three required parquet inputs."""
    compounds = pd.read_parquet(compound_path).copy()
    targets = pd.read_parquet(target_path).copy()
    activities = pd.read_parquet(activity_path).copy()

    compound_required = [
        "compound_id",
        "canonical_smiles",
        "heavy_atom_count",
        "group_id",
    ]
    activity_required = ["compound_id", "target_id"]
    target_required = ["target_id"]

    for col in compound_required:
        if col not in compounds.columns:
            raise KeyError(f"compound_table missing required column: {col}")
    for col in activity_required:
        if col not in activities.columns:
            raise KeyError(f"activity_table missing required column: {col}")
    for col in target_required:
        if col not in targets.columns:
            raise KeyError(f"target_table missing required column: {col}")

    compounds = compounds[compound_required].drop_duplicates(subset=["compound_id"]).copy()
    compounds["canonical_smiles"] = compounds["canonical_smiles"].fillna("").astype(str)
    compounds["heavy_atom_count"] = pd.to_numeric(
        compounds["heavy_atom_count"], errors="coerce"
    ).fillna(-1).astype(np.int64)
    compounds["group_id"] = compounds["group_id"].fillna("__missing_group__").astype(str)
    compounds["size_bin"] = compounds["heavy_atom_count"].map(infer_size_bin)
    compounds["query_id"] = compounds["compound_id"].astype(str)

    activities = activities[activity_required].drop_duplicates().copy()
    activities["compound_id"] = activities["compound_id"].astype(compounds["compound_id"].dtype)
    activities["target_id"] = activities["target_id"].astype(str)

    targets = targets.drop_duplicates(subset=["target_id"]).copy()
    targets["target_id"] = targets["target_id"].astype(str)

    missing_activity_compounds = sorted(
        set(activities["compound_id"].unique()) - set(compounds["compound_id"].unique())
    )
    if missing_activity_compounds:
        raise ValueError(
            f"activity_table references unknown compound_id values: "
            f"{missing_activity_compounds[:10]}"
        )

    return compounds, targets, activities


def build_truth_map(activity_df):
    """Build query truth labels from activity rows."""
    return activity_df.groupby("compound_id")["target_id"].agg(lambda x: set(x)).to_dict()


class PlaceholderPredictionAdapter:
    """
    Placeholder scoring backend.

    Replace `score_query_to_target` with the production scoring engine later.
    The current logic uses string token overlap as a 2D proxy and heavy-atom
    distance as a 3D proxy so the script remains executable with the current inputs.
    """

    def __init__(self, reference_df):
        self.reference_df = reference_df.copy()
        if self.reference_df.empty:
            self.target_to_refs = {}
            return

        self.reference_df["smiles_token_set"] = self.reference_df["canonical_smiles"].map(
            self._smiles_token_set
        )
        self.target_to_refs = {
            target_id: group[["canonical_smiles", "heavy_atom_count", "smiles_token_set"]]
            .reset_index(drop=True)
            for target_id, group in self.reference_df.groupby("target_id", sort=False)
        }

    @staticmethod
    def _smiles_token_set(smiles):
        text = str(smiles or "")
        if len(text) < 2:
            return {text} if text else set()
        return {text[idx : idx + 2] for idx in range(len(text) - 1)}

    @staticmethod
    def _max_jaccard_similarity(query_tokens, reference_tokens):
        if not reference_tokens:
            return 0.0
        if not query_tokens and not reference_tokens:
            return 1.0
        union_size = len(query_tokens | reference_tokens)
        if union_size == 0:
            return 0.0
        return len(query_tokens & reference_tokens) / union_size

    @staticmethod
    def _size_similarity(query_heavy_atoms, reference_heavy_atoms):
        delta = abs(int(query_heavy_atoms) - int(reference_heavy_atoms))
        return 1.0 / (1.0 + (delta / 10.0))

    def score_query_to_target(self, query_row, target_id):
        refs = self.target_to_refs.get(target_id)
        if refs is None or refs.empty:
            return 0.0, 0.0, 0.0

        query_tokens = self._smiles_token_set(query_row["canonical_smiles"])
        query_ha = int(query_row["heavy_atom_count"])

        max_sim_2d = 0.0
        max_sim_3d = 0.0
        for ref in refs.itertuples(index=False):
            sim_2d = self._max_jaccard_similarity(query_tokens, ref.smiles_token_set)
            sim_3d = self._size_similarity(query_ha, ref.heavy_atom_count)
            if sim_2d > max_sim_2d:
                max_sim_2d = sim_2d
            if sim_3d > max_sim_3d:
                max_sim_3d = sim_3d

        combined_score = 0.7 * max_sim_2d + 0.3 * max_sim_3d
        return float(combined_score), float(max_sim_2d), float(max_sim_3d)


def run_grouped_cv(
    compounds_df,
    targets_df,
    activities_df,
    truth_map,
    n_splits,
    keep_top_rank,
    keep_min_score,
):
    """Generate grouped-CV out-of-fold predictions."""
    groups = compounds_df["group_id"].astype(str).values
    n_unique_groups = compounds_df["group_id"].nunique()
    effective_splits = min(int(n_splits), int(n_unique_groups))
    if effective_splits < 2:
        raise ValueError("GroupKFold requires at least 2 unique group_id values.")

    gkf = GroupKFold(n_splits=effective_splits)
    target_ids = targets_df["target_id"].astype(str).tolist()
    target_meta = targets_df.copy()
    cv_scheme = f"groupkfold_group_id_{effective_splits}fold"

    all_rows = []
    fold_sizes = []

    dummy_x = np.zeros((len(compounds_df), 1), dtype=np.int8)
    dummy_y = np.zeros(len(compounds_df), dtype=np.int8)

    for fold_id, (train_idx, valid_idx) in enumerate(
        gkf.split(dummy_x, dummy_y, groups=groups), start=1
    ):
        train_compounds = compounds_df.iloc[train_idx].copy()
        valid_compounds = compounds_df.iloc[valid_idx].copy()
        train_compound_ids = set(train_compounds["compound_id"].tolist())

        reference_activity = activities_df[
            activities_df["compound_id"].isin(train_compound_ids)
        ].copy()
        reference_df = reference_activity.merge(
            train_compounds[
                ["compound_id", "canonical_smiles", "heavy_atom_count", "group_id"]
            ],
            on="compound_id",
            how="inner",
        )
        adapter = PlaceholderPredictionAdapter(reference_df=reference_df)

        for query_row in valid_compounds.itertuples(index=False):
            query_truth = truth_map.get(query_row.compound_id, set())
            query_records = []
            for target_id in target_ids:
                combined_score, max_sim_2d, max_sim_3d = adapter.score_query_to_target(
                    query_row._asdict(), target_id
                )
                query_records.append(
                    {
                        "query_id": str(query_row.query_id),
                        "compound_id": query_row.compound_id,
                        "query_smiles": query_row.canonical_smiles,
                        "query_heavy_atom_count": int(query_row.heavy_atom_count),
                        "size_bin": query_row.size_bin,
                        "cv_scheme": cv_scheme,
                        "fold_id": int(fold_id),
                        "group_id": query_row.group_id,
                        "target_id": str(target_id),
                        "combined_score": combined_score,
                        "max_sim_2d": max_sim_2d,
                        "max_sim_3d": max_sim_3d,
                        "is_true_target": bool(str(target_id) in query_truth),
                    }
                )

            query_df = pd.DataFrame(query_records)
            query_df = query_df.sort_values(
                ["combined_score", "max_sim_2d", "max_sim_3d", "target_id"],
                ascending=[False, False, False, True],
            ).reset_index(drop=True)
            query_df["rank"] = np.arange(1, len(query_df) + 1, dtype=np.int64)
            query_df = query_df[
                (query_df["rank"] <= int(keep_top_rank))
                | (query_df["combined_score"] >= float(keep_min_score))
            ].copy()
            query_df = query_df.merge(target_meta, on="target_id", how="left")
            all_rows.append(query_df)

        fold_sizes.append(
            {
                "fold_id": int(fold_id),
                "n_train_compounds": int(len(train_compounds)),
                "n_valid_compounds": int(len(valid_compounds)),
                "n_reference_rows": int(len(reference_df)),
            }
        )
        print(
            f"[fold {fold_id}] train_compounds={len(train_compounds):,} "
            f"valid_compounds={len(valid_compounds):,} "
            f"reference_rows={len(reference_df):,}"
        )

    if not all_rows:
        return pd.DataFrame(), fold_sizes

    out_df = pd.concat(all_rows, ignore_index=True)
    ordered_cols = [
        "query_id",
        "compound_id",
        "query_smiles",
        "query_heavy_atom_count",
        "size_bin",
        "cv_scheme",
        "fold_id",
        "group_id",
        "target_id",
        "rank",
        "combined_score",
        "max_sim_2d",
        "max_sim_3d",
        "is_true_target",
    ]
    remaining_cols = [col for col in out_df.columns if col not in ordered_cols]
    out_df = out_df[ordered_cols + remaining_cols]
    return out_df, fold_sizes


def save_cv_predictions(predictions_df, output_path):
    """Persist prediction rows as parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(output_path, index=False)
    print(f"Saved {len(predictions_df):,} rows to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate grouped-CV out-of-fold target prediction rows."
    )
    parser.add_argument(
        "--compound-table",
        default=str(DEFAULT_DATA_DIR / "compound_table.parquet"),
        help="Path to compound_table.parquet",
    )
    parser.add_argument(
        "--target-table",
        default=str(DEFAULT_DATA_DIR / "target_table.parquet"),
        help="Path to target_table.parquet",
    )
    parser.add_argument(
        "--activity-table",
        default=str(DEFAULT_DATA_DIR / "activity_table.parquet"),
        help="Path to activity_table.parquet",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_DATA_DIR / "cv_predictions.parquet"),
        help="Output parquet path",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Requested GroupKFold split count; capped by unique group count",
    )
    parser.add_argument(
        "--keep-top-rank",
        type=int,
        default=100,
        help="Keep rows with rank <= this threshold",
    )
    parser.add_argument(
        "--keep-min-score",
        type=float,
        default=0.02,
        help="Keep rows with combined_score >= this threshold",
    )
    args = parser.parse_args()

    print(">> Loading inputs")
    compounds_df, targets_df, activities_df = load_inputs(
        compound_path=args.compound_table,
        target_path=args.target_table,
        activity_path=args.activity_table,
    )
    truth_map = build_truth_map(activities_df)

    print(">> Running grouped CV OOF scoring")
    predictions_df, fold_sizes = run_grouped_cv(
        compounds_df=compounds_df,
        targets_df=targets_df,
        activities_df=activities_df,
        truth_map=truth_map,
        n_splits=args.n_splits,
        keep_top_rank=args.keep_top_rank,
        keep_min_score=args.keep_min_score,
    )

    print(">> Saving parquet output")
    save_cv_predictions(predictions_df=predictions_df, output_path=args.output)

    print(">> Completed")
    print(
        f"   compounds={len(compounds_df):,} targets={len(targets_df):,} "
        f"activity_rows={len(activities_df):,} prediction_rows={len(predictions_df):,}"
    )
    for fold_stat in fold_sizes:
        print(
            f"   fold={fold_stat['fold_id']} "
            f"train={fold_stat['n_train_compounds']:,} "
            f"valid={fold_stat['n_valid_compounds']:,} "
            f"reference_rows={fold_stat['n_reference_rows']:,}"
        )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
