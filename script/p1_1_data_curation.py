# -*- coding: utf-8 -*-
"""
P1. Data Collection for SwissTargetPrediction (ChEMBL 36)
- Organism: human/mouse/rat
- Target type: SINGLE PROTEIN or PROTEIN COMPLEX
- Assay type: B (Binding)
- Activity types: Ki, Kd, IC50, EC50
- Heavy atoms <= 80
- Positive: activity_uM < 10
- Negative: activity_uM >= 500 (or non-interacting sampling), exclude weak binders 10~100 uM and ambiguous 10~500 uM
- 10 negatives per positive with pair_id
- Scaffold: RDKit Murcko (placeholder for OPREA-1)
- 10-fold CV without leakage (pair_id grouped)

Outputs:
- Parquet: features_store/chembl36_stp_training_set.parquet
- CSV sample (1000 rows): results/chembl36_stp_training_set_sample.csv
- Log: features_store/logs/p1_data_curation.log
- `assay_id` is included for Positive rows only (Negative rows are NULL).
"""

import json
import random
import logging
import multiprocessing
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
except ImportError:
    from rdkit.Chem import rdMolStandardize
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import subprocess

# -------------------------
# Config
# -------------------------
DB_URL = "postgresql://postgres:99pqpeqt@localhost:5432/chembl_36"
SEED = 42
MAX_HEAVY_ATOMS = 80
NEG_PER_POS = 10
NEG_MAX_ATTEMPTS = 2000
MAX_CORES = 24
N_SPLITS = 10

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "features_store"
RESULTS_DIR = ROOT / "results"
BLAST_DIR = ROOT / "blast_work"
LOG_DIR = FEATURES_DIR / "logs"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BLAST_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "p1_data_curation.log"

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("p1_data_curation")
logger.setLevel(logging.INFO)

_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setFormatter(_fmt)
sh = logging.StreamHandler()
sh.setFormatter(_fmt)
logger.addHandler(fh)
logger.addHandler(sh)

# -------------------------
# Utils
# -------------------------
random.seed(SEED)
np.random.seed(SEED)

_UNCHARGER = rdMolStandardize.Uncharger()


def _calc_heavy_atoms(smiles: str) -> int:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol.GetNumHeavyAtoms() if mol else 9999
    except Exception:
        return 9999


def _calc_scaffold(smiles: str) -> str:
    """RDKit Murcko scaffold (placeholder for OPREA-1)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        return None
    except Exception:
        return None


def _standardize_activity_to_uM(value: float, unit: str) -> float:
    if value is None or pd.isna(value):
        return np.nan
    if unit is None or pd.isna(unit):
        return np.nan
    u = str(unit).strip()
    if u.lower() in ("nm", "nanomolar"):
        return value / 1000.0
    if u.lower() in ("um", "µm", "micromolar"):
        return value
    if u.lower() in ("pm", "picomolar"):
        return value / 1_000_000.0
    if u.lower() in ("mm", "millimolar"):
        return value * 1000.0
    # Unknown unit: return NaN to be safe
    return np.nan


def _standardize_smiles_record(smiles: str) -> Tuple[str, bool, str]:
    """
    Return: (standardized_smiles, changed_flag, status)
    status: success | invalid_smiles | standardization_failed
    """
    if smiles is None or pd.isna(smiles):
        return None, False, "invalid_smiles"
    try:
        mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
        if mol is None:
            return None, False, "invalid_smiles"

        # 1) remove counter ions / solvents by fragment parent extraction
        mol = rdMolStandardize.FragmentParent(mol)

        # 2) neutralization
        mol = _UNCHARGER.uncharge(mol)

        # 3) sanitize and kekulize
        Chem.SanitizeMol(mol)
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            # Keep standardized non-kekulized form if kekulization is not possible.
            pass

        std_smi = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
        old_smi = Chem.MolToSmiles(Chem.MolFromSmiles(str(smiles)), isomericSmiles=True, kekuleSmiles=True)
        changed = std_smi != old_smi
        return std_smi, changed, "success"
    except Exception:
        return None, False, "standardization_failed"


# -------------------------
# 1. Data Fetch
# -------------------------
class ChemblDataFetcher:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)

    def fetch_data(self) -> pd.DataFrame:
        logger.info("Fetching data from ChEMBL 36...")
        query = """
        SELECT DISTINCT
            act.molregno,
            act.assay_id,
            md.chembl_id AS mol_chembl_id,
            cs.standard_inchi_key AS inchikey,
            cs.canonical_smiles,
            act.standard_value,
            act.standard_type,
            act.standard_units,
            ass.confidence_score,
            d.year AS publication_year,
            td.chembl_id AS target_chembl_id,
            td.pref_name AS target_name,
            td.target_type,
            td.organism,
            c.accession AS uniprot_id,
            c.sequence
        FROM activities act
        JOIN assays ass ON act.assay_id = ass.assay_id
        LEFT JOIN docs d ON ass.doc_id = d.doc_id
        JOIN target_dictionary td ON ass.tid = td.tid
        JOIN target_components tc ON td.tid = tc.tid
        JOIN component_sequences c ON tc.component_id = c.component_id
        JOIN molecule_dictionary md ON act.molregno = md.molregno
        JOIN compound_structures cs ON md.molregno = cs.molregno
        WHERE
            ass.assay_type = 'B'
            AND td.organism IN ('Homo sapiens', 'Mus musculus', 'Rattus norvegicus')
            AND td.target_type IN ('SINGLE PROTEIN', 'PROTEIN COMPLEX')
            AND act.standard_type IN ('Ki', 'Kd', 'IC50', 'EC50')
            AND act.standard_value IS NOT NULL
            AND cs.canonical_smiles IS NOT NULL;
        """
        df = pd.read_sql(query, self.engine)
        logger.info("Fetched %d raw records.", len(df))
        return df


# -------------------------
# 2. Preprocess
# -------------------------
class DataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        max_limit = 60
        self.n_cores = min(max_limit, multiprocessing.cpu_count(), MAX_CORES)
        self.smiles_stats = {}

    def standardize_smiles(self):
        logger.info(
            "Standardizing SMILES (counter ion/solvent removal, neutralization, kekulization) using %d cores...",
            self.n_cores,
        )
        self.df["canonical_smiles_original"] = self.df["canonical_smiles"]

        smiles_list = self.df["canonical_smiles_original"].tolist()
        try:
            with multiprocessing.Pool(self.n_cores) as pool:
                std_results = pool.map(_standardize_smiles_record, smiles_list)
        except Exception as e:
            logger.warning("Multiprocessing failed (%s). Falling back to serial.", e)
            std_results = [_standardize_smiles_record(s) for s in smiles_list]

        std_smiles, changed_flags, status = zip(*std_results)
        self.df["canonical_smiles"] = std_smiles
        self.df["smiles_was_modified"] = changed_flags
        self.df["smiles_standardize_status"] = status

        before = len(self.df)
        self.df = self.df[self.df["smiles_standardize_status"] == "success"].copy()
        self.df = self.df[self.df["canonical_smiles"].notna()].copy()
        after = len(self.df)

        total = len(status)
        changed_cnt = int(np.sum(np.array(changed_flags, dtype=np.int8)))
        unchanged_cnt = int(total - changed_cnt)
        invalid_cnt = int(sum(s == "invalid_smiles" for s in status))
        failed_cnt = int(sum(s == "standardization_failed" for s in status))

        self.smiles_stats = {
            "input_rows": int(total),
            "retained_rows": int(after),
            "dropped_rows": int(before - after),
            "changed_rows": changed_cnt,
            "unchanged_rows": unchanged_cnt,
            "invalid_smiles": invalid_cnt,
            "standardization_failed": failed_cnt,
        }

        logger.info("SMILES standardization summary: %s", json.dumps(self.smiles_stats, ensure_ascii=False))

        # Keep a compact diff sample for auditing
        diff_df = self.df[self.df["smiles_was_modified"]].copy()
        if not diff_df.empty:
            diff_sample_path = RESULTS_DIR / "p1_1_smiles_diff_sample.csv"
            cols = [
                "molregno",
                "mol_chembl_id",
                "target_chembl_id",
                "canonical_smiles_original",
                "canonical_smiles",
            ]
            diff_df[cols].head(1000).to_csv(diff_sample_path, index=False)
            logger.info("Saved SMILES diff sample: %s (rows=%d)", diff_sample_path, min(1000, len(diff_df)))

    def filter_heavy_atoms(self):
        logger.info("Filtering heavy atoms (<= %d) using %d cores...", MAX_HEAVY_ATOMS, self.n_cores)
        try:
            with multiprocessing.Pool(self.n_cores) as pool:
                self.df["heavy_atoms"] = pool.map(_calc_heavy_atoms, self.df["canonical_smiles"].tolist())
        except Exception as e:
            logger.warning("Multiprocessing failed (%s). Falling back to serial.", e)
            self.df["heavy_atoms"] = self.df["canonical_smiles"].map(_calc_heavy_atoms)
        before = len(self.df)
        self.df = self.df[self.df["heavy_atoms"] <= MAX_HEAVY_ATOMS].copy()
        logger.info("Heavy atoms filter: %d -> %d", before, len(self.df))

    def generate_scaffolds(self):
        logger.info("Generating scaffolds (RDKit Murcko, placeholder for OPREA-1)...")
        try:
            with multiprocessing.Pool(self.n_cores) as pool:
                self.df["scaffold_smiles"] = pool.map(_calc_scaffold, self.df["canonical_smiles"].tolist())
        except Exception as e:
            logger.warning("Multiprocessing failed (%s). Falling back to serial.", e)
            self.df["scaffold_smiles"] = self.df["canonical_smiles"].map(_calc_scaffold)
        self.df = self.df.dropna(subset=["scaffold_smiles"]).copy()

    def standardize_activity(self):
        logger.info("Standardizing activity values to uM...")
        self.df["activity_uM"] = [
            _standardize_activity_to_uM(v, u)
            for v, u in zip(self.df["standard_value"], self.df["standard_units"])
        ]

        # Labeling
        conditions = [
            (self.df["activity_uM"] < 10),
            (self.df["activity_uM"] >= 500)
        ]
        choices = ["Active", "Inactive"]
        self.df["label"] = np.select(conditions, choices, default="Ambiguous")

    def remove_ambiguity_and_conflicts(self):
        logger.info("Removing ambiguous data and conflicting labels...")
        before = len(self.df)
        self.df = self.df[self.df["label"] != "Ambiguous"].copy()
        grouped = self.df.groupby(["mol_chembl_id", "target_chembl_id"])["label"].nunique()
        conflicting_pairs = grouped[grouped > 1].index
        self.df.set_index(["mol_chembl_id", "target_chembl_id"], inplace=True)
        self.df.drop(index=conflicting_pairs, inplace=True)
        self.df.reset_index(inplace=True)
        # Keep a deterministic assay_id when multiple active rows exist for the same pair.
        self.df.sort_values(["mol_chembl_id", "target_chembl_id", "assay_id"], inplace=True)
        self.df.drop_duplicates(subset=["mol_chembl_id", "target_chembl_id"], inplace=True)
        logger.info("Ambiguity/conflicts removed: %d -> %d", before, len(self.df))

    def add_source_version(self):
        self.df["source_version"] = "ChEMBL_36_2025"


# -------------------------
# 3. BLAST (Paralog Map)
# -------------------------
class BlastEngine:
    def __init__(self, df: pd.DataFrame, work_dir: Path = BLAST_DIR):
        self.df = df
        self.work_dir = work_dir
        self.fasta_file = work_dir / "targets.fasta"
        self.db_name = work_dir / "target_db"
        self.paralog_map = {}
        self.n_threads = str(min(multiprocessing.cpu_count(), MAX_CORES))

    def prepare_blast_db(self):
        logger.info("Preparing BLAST database...")
        unique_targets = self.df[["target_chembl_id", "sequence"]].drop_duplicates().dropna()
        records = [SeqRecord(Seq(row["sequence"]), id=row["target_chembl_id"], description="")
                   for _, row in unique_targets.iterrows()]
        SeqIO.write(records, str(self.fasta_file), "fasta")
        cmd = ["makeblastdb", "-dbtype", "prot", "-in", str(self.fasta_file), "-out", str(self.db_name)]
        subprocess.run(cmd, check=True, capture_output=True)

    def run_all_vs_all_blast(self, e_value_thresh=1e-5):
        logger.info("Running all-vs-all BLAST (threads=%s)...", self.n_threads)
        output_file = self.work_dir / "blast_results.tsv"
        cmd = [
            "blastp",
            "-query", str(self.fasta_file),
            "-db", str(self.db_name),
            "-outfmt", "6",
            "-out", str(output_file),
            "-evalue", str(e_value_thresh),
            "-num_threads", self.n_threads
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        try:
            blast_df = pd.read_csv(output_file, sep="\t", header=None,
                                   names=["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
                                          "qstart", "qend", "sstart", "send", "evalue", "bitscore"])
            for _, row in blast_df.iterrows():
                if row["qseqid"] != row["sseqid"]:
                    self.paralog_map.setdefault(row["qseqid"], set()).add(row["sseqid"])
        except pd.errors.EmptyDataError:
            logger.warning("BLAST result empty.")


# -------------------------
# 4. Dataset Builder
# -------------------------
_shared_all_mols = None
_shared_known_interactions = None
_shared_paralog_map = None
_shared_mol_meta = None


def _init_worker(all_mols, known_interactions, paralog_map, mol_meta):
    global _shared_all_mols, _shared_known_interactions, _shared_paralog_map, _shared_mol_meta
    _shared_all_mols = all_mols
    _shared_known_interactions = known_interactions
    _shared_paralog_map = paralog_map
    _shared_mol_meta = mol_meta


def _process_target_centric(pos_row_dict):
    all_mols = _shared_all_mols
    known_interactions_set = _shared_known_interactions
    paralog_map = _shared_paralog_map
    mol_meta = _shared_mol_meta

    query_mol = pos_row_dict["mol_chembl_id"]
    query_target = pos_row_dict["target_chembl_id"]

    pos_entry = pos_row_dict.copy()
    pos_entry["set_type"] = "Positive"
    pos_entry["pair_id"] = f"{query_mol}_{query_target}"

    results = [pos_entry]

    neg_samples = []
    selected_neg_mols = set()
    attempts = 0
    target_paralogs = paralog_map.get(query_target, set())

    while len(neg_samples) < NEG_PER_POS and attempts < NEG_MAX_ATTEMPTS:
        attempts += 1
        rand_mol_id = np.random.choice(all_mols)

        if rand_mol_id == query_mol:
            continue
        if rand_mol_id in selected_neg_mols:
            continue
        if (rand_mol_id, query_target) in known_interactions_set:
            continue

        # Paralog check
        is_paralog_active = False
        for p_target in target_paralogs:
            if (rand_mol_id, p_target) in known_interactions_set:
                is_paralog_active = True
                break
        if is_paralog_active:
            continue

        selected_neg_mols.add(rand_mol_id)

        neg_entry = pos_row_dict.copy()
        neg_entry["mol_chembl_id"] = rand_mol_id

        if rand_mol_id in mol_meta:
            neg_entry["molregno"] = mol_meta[rand_mol_id]["molregno"]
            neg_entry["canonical_smiles"] = mol_meta[rand_mol_id]["canonical_smiles"]
            neg_entry["canonical_smiles_original"] = mol_meta[rand_mol_id]["canonical_smiles_original"]
            neg_entry["heavy_atoms"] = mol_meta[rand_mol_id]["heavy_atoms"]
            neg_entry["scaffold_smiles"] = mol_meta[rand_mol_id]["scaffold_smiles"]
            neg_entry["inchikey"] = mol_meta[rand_mol_id]["inchikey"]

        neg_entry["activity_uM"] = np.nan
        neg_entry["standard_value"] = np.nan
        neg_entry["confidence_score"] = np.nan
        neg_entry["assay_id"] = pd.NA
        neg_entry["label"] = "Assumed_Inactive"
        neg_entry["set_type"] = "Negative"
        neg_entry["pair_id"] = f"{query_mol}_{query_target}"

        neg_samples.append(neg_entry)

    if len(neg_samples) < NEG_PER_POS:
        return []  # drop this positive if insufficient negatives

    results.extend(neg_samples)
    return results


class DatasetBuilder:
    def __init__(self, df: pd.DataFrame, blast_engine: BlastEngine):
        self.df = df
        self.blast_engine = blast_engine
        self.final_dataset = []
        max_limit = 60
        self.n_cores = min(max_limit, multiprocessing.cpu_count(), MAX_CORES)

    def build_balanced_dataset(self):
        logger.info("Building dataset (Target-Centric) with %d cores...", self.n_cores)
        positives = self.df[self.df["label"] == "Active"]

        active_pairs = set(zip(self.df["mol_chembl_id"], self.df["target_chembl_id"]))
        all_mols = self.df["mol_chembl_id"].unique()
        mol_meta = self.df[
            [
                "mol_chembl_id",
                "molregno",
                "canonical_smiles",
                "canonical_smiles_original",
                "heavy_atoms",
                "scaffold_smiles",
                "inchikey",
            ]
        ].drop_duplicates("mol_chembl_id").set_index("mol_chembl_id").to_dict("index")

        paralog_map = self.blast_engine.paralog_map
        tasks = positives.to_dict("records")

        processed_data = []
        with multiprocessing.Pool(processes=self.n_cores,
                                  initializer=_init_worker,
                                  initargs=(all_mols, active_pairs, paralog_map, mol_meta)) as pool:
            for result in tqdm(pool.imap_unordered(_process_target_centric, tasks, chunksize=200), total=len(tasks)):
                if result:
                    processed_data.extend(result)

        self.final_dataset = pd.DataFrame(processed_data)
        logger.info("Final dataset size: %d", len(self.final_dataset))

    def add_cross_validation_folds(self, n_splits=10):
        logger.info("Assigning CV folds (%d-fold)...", n_splits)
        pos_pairs = self.final_dataset[self.final_dataset["set_type"] == "Positive"][["pair_id", "target_chembl_id"]].drop_duplicates()
        pair_ids = pos_pairs["pair_id"].values
        y = pos_pairs["target_chembl_id"].values

        fold_map: Dict[str, int] = {}
        try:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            for fold_idx, (_, test_idx) in enumerate(skf.split(pair_ids, y)):
                for pid in pair_ids[test_idx]:
                    fold_map[pid] = fold_idx + 1
        except Exception as e:
            logger.warning("Stratified split failed (%s). Falling back to random split.", e)
            rng = np.random.RandomState(SEED)
            rng.shuffle(pair_ids)
            folds = np.array_split(pair_ids, n_splits)
            for fold_idx, pairs in enumerate(folds):
                for pid in pairs:
                    fold_map[pid] = fold_idx + 1

        self.final_dataset["cv_fold"] = self.final_dataset["pair_id"].map(fold_map)


# -------------------------
# Main
# -------------------------

def main():
    try:
        logger.info("=== P1 Data Curation Started ===")
        fetcher = ChemblDataFetcher(DB_URL)
        df_raw = fetcher.fetch_data()
        if df_raw.empty:
            logger.error("No data fetched. Exiting.")
            return

        pre = DataPreprocessor(df_raw)
        pre.standardize_smiles()
        pre.filter_heavy_atoms()
        pre.standardize_activity()
        pre.remove_ambiguity_and_conflicts()
        pre.generate_scaffolds()
        pre.add_source_version()
        df_clean = pre.df

        blast_engine = BlastEngine(df_clean)
        blast_engine.prepare_blast_db()
        blast_engine.run_all_vs_all_blast(e_value_thresh=1e-5)

        builder = DatasetBuilder(df_clean, blast_engine)
        builder.build_balanced_dataset()
        builder.add_cross_validation_folds(n_splits=N_SPLITS)

        final_df = builder.final_dataset
        if "assay_id" in final_df.columns:
            final_df["assay_id"] = final_df["assay_id"].astype("Int64")
            final_df.loc[final_df["set_type"] == "Negative", "assay_id"] = pd.NA
            pos_mask = final_df["set_type"] == "Positive"
            logger.info(
                "assay_id summary | pos_non_null=%d | pos_null=%d | neg_non_null=%d",
                int(final_df.loc[pos_mask, "assay_id"].notna().sum()),
                int(final_df.loc[pos_mask, "assay_id"].isna().sum()),
                int(final_df.loc[~pos_mask, "assay_id"].notna().sum()),
            )

        # Save outputs
        parquet_path = FEATURES_DIR / "chembl36_stp_training_set.parquet"
        csv_sample_path = RESULTS_DIR / "chembl36_stp_training_set_sample.csv"

        logger.info("Saving Parquet: %s", parquet_path)
        final_df.to_parquet(parquet_path, index=False, compression="snappy")

        logger.info("Saving CSV sample (1000 rows): %s", csv_sample_path)
        sample_df = final_df.sample(n=min(1000, len(final_df)), random_state=SEED)
        sample_df.to_csv(csv_sample_path, index=False)

        # Stats summary
        stats = {
            "total_rows": int(len(final_df)),
            "pos_rows": int((final_df["set_type"] == "Positive").sum()),
            "neg_rows": int((final_df["set_type"] == "Negative").sum()),
            "unique_pairs": int(final_df["pair_id"].nunique()),
            "unique_targets": int(final_df["target_chembl_id"].nunique()),
            "unique_molecules": int(final_df["mol_chembl_id"].nunique()),
            "smiles_standardization": pre.smiles_stats,
        }
        logger.info("Stats: %s", json.dumps(stats, ensure_ascii=False))

        changed_rows = int(final_df["smiles_was_modified"].sum()) if "smiles_was_modified" in final_df.columns else 0
        logger.info(
            "SMILES compare summary | preserved_col=canonical_smiles_original | used_col=canonical_smiles | changed_rows=%d",
            changed_rows,
        )

        logger.info("=== P1 Data Curation Completed ===")
    except Exception as e:
        logger.exception("Fatal error: %s", e)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
