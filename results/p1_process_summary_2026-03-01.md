# P1 Process Summary (saved: 2026-03-01)

## Scope
- Phase: P1 (data curation, validation, summary)
- Project path: `/mnt/d/stp_proj_v3`

## 1) Rule updates related to P1/P5
- Updated `AGENTS.md` with script series naming rules (`p1_1`, `p1_2`, ...).
- Added P1 SMILES handling rules:
  - preserve original SMILES in `canonical_smiles_original`
  - use standardized SMILES in `canonical_smiles`
  - log original vs standardized comparison
- Added P5 query normalization rule:
  - all `p5_*.py` must normalize query SMILES (standardization, counter ion/solvent removal, neutralization, kekulization)

## 2) `p1_1_data_curation.py` improvements
File: `script/p1_1_data_curation.py`

### SMILES pipeline integration
- Added SMILES normalization stage:
  - fragment parent extraction (remove counter ions/solvents)
  - neutralization (uncharging)
  - kekulization attempt
- Preserved source SMILES in `canonical_smiles_original`.
- Switched downstream processing to standardized `canonical_smiles`.
- Added columns:
  - `smiles_was_modified`
  - `smiles_standardize_status`
- Added diff sample output:
  - `results/p1_1_smiles_diff_sample.csv`

### Data consistency fixes
- Fixed negative sampling metadata consistency:
  - negative row `molregno` now updated consistently with `mol_chembl_id`
- Carried `canonical_smiles_original` into sampled negatives.

### assay_id integration (requested)
- Query now fetches `act.assay_id`.
- Deterministic dedup rule keeps stable assay id for pair duplicates.
- Final output rule:
  - Positive rows: `assay_id` populated
  - Negative rows: `assay_id` is NULL
- Added assay_id summary logging.

## 3) Runtime environment setup for P1 execution
- Created `micromamba` env: `stp-rdkit`
- Installed required packages:
  - `rdkit`, `biopython`, `scikit-learn`, `tqdm`, `sqlalchemy`, `psycopg2`, `pyarrow`

## 4) P1_1 execution status/result
- Pipeline executed successfully through final dataset generation.
- Core result file updated:
  - `features_store/chembl36_stp_training_set.parquet`
- Logged dataset size:
  - `15,025,219` rows
- SMILES standardization summary observed in logs:
  - input: `2,462,207`
  - retained: `2,462,196`
  - changed: `323,720`
  - standardization failed: `11`

## 5) assay_id backfill/verification completed
- Added `assay_id` to current parquet for Positive rows only.
- Validation counts:
  - positive non-null assay_id: `1,365,929`
  - positive null assay_id: `0`
  - negative non-null assay_id: `0`
- Backup created before overwrite:
  - `features_store/chembl36_stp_training_set.before_assay_id.parquet`

## 6) Sample CSV updates
- Regenerated sample CSV to include `assay_id`:
  - `results/chembl36_stp_training_set_sample.csv`
- Created inspection CSV for assay behavior:
  - `results/chembl36_stp_training_set_assayid_check_sample.csv`

## 7) P1_2 validation execution
- Executed `script/p1_2_validate.py`
- Output:
  - `results/p1_validation_report.json`
- Key validation result:
  - all major checks passed (`missing_columns`, leakage, pair size, thresholds, etc. all zero-issue)

## 8) P1_3 summary execution
- Executed `script/p1_3_summarize_stp_data.py`
- Outputs:
  - `results/p1_summary_table.csv`
  - `results/p1_heavy_atoms_distribution.png`
  - `results/p1_publication_year_distribution.csv`
- Summary highlights:
  - Positive rows: `1,365,929`
  - Unique active molecules: `893,431`
  - Heavy atoms median/max: `32.0 / 80.0`

## 9) Current P1 state
- P1 outputs and validation/summary artifacts are generated.
- `p1_1_data_curation.py` now includes SMILES normalization and assay_id handling by default, so additional post-processing for assay_id is no longer required.
