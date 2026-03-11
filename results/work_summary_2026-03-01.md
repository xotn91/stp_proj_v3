# STP Project Work Summary (2026-03-01)

## 1) Rule/Policy Updates
- Updated project rules in `AGENTS.md` to enforce script series naming:
  - `p{phase}_{series}_{task}.py` (e.g., `p1_1_*`, `p1_2_*`)
- Added P1 SMILES handling rules in `AGENTS.md`:
  - Preserve original SMILES in `canonical_smiles_original`
  - Use standardized SMILES in `canonical_smiles` for downstream phases
  - Log comparison stats between original vs standardized SMILES
- Added P5 rule in `AGENTS.md`:
  - All `p5_*.py` scripts must apply SMILES normalization to new query molecules
  - Required steps: standardization, counter ion/solvent removal, neutralization, kekulization

## 2) p1_1_data_curation.py Main Enhancements
File: `script/p1_1_data_curation.py`

### SMILES pipeline integrated
- Added SMILES normalization step before heavy atom/scaffold processing:
  - Fragment parent extraction (counter ion/solvent removal)
  - Uncharging (neutralization)
  - Kekulization attempt
- Preserved source SMILES as `canonical_smiles_original`
- Replaced working SMILES with standardized value in `canonical_smiles`
- Added status columns:
  - `smiles_was_modified`
  - `smiles_standardize_status`
- Added SMILES diff sample output:
  - `results/p1_1_smiles_diff_sample.csv`

### Data consistency fixes
- Negative sampling now updates `molregno` consistently with `mol_chembl_id`
- Included `canonical_smiles_original` in negative-side metadata propagation

### assay_id integration (requested later)
- Fetch query now includes `act.assay_id`
- For duplicated positive pair rows, deterministic row selection keeps stable `assay_id`
  - sorted by `mol_chembl_id`, `target_chembl_id`, `assay_id` before `drop_duplicates`
- Negative rows explicitly set `assay_id = NULL`
- Final dataframe enforces:
  - Positive rows: `assay_id` retained
  - Negative rows: `assay_id` null
- Added log summary for assay_id completeness:
  - `pos_non_null`, `pos_null`, `neg_non_null`

## 3) Environment Setup
- Created runtime env: `micromamba` env `stp-rdkit`
- Installed packages:
  - `rdkit`, `biopython`, `scikit-learn`, `tqdm`, `sqlalchemy`, `psycopg2`, `pyarrow`

## 4) Execution/Validation Results
### p1_1 execution status
- Full pipeline executed through dataset construction and parquet save.
- Final dataset size in logs: `15,025,219`
- SMILES standardization summary observed:
  - `input_rows`: `2,462,207`
  - `retained_rows`: `2,462,196`
  - `changed_rows`: `323,720`
  - `standardization_failed`: `11`

### Output status
- Parquet updated:
  - `features_store/chembl36_stp_training_set.parquet`
- CSV sample initially hit a temporary permission issue, then re-saved successfully.
- Current sample CSV contains `assay_id`:
  - `results/chembl36_stp_training_set_sample.csv`

## 5) assay_id backfill operation (already completed)
- Added `assay_id` to current parquet for positive rows only and validated:
  - Positive non-null `assay_id`: `1,365,929`
  - Positive null `assay_id`: `0`
  - Negative non-null `assay_id`: `0`
- Backup created before overwrite:
  - `features_store/chembl36_stp_training_set.before_assay_id.parquet`

## 6) Additional verification artifact
- Small check CSV generated to inspect assay_id behavior:
  - `results/chembl36_stp_training_set_assayid_check_sample.csv`
  - includes Positive + Negative samples with `assay_id`

## 7) Current state
- `p1_1_data_curation.py` now natively produces outputs with SMILES normalization policy and `assay_id` handling aligned to requirements.
- Additional manual post-processing for assay_id should no longer be necessary.
