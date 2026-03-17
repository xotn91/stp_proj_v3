# P4 LOO Prediction Log Work Summary

## Scope

This work created and iteratively revised `script/p4_0_generate_cv_predictions.py` to generate leave-one-out prediction logs for SwissTargetPrediction-style analysis from:

- `data/compound_table.parquet`
- `data/target_table.parquet`
- `data/activity_table.parquet`

The final workflow uses:

- human targets only
- size-bin sampling
- leave-one-out exclusion of the query molecule itself
- dual-GPU execution
- project FP2 / ES5D / coefficient assets for scoring

## Final Script

- Script: `/mnt/d/stp_proj_v3/script/p4_0_generate_cv_predictions.py`
- Current engine mode: actual dual-GPU score engine
- GPU usage: `cuda:0` and `cuda:1`

## Major Iterations

### 1. Initial grouped-CV version

The first version was written as a grouped-CV out-of-fold script. This did not match the later leave-one-out requirements and was replaced.

### 2. Leave-one-out placeholder version

The script was rewritten to:

- use `cv_scheme='loo'`
- apply the required size bins:
  - `01_15`
  - `16_20`
  - `21_25`
  - `26_30`
  - `31_35`
  - `36_40`
  - `41_plus`
- sample up to 1000 query molecules per size bin
- generate `rank`
- generate `is_true_target`
- save only rows satisfying:
  - `rank <= 100`
  - or `combined_score >= 0.02`

### 3. Dual-GPU placeholder version

The placeholder adapter was moved to a dual-GPU implementation using `torch`. This enabled:

- GPU memory cloning to both devices
- query batch splitting across both GPUs
- faster leave-one-out scoring

### 4. Placeholder calibration

The placeholder score calibration was revised several times because the initial versions were too dense and caused nearly all targets to be retained.

The final placeholder calibration before engine replacement produced a sparse-enough output:

- total rows: `960,116`
- total queries: `7,000`
- median rows per query: `100`

### 5. Actual engine replacement

The adapter was then replaced with the project’s actual engine family using:

- `/mnt/d/stp_proj_v3/features_store/final_training_meta.parquet`
- `/mnt/d/stp_proj_v3/features_store/fp2_aligned.memmap`
- `/mnt/d/stp_proj_v3/features_store/es5d_db_k20.memmap`
- `/mnt/d/stp_proj_v3/features_store/stp_coef_FusionK5_paired_sum.json`

The current script resolves:

- `compound_id -> memmap_idx`

and then uses the dual-GPU engine to compute:

- `max_sim_2d`
- `max_sim_3d`
- `combined_score`

for each leave-one-out query.

## Input Preparation

The `data/*.parquet` files were regenerated to match the required schema:

### `data/compound_table.parquet`

- `compound_id`
- `canonical_smiles`
- `heavy_atom_count`
- `species`

### `data/target_table.parquet`

- `target_id`
- `target_name`
- `species`

### `data/activity_table.parquet`

- `compound_id`
- `target_id`

## Actual Engine Pilot Result

A pilot run with:

- `max-queries-per-size-bin = 1`
- `keep-top-rank = 20`

produced:

- output file: `/mnt/d/stp_proj_v3/data/cv_predictions_actual_engine_pilot.parquet`
- queries: `7`
- rows: `31,668`

This means all `4,524` human targets were retained for every pilot query.

### Interpretation

The actual engine replacement worked technically, but the current save policy:

- `rank <= 100 or combined_score >= 0.02`

is too loose when applied directly to the actual engine scores. In the pilot run, the score threshold retained essentially all targets.

## Actual Engine Technical Status

### Confirmed working

- dual-GPU device discovery
- host memmap loading
- FP2 / ES5D cloning to both GPUs
- coefficient loading
- leave-one-out self-exclusion inside target active sets
- score generation and parquet output

### Current limitation

The actual engine output is too dense under the current threshold rule. Therefore:

- the script is technically functional
- but the current `combined_score >= 0.02` policy is not yet well calibrated for the actual engine output scale

## Latest Accepted Sparse Output

Before the actual engine swap, the calibrated dual-GPU placeholder run produced:

- output file: `/mnt/d/stp_proj_v3/data/cv_predictions.parquet`
- rows: `960,116`
- queries: `7,000`
- median rows per query: `100`
- 99th percentile rows per query: about `1,251`

Additional review showed:

- true-target rows: `10,247`
- queries with at least one retained true target: `6,829`
- query-level top-1 hit rate: about `0.772`
- query-level top-5 hit rate: about `0.939`
- query-level top-10 hit rate: about `0.965`
- query-level top-15 hit rate: about `0.975`
- query-level top-100 hit rate: about `0.998`

## Review Artifacts

Review plots and summary tables were generated under:

- `/mnt/d/stp_proj_v3/results/cv_predictions_review`

Files created:

- `01_rows_per_query_hist.png`
- `02_rows_per_query_by_size_bin.png`
- `03_combined_score_hist.png`
- `04_query_level_hit_rate_curve.png`
- `05_probability_vs_2d_similarity.png`
- `06_probability_vs_3d_similarity.png`
- `07_probability_vs_similarity_panel.png`
- `summary_metrics.csv`

## Current Recommendation

The actual engine should be kept, but one of the following must be changed before a full final production run:

1. Calibrate the actual-engine probability scale and use a higher score threshold.
2. Retain only `rank <= 100` for the actual-engine run.
3. Learn a separate post-score calibration specifically for the leave-one-out logging output.

## Practical Conclusion

As of this summary:

- the script architecture is complete
- dual-GPU execution is active
- the actual engine has been integrated successfully
- the remaining issue is calibration of the save threshold, not engine integration
