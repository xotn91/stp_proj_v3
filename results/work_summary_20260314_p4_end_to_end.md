# P4 End-to-End Work Summary

## Scope

This document summarizes the end-to-end `p4` work performed for leave-one-out SwissTargetPrediction-style prediction logging, quality analysis, and post-hoc calibration in `/mnt/d/stp_proj_v3`.

Primary scripts:

- `/mnt/d/stp_proj_v3/script/p4_0_generate_cv_predictions.py`
- `/mnt/d/stp_proj_v3/script/p4_1_analyze_cv_predictions.py`
- `/mnt/d/stp_proj_v3/script/p4_2_apply_rank_calibration.py`
- `/mnt/d/stp_proj_v3/script/p4_3_score_calibration.py`

## Final P4 Pipeline

### 1. Prediction log generation

`p4_0_generate_cv_predictions.py` generates leave-one-out prediction logs from:

- `data/compound_table.parquet`
- `data/target_table.parquet`
- `data/activity_table.parquet`

The implemented production workflow:

- uses human targets only
- samples query molecules by heavy-atom size bin
- excludes the query molecule itself from the reference set
- uses project FP2 and ES5D assets with the actual scoring engine
- supports dual-GPU execution
- writes prediction logs and sidecar summaries

### 2. Quality analysis

`p4_1_analyze_cv_predictions.py` computes:

- overall Top-K hit curves
- size-bin-specific Top-K curves
- logistic fits over size-bin rank precision curves

### 3. Rank-based calibration

`p4_2_apply_rank_calibration.py` adds a rank-derived interpretation column:

- `calibrated_rank_hit_rate`

This is a rank-based hit-rate proxy and is not a row-level posterior probability.

### 4. Score-based calibration

`p4_3_score_calibration.py` fits score-to-empirical-precision mappings by size bin and adds:

- `calibrated_score_precision`

This is the most practical post-hoc precision proxy produced so far.

## Input Schema Used

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

## Engine and Coefficient Rules

### Actual engine assets

- `/mnt/d/stp_proj_v3/features_store/final_training_meta.parquet`
- `/mnt/d/stp_proj_v3/features_store/fp2_aligned.memmap`
- `/mnt/d/stp_proj_v3/features_store/es5d_db_k20.memmap`

### Current coefficient file

- `/mnt/d/stp_proj_v3/features_store/p3_run_full_aug_scaf_off_20260311/p3_1_1_K1_paired_trainer_fast__K1__paired_sum__all__C10p0__Y2023__Hd6d53c2431/stp_coef_K1_paired_sum_all.json`

### Coefficient application rule

The prediction engine does not choose coefficients by the coarse `size_bin` labels.

Instead:

- each query uses its integer `heavy_atom_count`
- that value is clamped internally to a supported range
- the corresponding coefficient row is selected for scoring

For the current K1 paired model:

- feature 1: top-1 normalized 3D similarity
- feature 2: top-1 normalized 2D similarity

The score is computed as:

- `logit = intercept(HA) + a3d(HA) * top1_3d + a2d(HA) * top1_2d`
- `combined_score = sigmoid(logit)`

### Threshold rule used in the latest run

The latest full run used the `stp2014` threshold preset:

- `thr2d = 0.30`
- `thr3d = 0.65`

Normalization was enabled, and below-threshold candidates were excluded with:

- `--exclude-below-threshold`

## Sampling and Monitoring Changes

`p4_0_generate_cv_predictions.py` was strengthened to support practical production runs:

- default query sampling reduced from `1000` to `300` per size bin
- progress logging with elapsed time, rate, and ETA
- line-buffered output for external-terminal live monitoring

Recommended full-run command:

```bash
cd /mnt/d/stp_proj_v3
/home/xotn91/.local/share/mamba/envs/stp/bin/python3.10 -u script/p4_0_generate_cv_predictions.py \
  --thr-preset stp2014 \
  --max-queries-per-size-bin 300 \
  --keep-min-score 0.05 \
  --exclude-below-threshold \
  --progress-interval 25 \
  --output /mnt/d/stp_proj_v3/data/cv_predictions.parquet \
  | tee /mnt/d/stp_proj_v3/results/p4_0_full_run_live.log
```

## Pilot Comparison Work

Pilot comparison artifacts were saved under:

- `/mnt/d/stp_proj_v3/results/p4_0_pilot_compare_20260312`

Compared settings:

- `baseline`
- `excl_thr`
- `score_005`
- `excl_thr_score_005`

Main conclusion:

- the strongest sparsity lever was `keep-min-score`
- the best balanced pilot setting was:
  - `--thr-preset stp2014`
  - `--keep-min-score 0.05`
  - `--exclude-below-threshold`

## Latest Full Production Run

### Output files

- `/mnt/d/stp_proj_v3/data/cv_predictions.parquet`
- `/mnt/d/stp_proj_v3/data/cv_predictions.query_summary.csv`
- `/mnt/d/stp_proj_v3/data/cv_predictions.size_bin_precision.csv`
- `/mnt/d/stp_proj_v3/data/cv_predictions.run_summary.json`
- `/mnt/d/stp_proj_v3/results/p4_0_full_run_live.log`

### Latest run configuration

- sampled queries: `2100`
- human targets: `4524`
- coefficient family: `K1 paired_sum`
- threshold preset: `stp2014`
- `keep_top_rank = 100`
- `keep_min_score = 0.05`
- `exclude_below_threshold = true`

### Latest run output summary

- output rows: `2,552,298`
- rows/query min: `100`
- rows/query median: `1132.5`
- rows/query max: `3456`
- full-target queries: `0`

### Query-level quality metrics

- Top-1 hit rate: `1.95%`
- Top-5 hit rate: `6.10%`
- Top-10 hit rate: `9.19%`
- Top-15 hit rate: `11.52%`
- Top-50 hit rate: `24.33%`
- Top-100 hit rate: `36.52%`
- MRR: `0.0528`
- median best true rank: `143`

### Size-bin observations

Relative strengths:

- `16_20` and `21_25` performed better at small cutoffs
- `21_25`, `31_35`, and `36_40` were relatively stronger at Top-100

Relative weaknesses:

- `01_15`
- `41_plus`

### Interpretation

This run is materially better than earlier dense outputs:

- storage volume is reduced
- no query retained all targets
- ranking metrics improved versus looser threshold settings

However:

- the output is still moderately dense
- raw `combined_score` is not yet a calibrated probability

## Analysis and Report Artifacts

### Review plots

- `/mnt/d/stp_proj_v3/results/cv_predictions_review`

### Quality report

- `/mnt/d/stp_proj_v3/results/p4_0_quality_20260312`

### Top-K and logistic rank-fit analysis

- `/mnt/d/stp_proj_v3/results/p4_1_cv_analysis_20260313`

Key files:

- `topk_overall.csv`
- `topk_by_size_bin.csv`
- `size_bin_logistic_params.csv`
- `size_bin_logistic_curves.csv`

### Final report package

- `/mnt/d/stp_proj_v3/results/p4_2_final_report_20260313`

### Score calibration package

- `/mnt/d/stp_proj_v3/results/p4_3_score_calibration_20260313`

Key files:

- `score_calibration_params.csv`
- `empirical_score_bins.csv`
- `score_calibration_curves.csv`
- `01_size_bin_score_calibration.png`

## Calibration Outputs

### Rank-calibrated output

- `/mnt/d/stp_proj_v3/data/cv_predictions_rank_calibrated.parquet`
- `/mnt/d/stp_proj_v3/data/cv_predictions_rank_calibrated.summary.json`

Added column:

- `calibrated_rank_hit_rate`

This should be treated as a rank-interpretation aid, not as a target posterior probability.

### Score-calibrated output

- `/mnt/d/stp_proj_v3/data/cv_predictions_score_calibrated.parquet`
- `/mnt/d/stp_proj_v3/data/cv_predictions_score_calibrated.summary.json`

Added column:

- `calibrated_score_precision`

This is the most useful calibrated value currently available for reporting and threshold support.

## Practical Recommendations

### What to use now

- ranking and candidate ordering:
  - `combined_score`
- reporting and post-hoc interpretability:
  - `calibrated_score_precision`

### What remains to improve

1. Compare additional `p3` coefficient outputs with the same pilot framework.
2. Consider size-bin-specific decision thresholds.
3. If needed, separate evaluation-time scoring from export-time filtering more explicitly.
4. If ranking must improve further, revisit alternative coefficient families beyond the current K1 paired setting.

## Current Status

As of this summary:

- the `p4` generation script is operational
- external-terminal live monitoring is supported
- the latest full run completed successfully
- downstream quality analysis is implemented
- both rank-based and score-based calibration workflows are implemented
- the score-based calibrated field is the current best interpretation layer
