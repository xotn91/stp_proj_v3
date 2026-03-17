# P3 K-Cache Work Summary

Date: 2026-03-14

## Scope

- Target script: `/mnt/d/stp_proj_v3/script/p3_1_1_K1_paired_trainer_fast.py`
- Goal:
  - review whether `Generating Feature Matrix` can be reused
  - support `K`-only reuse
  - keep `scaffold on/off` reuse out of scope for now
  - prepare an external-terminal runner that executes `K=5` first and then reuses it for `K=1`

## Main Conclusions

### 1. What `Generating Feature Matrix` does

- The step reads the prepared FP2/ES5D memmap inputs and loads them into GPU-backed tensors.
- It computes per-query similarity against target actives:
  - 2D via matrix multiplication / Tanimoto-style scoring
  - 3D via `torch.cdist`
- It then builds a reduced feature matrix of shape roughly:
  - `N rows x (K * 2 columns)`
- The reduced matrix is converted into `train_df` / `future_features_df` and used by later CV/training/OOT stages.

### 2. Why this is the main bottleneck

- The expensive part is not the downstream LR/HGB training itself.
- The expensive part is repeated 2D/3D similarity generation for all rows during `Generating Feature Matrix`.
- 3D similarity (`torch.cdist`) is the heaviest section.

### 3. Reuse feasibility by option type

- `K=5 -> K=1`:
  - reusable
  - if the same feature-generation conditions are used, `K=1` can reuse the first top-1 slice from a previously generated `K=5` feature matrix

- `scaffold off -> scaffold on`:
  - not reusable with the current final-feature-only output
  - reason: scaffold masking changes candidate validity before top-K selection, so the selected top-K set can change
  - current code stores only final top-K features, not enough intermediate candidate information

- Threshold changes (`stp2019`, etc.):
  - not reusable in the current implementation
  - thresholding/normalization is applied during feature generation

## Code Changes

Updated file:
- `/mnt/d/stp_proj_v3/script/p3_1_1_K1_paired_trainer_fast.py`

Added capability:
- optional feature cache for `K`-only reuse

New CLI option:
- `--feature_cache_dir`

Behavior:
- when the same feature-generation signature is reused and a cached feature matrix with `K >= requested K` exists:
  - the script loads the cached parquet
  - slices the feature columns down to the requested `K`
  - skips re-running `Generating Feature Matrix`

Cache signature includes:
- meta/fp2/es5d input paths
- `k_policy`
- `scaffold_mask`
- threshold settings
- `cv_scheme`
- pair rules
- `cutoff_year`
- other feature-generation-affecting options

This means:
- only true `K`-only reruns reuse cache
- changes to scaffold/threshold/policy/cv/cutoff produce a different cache signature and force recomputation

## External Terminal Runner Added

New runner:
- `/mnt/d/stp_proj_v3/script/run_p3_1_1_full_aug_scaf_off_k5_then_k1_reuse_20260314.sh`

Purpose:
- run `K=5` first
- save feature cache
- run `K=1` next with the same conditions
- reuse the `K=5` feature cache

Execution:

```bash
cd /mnt/d/stp_proj_v3/script
./run_p3_1_1_full_aug_scaf_off_k5_then_k1_reuse_20260314.sh
```

Run layout:
- K=5 output:
  - `/mnt/d/stp_proj_v3/features_store/p3_run_full_aug_scaf_off_k5_then_k1_reuse_20260314/k5_run`
- K=1 output:
  - `/mnt/d/stp_proj_v3/features_store/p3_run_full_aug_scaf_off_k5_then_k1_reuse_20260314/k1_reuse_run`
- cache:
  - `/mnt/d/stp_proj_v3/features_store/p3_run_full_aug_scaf_off_k5_then_k1_reuse_20260314/feature_cache`
- logs:
  - `/mnt/d/stp_proj_v3/features_store/logs/`

## Data / Scope Notes

- Current work was limited to the existing `<=1 uM` positive-ready dataset line.
- The separate `<10 uM` positive-threshold track was intentionally excluded from this change.

## Operational Notes

- `K=1` can reuse `K=5`
- `K=5` cannot be reconstructed from `K=1`
- `scaffold on/off` still requires separate feature generation
- threshold-setting changes still require separate feature generation

