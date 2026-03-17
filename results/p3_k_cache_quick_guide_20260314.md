# P3 K-Cache Quick Guide

## Purpose

- Run `p3_1_1` with `K=5` first
- Reuse the generated feature matrix for `K=1`
- Limit reuse to `K`-only changes

## Script

- Runner:
  - `/mnt/d/stp_proj_v3/script/run_p3_1_1_full_aug_scaf_off_k5_then_k1_reuse_20260314.sh`

## Run

```bash
cd /mnt/d/stp_proj_v3/script
./run_p3_1_1_full_aug_scaf_off_k5_then_k1_reuse_20260314.sh
```

## What It Does

1. Runs `p3_1_1` with `K=5`
2. Saves feature cache
3. Runs `p3_1_1` with `K=1`
4. Reuses the `K=5` feature cache for the `K=1` run

## Output Paths

- K=5:
  - `/mnt/d/stp_proj_v3/features_store/p3_run_full_aug_scaf_off_k5_then_k1_reuse_20260314/k5_run`
- K=1:
  - `/mnt/d/stp_proj_v3/features_store/p3_run_full_aug_scaf_off_k5_then_k1_reuse_20260314/k1_reuse_run`
- Cache:
  - `/mnt/d/stp_proj_v3/features_store/p3_run_full_aug_scaf_off_k5_then_k1_reuse_20260314/feature_cache`
- Logs:
  - `/mnt/d/stp_proj_v3/features_store/logs/`

## Important Limits

- Reuse works only when `K` changes
- Reuse does not apply if these change:
  - `scaffold_mask`
  - threshold settings
  - `k_policy`
  - `cv_scheme`
  - `cutoff_year`
  - input dataset paths

