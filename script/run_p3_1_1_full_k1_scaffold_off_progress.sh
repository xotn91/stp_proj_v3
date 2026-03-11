#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/mnt/d/stp_proj_v3"
PYTHON_BIN="/home/xotn91/.local/share/mamba/envs/stp/bin/python3.10"
SCRIPT_PATH="$PROJECT_ROOT/script/p3_1_1_K1_paired_trainer_fast.py"

META_FILE="$PROJECT_ROOT/features_store/p3_ready_pos_le1uM_neg10_keepall_20260310/final_training_meta.with_dmpfill_plus16.parquet"
FP2_MEMMAP="$PROJECT_ROOT/features_store/p3_ready_pos_le1uM_neg10_keepall_20260310/fp2_aligned.with_dmp16.memmap"
ES5D_MEMMAP="$PROJECT_ROOT/features_store/p3_ready_pos_le1uM_neg10_keepall_20260310/es5d_db_k20.with_dmp16.memmap"
OUT_DIR="$PROJECT_ROOT/features_store/p3_seq_full_scaf_off_20260310"

mkdir -p "$OUT_DIR"

exec env PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" "$SCRIPT_PATH" \
    --meta_file "$META_FILE" \
    --fp2_memmap "$FP2_MEMMAP" \
    --es5d_memmap "$ES5D_MEMMAP" \
    --out_dir "$OUT_DIR" \
    --k_mode 1 \
    --k_policy paired_sum \
    --c_reg 10.0 \
    --cutoff_year 2023 \
    --batch_size 4096 \
    --chunk_3d_b 128 \
    --min_neg_per_pair 7 \
    --max_neg_per_pair 10 \
    --scaffold_mask off \
    --assay_mask off
