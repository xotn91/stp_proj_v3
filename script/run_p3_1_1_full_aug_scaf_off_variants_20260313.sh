#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/mnt/d/stp_proj_v3"
PYTHON_BIN="/home/xotn91/.local/share/mamba/envs/stp/bin/python3.10"
SCRIPT_PATH="$PROJECT_ROOT/script/p3_1_1_K1_paired_trainer_fast.py"
LOG_DIR="$PROJECT_ROOT/features_store/logs"

META_FILE="$PROJECT_ROOT/features_store/p3_ready_pos_le1uM_neg10_keepall_20260310/final_training_meta.with_dmpfill_plus16.parquet"
FP2_MEMMAP="$PROJECT_ROOT/features_store/p3_ready_pos_le1uM_neg10_keepall_20260310/fp2_aligned.with_dmp16.memmap"
ES5D_MEMMAP="$PROJECT_ROOT/features_store/p3_ready_pos_le1uM_neg10_keepall_20260310/es5d_db_k20.with_dmp16.memmap"

mkdir -p "$LOG_DIR"

run_case() {
    local label="$1"
    local out_dir="$2"
    shift 2

    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local log_file="$LOG_DIR/${label}_${ts}.log"

    mkdir -p "$out_dir"

    echo "============================================================"
    echo "START: $label"
    echo "TIME : $(date '+%F %T')"
    echo "LOG  : $log_file"
    echo "OUT  : $out_dir"
    echo "============================================================"

    PYTHONUNBUFFERED=1 \
        "$PYTHON_BIN" "$SCRIPT_PATH" \
        --meta_file "$META_FILE" \
        --fp2_memmap "$FP2_MEMMAP" \
        --es5d_memmap "$ES5D_MEMMAP" \
        --out_dir "$out_dir" \
        --c_reg 10.0 \
        --cutoff_year 2023 \
        --batch_size 4096 \
        --chunk_3d_b 128 \
        --min_neg_per_pair 7 \
        --max_neg_per_pair 10 \
        --scaffold_mask off \
        --assay_mask off \
        "$@" 2>&1 | tee "$log_file"

    echo
    echo "DONE : $label"
    echo "TIME : $(date '+%F %T')"
    echo
}

run_case \
    "p3_1_1_full_aug_scaf_off_k5" \
    "$PROJECT_ROOT/features_store/p3_run_full_aug_scaf_off_k5_20260313" \
    --k_mode 5

run_case \
    "p3_1_1_full_aug_scaf_off_k1_stp2019" \
    "$PROJECT_ROOT/features_store/p3_run_full_aug_scaf_off_k1_stp2019_20260313" \
    --k_mode 1 \
    --thr_preset stp2019

run_case \
    "p3_1_1_full_aug_scaf_off_k5_stp2019" \
    "$PROJECT_ROOT/features_store/p3_run_full_aug_scaf_off_k5_stp2019_20260313" \
    --k_mode 5 \
    --thr_preset stp2019
