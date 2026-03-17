#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/mnt/d/stp_proj_v3"
PYTHON_BIN="/home/xotn91/.local/share/mamba/envs/stp/bin/python3.10"
LOG_DIR="$PROJECT_ROOT/features_store/logs"

MASTER_PARQUET="$PROJECT_ROOT/features_store/chembl36_stp_training_set.neg_ge100_aug.parquet"
ES5D_META="$PROJECT_ROOT/features_store/es5d_meta_db.parquet"

BASE_OUT_DIR="$PROJECT_ROOT/features_store/p3_ready_pos_lt10_neg7plus_20260316"
FINAL_OUT_DIR="$PROJECT_ROOT/features_store/p3_ready_pos_lt10_neg10_keepall_20260316"

mkdir -p "$LOG_DIR" "$BASE_OUT_DIR" "$FINAL_OUT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/prepare_pos_lt10_current_mode_${TS}.log"

{
    echo "START: $(date '+%F %T')"
    echo "LOG  : $LOG_FILE"
    echo "BASE : $BASE_OUT_DIR"
    echo "FINAL: $FINAL_OUT_DIR"

    "$PYTHON_BIN" "$PROJECT_ROOT/script/p3_0_prepare_current_mode_meta.py" \
        --master_parquet "$MASTER_PARQUET" \
        --es5d_meta "$ES5D_META" \
        --out_dir "$BASE_OUT_DIR" \
        --min_neg_per_pair 7 \
        --max_neg_per_pair 10

    "$PYTHON_BIN" "$PROJECT_ROOT/script/p1_4_augment_negatives_current_mode.py" \
        --base_meta "$BASE_OUT_DIR/final_training_meta.parquet" \
        --train_parquet "$MASTER_PARQUET" \
        --es5d_meta "$ES5D_META" \
        --out_dir "$FINAL_OUT_DIR" \
        --target_neg_per_pair 10 \
        --prefer_ge_uM 100

    echo "DONE : $(date '+%F %T')"
} 2>&1 | tee "$LOG_FILE"
