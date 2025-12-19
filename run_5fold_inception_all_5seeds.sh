#!/usr/bin/env bash
set -euo pipefail

TASKSET=(
  "folds_tsv_DDK_ANALYSIS_PATAKA"
  "folds_tsv_SUSTAINED-VOWELS_onlyA123"
  "folds_tsv_SENTENCES"
)
SCRIPT="/home/yzhong/data/storage2/gits/CNN-PD-Voice/Inception_pd_detection_voice.py"
BASE_OUT_ROOT="/home/yzhong/data/storage2/gits/CNN-PD-Voice/outputs/inception_runs_5fold_all_5seeds"

EPOCHS=20
PATIENCE=5
MIN_EPOCHS=5
BATCH_SIZE=4
IMG_SIZE=600

NUM_RUNS=5
BASE_SEED=42

for task in "${TASKSET[@]}"; do
  FOLDS_ROOT="/data/storage2/gits/CNN-PD-Voice/split_5fold/folds_v2.1_early_validation_newcut/${task}"
  DATASET_NAME="$(basename "$FOLDS_ROOT")"
  OUT_ROOT="$BASE_OUT_ROOT/$DATASET_NAME"

  mkdir -p "$OUT_ROOT"

  for run_idx in 1 2 3 4 5; do
    seed=$((BASE_SEED + run_idx - 1))
    run_root="$OUT_ROOT/run_${run_idx}"
    mkdir -p "$run_root"

    for k in 1 2 3 4 5; do
      fold_dir="$FOLDS_ROOT/fold_${k}"
      train_tsv="$fold_dir/sub_splits/train.tsv"
      val_tsv="$fold_dir/sub_splits/val_early6PD6HC.tsv"
      test_tsv="$fold_dir/test_early6PD6HC.tsv"

      out_dir="$run_root/fold_${k}"
      mkdir -p "$out_dir"

      python "$SCRIPT" \
        --train_tsv "$train_tsv" \
        --val_tsv "$val_tsv" \
        --test_tsv "$test_tsv" \
        --output_dir "$out_dir" \
        --epochs "$EPOCHS" \
        --early_stop_patience "$PATIENCE" \
        --min_epochs "$MIN_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --img_size "$IMG_SIZE" \
        --seed "$seed"
    done
  done
done
