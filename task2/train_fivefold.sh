#!/usr/bin/env bash
set -euo pipefail

: "${nnUNet_results:?Set nnUNet_results}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATASET=Dataset502_MVAA_Task2_TEE
TRAINER=nnUNetTrainer_250epochs
MODEL="$nnUNet_results/$DATASET/${TRAINER}__nnUNetPlans__3d_fullres"

for fold in 0 1 2 3 4; do
  fold_dir="$MODEL/fold_$fold"
  if [[ ! -s "$fold_dir/checkpoint_final.pth" ]]; then
    nnUNetv2_train "$DATASET" 3d_fullres "$fold" -tr "$TRAINER" --npz
  else
    echo "fold $fold: checkpoint_final.pth exists; skipping training"
  fi

  if [[ ! -d "$fold_dir/validation_final" ]]; then
    nnUNetv2_train "$DATASET" 3d_fullres "$fold" -tr "$TRAINER" --val --npz
    mv "$fold_dir/validation" "$fold_dir/validation_final"
  fi

  if [[ ! -d "$fold_dir/validation_best" ]]; then
    nnUNetv2_train "$DATASET" 3d_fullres "$fold" -tr "$TRAINER" \
      --val --val_best --npz
    mv "$fold_dir/validation" "$fold_dir/validation_best"
  fi
done

python "$SCRIPT_DIR/select_checkpoints.py" --results-root "$MODEL"
echo "TASK2_TRAINING_COMPLETE"
