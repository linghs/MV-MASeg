#!/usr/bin/env bash
set -euo pipefail

: "${nnUNet_raw:?Set nnUNet_raw}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${nnUNet_results:?Set nnUNet_results}"

DATASET=Dataset501_MVAA_Task1_CT
TRAINER=nnUNetTrainer_250epochs
MODEL="$nnUNet_results/$DATASET/${TRAINER}__nnUNetPlans__3d_fullres"

for fold in 0 1 2 3 4; do
  if [[ ! -s "$MODEL/fold_$fold/checkpoint_final.pth" ]]; then
    nnUNetv2_train "$DATASET" 3d_fullres "$fold" -tr "$TRAINER" --npz
  else
    echo "fold $fold: checkpoint_final.pth exists; skipping training"
  fi
done

choices=(best final best final final)
for fold in 0 1 2 3 4; do
  fold_dir="$MODEL/fold_$fold"
  ln -sfn "checkpoint_${choices[$fold]}.pth" "$fold_dir/checkpoint_selected.pth"
done

echo "TASK1_SUPERVISED_COMPLETE"
