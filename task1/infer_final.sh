#!/usr/bin/env bash
set -euo pipefail

: "${nnUNet_raw:?Set nnUNet_raw}"
: "${nnUNet_results:?Set nnUNet_results}"
: "${TASK1_OUTPUT:?Set TASK1_OUTPUT}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
SUP_DATASET=Dataset501_MVAA_Task1_CT
PSEUDO_DATASET=Dataset511_MVAA_Task1_PseudoTop200
SUP_TRAINER=nnUNetTrainer_250epochs
PSEUDO_TRAINER=nnUNetTrainerPseudoTop200
INPUT="$nnUNet_raw/$SUP_DATASET/imagesTs"
MAPPING="$nnUNet_raw/$SUP_DATASET/mvaa_task1_mapping.json"

mkdir -p "$TASK1_OUTPUT"
PLAIN5="$TASK1_OUTPUT/plain_selected_5fold_tta"
PLAIN3="$TASK1_OUTPUT/plain_fold3_tta"
PSEUDO3="$TASK1_OUTPUT/pseudo_fold3_tta"
REPLACED="$TASK1_OUTPUT/final_raw"
FINAL="$TASK1_OUTPUT/t1_ct"

nnUNetv2_predict -i "$INPUT" -o "$PLAIN5" -d "$SUP_DATASET" \
  -c 3d_fullres -f 0 1 2 3 4 -tr "$SUP_TRAINER" \
  -chk checkpoint_selected.pth --save_probabilities --continue_prediction -npp 3 -nps 3
nnUNetv2_predict -i "$INPUT" -o "$PLAIN3" -d "$SUP_DATASET" \
  -c 3d_fullres -f 3 -tr "$SUP_TRAINER" \
  -chk checkpoint_selected.pth --save_probabilities --continue_prediction -npp 3 -nps 3
nnUNetv2_predict -i "$INPUT" -o "$PSEUDO3" -d "$PSEUDO_DATASET" \
  -c 3d_fullres -f 3 -tr "$PSEUDO_TRAINER" \
  -chk checkpoint_best.pth --save_probabilities --continue_prediction -npp 3 -nps 3

python "$SCRIPT_DIR/replace_fold_probability.py" \
  --ensemble-dir "$PLAIN5" --old-fold-dir "$PLAIN3" \
  --new-fold-dir "$PSEUDO3" --output-dir "$REPLACED" \
  --fold-weight 0.2 --threshold 0.5
python "$REPO_ROOT/common/finalize_predictions.py" \
  --task 1 --prediction-dir "$REPLACED" --mapping-json "$MAPPING" \
  --output-dir "$FINAL" --overwrite
echo "TASK1_INFERENCE_COMPLETE OUTPUT=$FINAL"
