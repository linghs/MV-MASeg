#!/usr/bin/env bash
set -euo pipefail

: "${nnUNet_raw:?Set nnUNet_raw}"
: "${TASK2_OUTPUT:?Set TASK2_OUTPUT}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
DATASET=Dataset502_MVAA_Task2_TEE
TRAINER=nnUNetTrainer_250epochs
INPUT="$nnUNet_raw/$DATASET/imagesTs"
MAPPING="$nnUNet_raw/$DATASET/mvaa_task2_mapping.json"
RAW="$TASK2_OUTPUT/raw_fivefold_tta"
FINAL="$TASK2_OUTPUT/t2_tee"

mkdir -p "$TASK2_OUTPUT"
nnUNetv2_predict -i "$INPUT" -o "$RAW" -d "$DATASET" \
  -c 3d_fullres -f 0 1 2 3 4 -tr "$TRAINER" \
  -chk checkpoint_selected.pth --save_probabilities --continue_prediction -npp 2 -nps 2
python "$REPO_ROOT/common/finalize_predictions.py" \
  --task 2 --prediction-dir "$RAW" --mapping-json "$MAPPING" \
  --output-dir "$FINAL" --overwrite
echo "TASK2_INFERENCE_COMPLETE OUTPUT=$FINAL"
