#!/usr/bin/env bash
set -euo pipefail

: "${nnUNet_raw:?Set nnUNet_raw}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${nnUNet_results:?Set nnUNet_results}"
: "${TASK1_DATA_ROOT:?Set TASK1_DATA_ROOT to reference_data/t1_ct}"
: "${TASK1_WORK:?Set TASK1_WORK}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SUP_DATASET=Dataset501_MVAA_Task1_CT
PSEUDO_DATASET=Dataset511_MVAA_Task1_PseudoTop200
SUP_TRAINER=nnUNetTrainer_250epochs
PSEUDO_TRAINER=nnUNetTrainerPseudoTop200
SUP_MODEL="$nnUNet_results/$SUP_DATASET/${SUP_TRAINER}__nnUNetPlans__3d_fullres"
PSEUDO_MODEL="$nnUNet_results/$PSEUDO_DATASET/${PSEUDO_TRAINER}__nnUNetPlans__3d_fullres"
UNLABELED_INPUT="$TASK1_WORK/unlabeled_input"
UNLABELED_MANIFEST="$TASK1_WORK/unlabeled_manifest.json"
SELECTION="$TASK1_WORK/selection_top200.json"

trainer_dir=$(python - <<'PY'
from pathlib import Path
import nnunetv2.training.nnUNetTrainer as package
print(Path(package.__file__).resolve().parent)
PY
)
cp "$SCRIPT_DIR/nnUNetTrainerPseudoTop200.py" "$trainer_dir/nnUNetTrainerPseudoTop200.py"

mkdir -p "$TASK1_WORK"
python "$SCRIPT_DIR/prepare_unlabeled.py" \
  --source-dir "$TASK1_DATA_ROOT/train/unlabeled" \
  --output-dir "$UNLABELED_INPUT" --manifest "$UNLABELED_MANIFEST"

for fold in 0 1 2 3 4; do
  output="$TASK1_WORK/fold_$fold"
  nnUNetv2_predict -i "$UNLABELED_INPUT" -o "$output" \
    -d "$SUP_DATASET" -c 3d_fullres -f "$fold" -tr "$SUP_TRAINER" \
    -chk checkpoint_selected.pth --save_probabilities --disable_tta \
    --continue_prediction -npp 4 -nps 4 --disable_progress_bar
  python "$SCRIPT_DIR/summarize_teacher.py" \
    --prediction-dir "$output" \
    --output-json "$TASK1_WORK/fold_${fold}_summary.json" \
    --delete-probabilities
done

python "$SCRIPT_DIR/select_pseudo_labels.py" \
  --root "$TASK1_WORK" \
  --labeled-dir "$TASK1_DATA_ROOT/train/labeled/labels" \
  --output-json "$SELECTION" --top-k 200

SELECTED_INPUT="$TASK1_WORK/selected_top200_input"
SELECTED_MANIFEST="$TASK1_WORK/selected_top200_manifest.json"
SELECTED_TEACHER="$TASK1_WORK/fold_0_top200_tta"
python "$SCRIPT_DIR/prepare_unlabeled.py" \
  --source-dir "$TASK1_DATA_ROOT/train/unlabeled" \
  --output-dir "$SELECTED_INPUT" --manifest "$SELECTED_MANIFEST" \
  --selection-json "$SELECTION"
nnUNetv2_predict -i "$SELECTED_INPUT" -o "$SELECTED_TEACHER" \
  -d "$SUP_DATASET" -c 3d_fullres -f 0 -tr "$SUP_TRAINER" \
  -chk checkpoint_selected.pth --continue_prediction \
  -npp 4 -nps 4 --disable_progress_bar

RAW_PSEUDO="$nnUNet_raw/$PSEUDO_DATASET"
PREP_SUP="$nnUNet_preprocessed/$SUP_DATASET"
PREP_PSEUDO="$nnUNet_preprocessed/$PSEUDO_DATASET"
python "$SCRIPT_DIR/build_pseudo_dataset.py" \
  --selection-json "$SELECTION" --unlabeled-manifest "$UNLABELED_MANIFEST" \
  --teacher-fold-dir "$SELECTED_TEACHER" \
  --labeled-images "$TASK1_DATA_ROOT/train/labeled/images" \
  --labeled-labels "$TASK1_DATA_ROOT/train/labeled/labels" \
  --output-dataset "$RAW_PSEUDO" \
  --original-splits "$PREP_SUP/splits_final.json" --fold 3
python "$SCRIPT_DIR/clone_plans.py" \
  --source-preprocessed "$PREP_SUP" --target-preprocessed "$PREP_PSEUDO" \
  --raw-dataset "$RAW_PSEUDO"
nnUNetv2_preprocess -d 511 -c 3d_fullres -np "${NNUNET_PREPROCESS_PROCESSES:-8}"
python "$SCRIPT_DIR/expand_pseudo_splits.py" \
  --source-splits "$PREP_SUP/splits_final.json" \
  --pseudo-labels "$RAW_PSEUDO/labelsTr" \
  --targets "$RAW_PSEUDO/splits_final.json" "$PREP_PSEUDO/splits_final.json"

if [[ ! -s "$PSEUDO_MODEL/fold_3/checkpoint_final.pth" ]]; then
  nnUNetv2_train "$PSEUDO_DATASET" 3d_fullres 3 -tr "$PSEUDO_TRAINER" \
    -pretrained_weights "$SUP_MODEL/fold_3/checkpoint_selected.pth" --npz
fi
nnUNetv2_train "$PSEUDO_DATASET" 3d_fullres 3 -tr "$PSEUDO_TRAINER" \
  --val --val_best --npz
test -s "$PSEUDO_MODEL/fold_3/checkpoint_best.pth"
echo "TASK1_SELF_TRAINING_COMPLETE"
