# MV-MASeg: MVAA 2026 Task 1 and Task 2

This repository contains the paper-aligned training and inference code for our
MVAA 2026 submission. Our team ranked **4th overall** in the challenge.

This release covers:

- **Task 1 (CT):** five-fold 3D full-resolution nnU-Net with
  consistency-guided pseudo-label selection, selective self-training, mirror
  test-time augmentation (TTA), and largest-connected-component (LCC)
  refinement.
- **Task 2 (3D TEE):** five-fold 3D full-resolution nnU-Net with
  validation-based checkpoint selection, mirror TTA, probability averaging,
  and combined-foreground LCC refinement.

Post-competition SegResNet experiments are intentionally excluded because they
were not part of the final paper.

## Results

Metrics below are the Codabench test results reported in the paper.

| Task | Method | DSC | HD | ASD |
|---|---|---:|---:|---:|
| Task 1 CT | nnU-Net + selected pseudo-label fold + TTA + LCC | 0.860973 | 4.544340 | 0.273540 |
| Task 2 TEE | selected five-fold nnU-Net + TTA + foreground LCC | 0.849475 | 11.079997 | 0.597933 |

## Repository layout

```text
configs/                 Exact plans, splits, and method metadata
common/                  Shared post-processing and submission conversion
task1/                   CT data conversion, pseudo-labeling, training, inference
task2/                   TEE data conversion, checkpoint selection, training, inference
environment.yml          Reproducible Conda environment
requirements.txt         Exact Python package versions from the training server
```

Challenge data and model checkpoints are not redistributed. Place them locally
as described below.

## Environment

The original environment was Python 3.10.20, PyTorch 2.7.1+cu118,
nnU-Net v2.8.0, CUDA 11.8, and an NVIDIA RTX 4080 (32 GB).

```bash
conda env create -f environment.yml
conda activate mvaa
```

Alternatively, create a Python 3.10 environment and install:

```bash
python -m pip install -r requirements.txt
```

Set the three nnU-Net paths before running any command:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

## Expected challenge data layout

```text
reference_data/
├── t1_ct/
│   ├── train/
│   │   ├── labeled/images/0001.nii.gz ... 0027.nii.gz
│   │   ├── labeled/labels/0001-seg.nii.gz ... 0027-seg.nii.gz
│   │   └── unlabeled/0001.nii.gz ... 1040.nii.gz
│   └── val/images/0001.nii.gz ... 0030.nii.gz
└── t2_tee/
    ├── train/train_001-US.nii.gz and train_001-label.nii.gz ...
    └── val/images/val_001-US.nii.gz ... val_020-US.nii.gz
```

## Task 1: CT

### 1. Prepare and preprocess

```bash
python task1/prepare_dataset.py \
  --source-root /path/to/reference_data/t1_ct \
  --nnunet-raw "$nnUNet_raw"

nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
cp configs/task1_splits_final.json \
  "$nnUNet_preprocessed/Dataset501_MVAA_Task1_CT/splits_final.json"
```

The released plan is provided at `configs/task1_nnunet_plans.json`. It uses CT
normalization, spacing `(0.5, 0.357421875, 0.357421875)` in nnU-Net's
transposed `(z, y, x)` order, patch size `(112, 128, 160)`, and batch size 2.

### 2. Train five supervised folds

```bash
bash task1/train_supervised.sh
```

Training uses `nnUNetTrainer_250epochs`, the 3D full-resolution PlainConvUNet,
deep-supervised Dice + cross-entropy loss, and the standard nnU-Net data
augmentation. The selected supervised checkpoints are `best, final, best,
final, final` for folds 0--4.

### 3. Select pseudo-labels and fine-tune fold 3

Run the complete pseudo-label pipeline after supervised training:

```bash
export TASK1_DATA_ROOT=/path/to/reference_data/t1_ct
export TASK1_WORK=/path/to/task1_pseudo_work
bash task1/run_self_training.sh
```

The script performs five-teacher prediction on all 1,040 unlabeled CT scans,
filters candidates using agreement, volume plausibility, and connectedness,
selects the top 200, regenerates their labels using the selected fold-0 teacher
with mirror TTA, and fine-tunes fold 3 for 100 epochs. The pseudo-label weight
ramps from 0.1 to 0.3, stays at 0.3, and decays to zero during the last 30% of
fine-tuning.

### 4. Final inference

```bash
export TASK1_OUTPUT=/path/to/task1_output
bash task1/infer_final.sh
```

The final probability is

```text
P_final = P_supervised_5fold + 0.2 * (P_pseudo_fold3 - P_supervised_fold3)
```

which is exactly an equal-weight five-member ensemble in which the supervised
fold-3 member is replaced by the pseudo-label student. LCC is applied after
thresholding.

## Task 2: 3D TEE

### 1. Prepare and preprocess

```bash
python task2/prepare_dataset.py \
  --source-root /path/to/reference_data/t2_tee \
  --nnunet-raw "$nnUNet_raw"

nnUNetv2_plan_and_preprocess -d 502 --verify_dataset_integrity
cp configs/task2_splits_final.json \
  "$nnUNet_preprocessed/Dataset502_MVAA_Task2_TEE/splits_final.json"
```

Task 2 uses dataset ID 502 in this public release to avoid colliding with Task
1. The original experiments used ID 501 independently. The released plan has
Z-score normalization, spacing `(0.5395808816, 0.2322079986, 0.3726583719)`,
patch size `(96, 128, 160)`, and batch size 2.

### 2. Train and select checkpoints

```bash
bash task2/train_fivefold.sh
```

Each fold uses `nnUNetTrainer_250epochs`. After validating both final and best
checkpoints, select the higher foreground-mean Dice per fold:

```bash
python task2/select_checkpoints.py \
  --results-root "$nnUNet_results/Dataset502_MVAA_Task2_TEE/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
```

The paper submission selected `best, final, final, best, final` for folds 0--4.

### 3. Final inference

```bash
export TASK2_OUTPUT=/path/to/task2_output
bash task2/infer_final.sh
```

nnU-Net performs sliding-window inference, mirror TTA, and equal probability
averaging over all five selected folds. Post-processing keeps the largest
6-connected component of the combined foreground while preserving leaflet
class labels inside that component.

## Checkpoints

The scripts expect the following filenames inside the standard nnU-Net results
folders:

- supervised/final models: `checkpoint_best.pth`, `checkpoint_final.pth`
- selected model alias/copy: `checkpoint_selected.pth`
- Task 1 pseudo student: dataset `Dataset511_MVAA_Task1_PseudoTop200`, fold 3,
  `checkpoint_best.pth`

Weights are not included because of their size and challenge-data terms. The
training scripts recreate them from the released splits and configuration.

## Reproducibility notes

- Configuration arrays follow nnU-Net's transposed `(z, y, x)` convention.
- TTA is enabled by default in `nnUNetv2_predict`; do not pass `--disable_tta`.
- Task 2 LCC is computed on `segmentation > 0`, not independently per class.
- The exact checkpoint choices and paper metrics are recorded in
  `configs/method_manifest.json`.
- Dataset IDs are unique in this combined release (501 for Task 1, 502 for Task
  2). This changes only the storage name, not the model, plans, splits, or
  predictions.

## Citation

If you use this code, please cite the MV-MASeg challenge paper. Full citation
metadata will be added after publication.
