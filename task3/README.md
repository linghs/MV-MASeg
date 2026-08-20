# Task 3: semi-supervised surgical-video segmentation

This folder contains the Task 3 code supplied for the post-challenge material
package. It segments the mitral-valve foreground (challenge label ID 10) from
surgical video frames.

The main paper configuration combines:

- a DINOv2 ViT-L/14 encoder;
- a lightweight CNN decoder;
- a 20-epoch encoder freeze followed by low-learning-rate fine-tuning;
- an EMA Mean Teacher for unlabeled frames;
- confidence-filtered positive and negative pseudo-label regions;
- Dice + focal supervised loss and flip TTA.

## Included files

```text
dataset.py                       data discovery, label-tar parsing, augmentation
model_factory.py                 DINOv2/DINOv3/CNN/nnU-Net-style model variants
train.py                         supervised and Mean Teacher training
generate_task3_predictions.py   checkpoint-driven inference and submission JSON
visualize_task3_comparison.py   qualitative comparison figure generator
utils.py                         seeding, logging, metrics, JSON helpers
```

The original ZIP also contained Jupyter checkpoint copies, `cufile.log`, and
three empty pretraining placeholders. They were intentionally excluded because
they are not source-of-truth implementations.

## Data layout

Labeled samples are grouped by video. Each annotated image has an associated
tar file containing the NIfTI label:

```text
labeled/
└── REC_xxx/
    ├── REC_xxx_000118.png
    └── REC_xxx_000118_png_Label.tar

unlabeled/images/**/*.png
test/images/**/*.png
```

The training split is performed by video rather than by frame to prevent
near-duplicate frames from leaking into validation. The paper dataset contains
180 labeled frames (118 with foreground), 1,379 unlabeled frames, and 48 test
frames.

## Dependencies

Install the repository-level environment first:

```bash
conda env create -f environment.yml
conda activate mvaa
```

Task 3 additionally uses `Pillow`, `timm`, `segmentation-models-pytorch`,
`safetensors`, and `matplotlib`; these are included in the root dependency
files.

## Paper configuration

```bash
python task3/train.py \
  --labeled-root /path/to/labeled \
  --unlabeled-root /path/to/unlabeled/images \
  --output-dir task3/runs/dinov2_mean_teacher \
  --arch dinov2_unet \
  --dinov2-name dinov2_vitl14 \
  --no-dinov2-pretrained \
  --dinov2-checkpoint /path/to/model.safetensors \
  --dinov2-decoder-channels 256 \
  --freeze-dinov2-epochs 20 \
  --encoder-lr 1e-5 \
  --target-label 10 \
  --image-size 336 588 \
  --epochs 150 \
  --batch-size 8 \
  --unlabeled-batch-size 8 \
  --lr 2e-4 \
  --weight-decay 1e-5 \
  --loss-type dice_focal \
  --semi-warmup-epochs 20 \
  --unsup-weight 0.6 \
  --unsup-ramp-epochs 30 \
  --ema-decay 0.99 \
  --pseudo-pos-thr 0.70 \
  --pseudo-neg-thr 0.10 \
  --val-video-count 2 \
  --val-only-fg \
  --val-tta \
  --amp
```

To let `timm` obtain pretrained weights instead, omit
`--no-dinov2-pretrained` and `--dinov2-checkpoint`.

Training creates:

```text
runs/<experiment>/
├── config.json
├── split.json
├── history.csv
├── train.log
├── final_metrics.json
└── checkpoints/bestmodel.pth
```

The checkpoint contains student weights, EMA teacher weights, command-line
arguments, validation metrics, and the selected probability threshold. The
student is used for validation/model selection and final inference; the EMA
teacher is used to generate pseudo-labels during training.

## Conservative alternative from the supplied archive

The supplied archive also documented a conservative experiment named
`task3_search_v07_conservative_semi`. Its differing settings were:

```text
semi_warmup_epochs = 30
unsup_weight       = 0.4
unsup_ramp_epochs  = 40
ema_decay          = 0.995
pseudo_pos_thr     = 0.75
pseudo_neg_thr     = 0.08
```

These are retained here as an optional experiment, not presented as the paper
configuration.

## Inference and submission export

```bash
python task3/generate_task3_predictions.py \
  --checkpoint task3/runs/dinov2_mean_teacher/checkpoints/bestmodel.pth \
  --data-dir /path/to/test/images \
  --output-dir task3/submission/t3_vid
```

Optional flags:

- `--video-folder VIDEO_ID` limits inference to a subfolder and may be repeated.
- `--device auto|cuda|cpu` chooses the execution device.
- `--no-tta` disables the four-way flip ensemble.
- `--no-amp` disables CUDA automatic mixed precision.

Outputs are single-channel PNG masks with values 0/255 at the original image
resolution, plus `task3_predictions.json` in the output directory.

## Qualitative comparison

```bash
python task3/visualize_task3_comparison.py \
  --runs-dir task3/runs \
  --data-root /path/to/labeled \
  --output task3/figures/task3_qualitative_comparison.png
```

## Reported result

The paper reports DSC 0.817, HD 42.05, and ASD 7.08 for the DINOv2-UNet +
freeze-thaw + Semantic Mean Teacher configuration.
