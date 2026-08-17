#!/usr/bin/env python3
"""Convert the released MVAA Task 1 layout to nnU-Net v2 format."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DATASET_FOLDER = "Dataset501_MVAA_Task1_CT"


def copy(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    labeled_images = args.source_root / "train" / "labeled" / "images"
    labeled_labels = args.source_root / "train" / "labeled" / "labels"
    test_images = args.source_root / "val" / "images"
    dataset = args.nnunet_raw / DATASET_FOLDER

    train_mapping = []
    for image in sorted(labeled_images.glob("*.nii.gz")):
        original = image.name.removesuffix(".nii.gz")
        label = labeled_labels / f"{original}-seg.nii.gz"
        if not label.is_file():
            raise FileNotFoundError(label)
        case = f"MVAA_{int(original):04d}"
        copy(image, dataset / "imagesTr" / f"{case}_0000.nii.gz", args.overwrite)
        copy(label, dataset / "labelsTr" / f"{case}.nii.gz", args.overwrite)
        train_mapping.append({"nnunet_case": case, "original_case": original})

    test_mapping = []
    for image in sorted(test_images.glob("*.nii.gz")):
        original = image.name.removesuffix(".nii.gz")
        case = f"MVAA_{int(original):04d}"
        copy(image, dataset / "imagesTs" / f"{case}_0000.nii.gz", args.overwrite)
        test_mapping.append(
            {
                "nnunet_case": case,
                "original_case": original,
                "expected_submission": f"{original}-pred.nii.gz",
            }
        )

    if len(train_mapping) != 27 or len(test_mapping) != 30:
        raise RuntimeError(
            f"Expected 27 train and 30 test cases; found {len(train_mapping)} and {len(test_mapping)}"
        )
    description = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "mitral_valve": 1},
        "numTraining": 27,
        "file_ending": ".nii.gz",
        "name": "MVAA_Task1_CT",
        "description": "MVAA Task 1 CT mitral valve segmentation",
        "reference": "MVAA 2026",
        "licence": "challenge data",
        "release": "1.0",
    }
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "dataset.json").write_text(
        json.dumps(description, indent=2), encoding="utf-8"
    )
    (dataset / "mvaa_task1_mapping.json").write_text(
        json.dumps({"train": train_mapping, "test": test_mapping}, indent=2),
        encoding="utf-8",
    )
    print(f"DATASET={dataset} TRAIN=27 TEST=30")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
