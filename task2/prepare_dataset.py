#!/usr/bin/env python3
"""Convert the released MVAA Task 2 layout to nnU-Net v2 format."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DATASET_FOLDER = "Dataset502_MVAA_Task2_TEE"


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

    dataset = args.nnunet_raw / DATASET_FOLDER
    train_mapping = []
    for image in sorted((args.source_root / "train").glob("train_*-US.nii.gz")):
        original = image.name.removesuffix("-US.nii.gz")
        label = args.source_root / "train" / f"{original}-label.nii.gz"
        if not label.is_file():
            raise FileNotFoundError(label)
        case = f"MVAA_T2_{original.removeprefix('train_')}"
        copy(image, dataset / "imagesTr" / f"{case}_0000.nii.gz", args.overwrite)
        copy(label, dataset / "labelsTr" / f"{case}.nii.gz", args.overwrite)
        train_mapping.append({"nnunet_case": case, "original_case": original})

    test_mapping = []
    for image in sorted((args.source_root / "val" / "images").glob("val_*-US.nii.gz")):
        original = image.name.removesuffix("-US.nii.gz")
        case = f"MVAA_T2_{original}"
        copy(image, dataset / "imagesTs" / f"{case}_0000.nii.gz", args.overwrite)
        test_mapping.append(
            {
                "nnunet_case": case,
                "original_case": original,
                "expected_submission": f"{original}-pred.nii.gz",
            }
        )

    if len(train_mapping) != 105 or len(test_mapping) != 20:
        raise RuntimeError(
            f"Expected 105 train and 20 test cases; found {len(train_mapping)} and {len(test_mapping)}"
        )
    description = {
        "channel_names": {"0": "US"},
        "labels": {"background": 0, "leaflet_1": 1, "leaflet_2": 2},
        "numTraining": 105,
        "file_ending": ".nii.gz",
        "name": "MVAA_Task2_TEE",
        "description": "MVAA Task 2 3D TEE mitral valve segmentation",
        "reference": "MVAA 2026",
        "licence": "challenge data",
        "release": "1.0",
    }
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "dataset.json").write_text(
        json.dumps(description, indent=2), encoding="utf-8"
    )
    (dataset / "mvaa_task2_mapping.json").write_text(
        json.dumps({"train": train_mapping, "test": test_mapping}, indent=2),
        encoding="utf-8",
    )
    print(f"DATASET={dataset} TRAIN=105 TEST=20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
