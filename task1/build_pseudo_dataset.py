#!/usr/bin/env python3
"""Build an nnU-Net raw dataset from real labels and selected pseudo-labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


def safe_symlink(source: Path, destination: Path) -> None:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.is_symlink() or destination.exists():
        if destination.resolve() != source:
            raise RuntimeError(f"Existing path points elsewhere: {destination}")
    else:
        os.symlink(source, destination)


def largest_component(mask: np.ndarray) -> np.ndarray:
    components, count = ndi.label(mask, ndi.generate_binary_structure(3, 1))
    if count == 0:
        return mask.astype(np.uint8)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return (components == int(sizes.argmax())).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--unlabeled-manifest", type=Path, required=True)
    parser.add_argument("--teacher-fold-dir", type=Path, required=True)
    parser.add_argument("--labeled-images", type=Path, required=True)
    parser.add_argument("--labeled-labels", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--original-splits", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--expected-top-k", type=int, default=200)
    args = parser.parse_args()

    selected = json.loads(args.selection_json.read_text(encoding="utf-8"))["selected"]
    if len(selected) != args.expected_top_k:
        raise RuntimeError(f"Expected {args.expected_top_k} cases, found {len(selected)}")
    selected_ids = {row["case_id"] for row in selected}
    manifest = json.loads(args.unlabeled_manifest.read_text(encoding="utf-8"))["records"]
    source_by_id = {row["case_id"]: Path(row["image"]) for row in manifest}

    images = args.output_dataset / "imagesTr"
    labels = args.output_dataset / "labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for index in range(1, 28):
        case = f"{index:04d}"
        safe_symlink(args.labeled_images / f"{case}.nii.gz", images / f"L{case}_0000.nii.gz")
        safe_symlink(args.labeled_labels / f"{case}-seg.nii.gz", labels / f"L{case}.nii.gz")

    for case_id in sorted(selected_ids):
        safe_symlink(source_by_id[case_id], images / f"P{case_id[1:]}_0000.nii.gz")
        teacher_path = args.teacher_fold_dir / f"{case_id}.nii.gz"
        image = sitk.ReadImage(str(teacher_path))
        pseudo = largest_component(sitk.GetArrayFromImage(image) > 0)
        output = sitk.GetImageFromArray(pseudo)
        output.CopyInformation(image)
        sitk.WriteImage(output, str(labels / f"P{case_id[1:]}.nii.gz"), True)

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "mitral_valve": 1},
        "numTraining": 27 + len(selected_ids),
        "file_ending": ".nii.gz",
        "name": args.output_dataset.name,
        "description": "MVAA Task 1 real labels plus filtered top-200 pseudo-labels",
    }
    (args.output_dataset / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2), encoding="utf-8"
    )
    original = json.loads(args.original_splits.read_text(encoding="utf-8"))[args.fold]
    split = {
        "train": [f"L{case.split('_')[-1]}" for case in original["train"]]
        + [f"P{case[1:]}" for case in sorted(selected_ids)],
        "val": [f"L{case.split('_')[-1]}" for case in original["val"]],
    }
    (args.output_dataset / "splits_final.json").write_text(
        json.dumps([split], indent=2), encoding="utf-8"
    )
    print(f"DATASET={args.output_dataset} TRAIN={len(split['train'])} VAL={len(split['val'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
