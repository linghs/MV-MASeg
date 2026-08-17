#!/usr/bin/env python3
"""Convert nnU-Net masks to the MVAA task folder format with optional LCC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


def foreground_lcc(segmentation: np.ndarray) -> np.ndarray:
    """Keep the largest 6-connected combined-foreground component."""
    components, count = ndi.label(
        segmentation > 0, structure=ndi.generate_binary_structure(3, 1)
    )
    if count == 0:
        return segmentation.astype(np.uint8, copy=True)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return np.where(
        components == int(sizes.argmax()), segmentation, 0
    ).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=(1, 2), required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--mapping-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-lcc", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    mapping = json.loads(args.mapping_json.read_text(encoding="utf-8"))
    cases = mapping["test"]
    expected_count = 30 if args.task == 1 else 20
    allowed_labels = {0, 1} if args.task == 1 else {0, 1, 2}
    manifest_name = "task1_predictions.json" if args.task == 1 else "task2_predictions.json"
    if len(cases) != expected_count:
        raise RuntimeError(f"Expected {expected_count} test cases, found {len(cases)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for item in cases:
        source = args.prediction_dir / f"{item['nnunet_case']}.nii.gz"
        destination = args.output_dir / item["expected_submission"]
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {destination}")

        image = sitk.ReadImage(str(source))
        segmentation = sitk.GetArrayFromImage(image).astype(np.uint8, copy=False)
        labels = set(int(value) for value in np.unique(segmentation))
        if not labels.issubset(allowed_labels):
            raise ValueError(f"Unexpected labels in {source}: {sorted(labels)}")
        if not args.no_lcc:
            segmentation = foreground_lcc(segmentation)

        result = sitk.GetImageFromArray(segmentation)
        result.CopyInformation(image)
        sitk.WriteImage(result, str(destination), True)
        records.append(
            {"case_id": item["original_case"], "segmentation": destination.name}
        )

    manifest = args.output_dir / manifest_name
    manifest.write_text(
        json.dumps({"cases": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"TASK={args.task} CASES={len(records)} LCC={not args.no_lcc}")
    print(f"OUTPUT={args.output_dir}")
    print(f"MANIFEST={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
