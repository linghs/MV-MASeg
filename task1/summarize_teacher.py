#!/usr/bin/env python3
"""Summarize one Task 1 teacher's probabilities for pseudo-label ranking."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=1040)
    parser.add_argument("--delete-probabilities", action="store_true")
    args = parser.parse_args()

    prediction_dir = args.prediction_dir.resolve()
    probability_files = sorted(prediction_dir.glob("U*.npz"))
    if len(probability_files) != args.expected_cases:
        raise RuntimeError(
            f"Expected {args.expected_cases} probabilities, found {len(probability_files)}"
        )

    structure = ndi.generate_binary_structure(3, 1)
    records = []
    for index, probability_path in enumerate(probability_files, 1):
        with np.load(probability_path) as archive:
            probabilities = archive["probabilities"]
        if probabilities.ndim != 4 or probabilities.shape[0] != 2:
            raise ValueError(f"Unexpected shape for {probability_path}: {probabilities.shape}")
        foreground = probabilities[1].astype(np.float32, copy=False)
        mask = foreground >= 0.5
        mask_path = probability_path.with_suffix(".nii.gz")
        image = sitk.ReadImage(str(mask_path))
        if tuple(reversed(image.GetSize())) != mask.shape:
            raise RuntimeError(f"Mask/probability mismatch: {probability_path.stem}")

        components, count = ndi.label(mask, structure=structure)
        sizes = np.bincount(components.ravel()) if count else np.array([0])
        largest = int(sizes[1:].max()) if count else 0
        voxels = int(mask.sum())
        band = (foreground >= 0.1) & (foreground <= 0.9)
        fg_values = foreground[mask]
        boundary_values = foreground[band]
        eps = 1e-7
        entropy = -(
            foreground * np.log(foreground + eps)
            + (1 - foreground) * np.log(1 - foreground + eps)
        ) / math.log(2)
        records.append(
            {
                "case_id": probability_path.stem,
                "mask": str(mask_path),
                "foreground_voxels": voxels,
                "foreground_mm3": float(voxels * np.prod(image.GetSpacing())),
                "foreground_confidence": float(fg_values.mean()) if fg_values.size else 0.0,
                "foreground_high_conf_fraction": float((fg_values >= 0.9).mean()) if fg_values.size else 0.0,
                "boundary_confidence": float((2 * np.abs(boundary_values - 0.5)).mean()) if boundary_values.size else 1.0,
                "boundary_entropy": float(entropy[band].mean()) if band.any() else 0.0,
                "component_count": int(count),
                "largest_component_fraction": float(largest / voxels) if voxels else 0.0,
            }
        )
        if index % 100 == 0:
            print(f"summarized={index}/{len(probability_files)}", flush=True)

    payload = {"count": len(records), "prediction_dir": str(prediction_dir), "records": records}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.delete_probabilities:
        for probability_path in probability_files:
            if probability_path.resolve().parent != prediction_dir:
                raise RuntimeError(f"Unsafe probability path: {probability_path}")
            probability_path.unlink()
            pickle_path = probability_path.with_suffix(".pkl")
            if pickle_path.exists():
                pickle_path.unlink()
    print(f"SUMMARY={args.output_json} CASES={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
