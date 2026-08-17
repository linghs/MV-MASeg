#!/usr/bin/env python3
"""Replace one member of an averaged nnU-Net ensemble in probability space."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def foreground(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        probabilities = archive["probabilities"]
    if probabilities.ndim != 4 or probabilities.shape[0] != 2:
        raise ValueError(f"Unexpected probability shape in {path}: {probabilities.shape}")
    return probabilities[1].astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--old-fold-dir", type=Path, required=True)
    parser.add_argument("--new-fold-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-weight", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--expected-cases", type=int, default=30)
    args = parser.parse_args()

    ensemble_files = sorted(args.ensemble_dir.glob("MVAA_*.npz"))
    if len(ensemble_files) != args.expected_cases:
        raise RuntimeError(f"Expected {args.expected_cases} cases, found {len(ensemble_files)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ensemble_path in ensemble_files:
        old_path = args.old_fold_dir / ensemble_path.name
        new_path = args.new_fold_dir / ensemble_path.name
        ensemble, old, new = foreground(ensemble_path), foreground(old_path), foreground(new_path)
        if ensemble.shape != old.shape or ensemble.shape != new.shape:
            raise RuntimeError(f"Shape mismatch for {ensemble_path.stem}")
        replaced = np.clip(ensemble + args.fold_weight * (new - old), 0.0, 1.0)
        mask = (replaced >= args.threshold).astype(np.uint8)
        reference = sitk.ReadImage(str(ensemble_path.with_suffix(".nii.gz")))
        if tuple(reversed(reference.GetSize())) != mask.shape:
            raise RuntimeError(f"Reference shape mismatch for {ensemble_path.stem}")
        output = sitk.GetImageFromArray(mask)
        output.CopyInformation(reference)
        sitk.WriteImage(output, str(args.output_dir / f"{ensemble_path.stem}.nii.gz"), True)
    print(f"CASES={len(ensemble_files)} OUTPUT={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
