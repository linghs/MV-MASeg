#!/usr/bin/env python3
"""Reuse the supervised Task 1 plans for the pseudo-labeled dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-preprocessed", type=Path, required=True)
    parser.add_argument("--target-preprocessed", type=Path, required=True)
    parser.add_argument("--raw-dataset", type=Path, required=True)
    args = parser.parse_args()

    args.target_preprocessed.mkdir(parents=True, exist_ok=True)
    plans = json.loads(
        (args.source_preprocessed / "nnUNetPlans.json").read_text(encoding="utf-8")
    )
    plans["dataset_name"] = args.raw_dataset.name
    (args.target_preprocessed / "nnUNetPlans.json").write_text(
        json.dumps(plans, indent=2), encoding="utf-8"
    )
    shutil.copy2(
        args.source_preprocessed / "dataset_fingerprint.json",
        args.target_preprocessed / "dataset_fingerprint.json",
    )
    shutil.copy2(args.raw_dataset / "dataset.json", args.target_preprocessed / "dataset.json")
    shutil.copy2(
        args.raw_dataset / "splits_final.json",
        args.target_preprocessed / "splits_final.json",
    )
    print(f"TARGET={args.target_preprocessed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
