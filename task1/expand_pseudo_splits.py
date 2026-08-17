#!/usr/bin/env python3
"""Expand the pseudo-label dataset split file to all original CV folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-splits", type=Path, required=True)
    parser.add_argument("--pseudo-labels", type=Path, required=True)
    parser.add_argument("--targets", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    original = json.loads(args.source_splits.read_text(encoding="utf-8"))
    pseudo = sorted(path.name.removesuffix(".nii.gz") for path in args.pseudo_labels.glob("P*.nii.gz"))
    if len(pseudo) != 200:
        raise RuntimeError(f"Expected 200 pseudo-labels, found {len(pseudo)}")

    def real_case(case: str) -> str:
        return f"L{case.split('_', 1)[1]}" if case.startswith("MVAA_") else case

    expanded = []
    for fold, split in enumerate(original):
        train = [real_case(case) for case in split["train"]] + pseudo
        val = [real_case(case) for case in split["val"]]
        if set(train) & set(val):
            raise RuntimeError(f"Train/validation overlap in fold {fold}")
        expanded.append({"train": train, "val": val})
        print(f"FOLD={fold} TRAIN={len(train)} VAL={len(val)}")
    payload = json.dumps(expanded, indent=2)
    for target in args.targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
        print(f"WROTE={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
