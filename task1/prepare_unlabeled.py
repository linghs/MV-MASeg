#!/usr/bin/env python3
"""Create an nnU-Net inference view of Task 1 unlabeled CT volumes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path)
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    args = parser.parse_args()

    selected = None
    if args.selection_json:
        payload = json.loads(args.selection_json.read_text(encoding="utf-8"))
        selected = {row["case_id"] for row in payload["selected"]}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sorted(args.source_dir.glob("*.nii.gz")):
        numeric = source.name.removesuffix(".nii.gz")
        case = f"U{int(numeric):04d}"
        if selected is not None and case not in selected:
            continue
        destination = args.output_dir / f"{case}_0000.nii.gz"
        if not destination.exists():
            if args.copy:
                shutil.copy2(source, destination)
            else:
                os.symlink(source.resolve(), destination)
        records.append({"case_id": case, "image": str(source.resolve())})

    expected = len(selected) if selected is not None else 1040
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} cases, found {len(records)}")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps({"records": records}, indent=2), encoding="utf-8"
    )
    print(f"CASES={len(records)} OUTPUT={args.output_dir} MANIFEST={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
