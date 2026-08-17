#!/usr/bin/env python3
"""Rank Task 1 unlabeled cases by five-teacher consistency and plausibility."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denominator = int(a.sum() + b.sum())
    return 2 * int(np.logical_and(a, b).sum()) / denominator if denominator else 1.0


def percentile_rank(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.linspace(0, 1, len(values), endpoint=True)
    return ranks if higher_is_better else 1 - ranks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--labeled-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=200)
    args = parser.parse_args()

    summaries = []
    for fold in range(5):
        payload = json.loads(
            (args.root / f"fold_{fold}_summary.json").read_text(encoding="utf-8")
        )
        summaries.append({record["case_id"]: record for record in payload["records"]})
    case_sets = [set(summary) for summary in summaries]
    if any(case_set != case_sets[0] for case_set in case_sets[1:]):
        raise RuntimeError("Teacher summaries contain different case sets")

    labeled_volumes = []
    for path in sorted(args.labeled_dir.glob("*.nii.gz")):
        image = sitk.ReadImage(str(path))
        mask = sitk.GetArrayFromImage(image) > 0
        labeled_volumes.append(float(mask.sum() * np.prod(image.GetSpacing())))
    if len(labeled_volumes) != 27:
        raise RuntimeError(f"Expected 27 labeled masks, found {len(labeled_volumes)}")
    volume_low = float(np.percentile(labeled_volumes, 5) * 0.8)
    volume_high = float(np.percentile(labeled_volumes, 95) * 1.2)

    rows = []
    for case_id in sorted(case_sets[0]):
        records = [summary[case_id] for summary in summaries]
        masks = [sitk.GetArrayFromImage(sitk.ReadImage(record["mask"])) > 0 for record in records]
        if any(mask.shape != masks[0].shape for mask in masks[1:]):
            raise RuntimeError(f"Teacher mask shapes differ for {case_id}")
        pairwise = [dice(masks[a], masks[b]) for a, b in itertools.combinations(range(5), 2)]
        volumes = np.asarray([record["foreground_mm3"] for record in records])
        median_volume = float(np.median(volumes))
        rows.append(
            {
                "case_id": case_id,
                "mean_pairwise_dice": float(np.mean(pairwise)),
                "min_pairwise_dice": float(np.min(pairwise)),
                "volume_mm3_median": median_volume,
                "volume_cv": float(np.std(volumes) / max(np.mean(volumes), 1e-6)),
                "foreground_confidence": float(np.mean([r["foreground_confidence"] for r in records])),
                "foreground_high_conf_fraction": float(np.mean([r["foreground_high_conf_fraction"] for r in records])),
                "boundary_confidence": float(np.mean([r["boundary_confidence"] for r in records])),
                "largest_component_fraction": float(np.mean([r["largest_component_fraction"] for r in records])),
                "plausible_volume": volume_low <= median_volume <= volume_high,
            }
        )

    features = {
        "mean_pairwise_dice": (0.45, True),
        "min_pairwise_dice": (0.15, True),
        "foreground_confidence": (0.10, True),
        "foreground_high_conf_fraction": (0.05, True),
        "boundary_confidence": (0.10, True),
        "largest_component_fraction": (0.05, True),
        "volume_cv": (0.10, False),
    }
    for name, (weight, higher) in features.items():
        ranks = percentile_rank(np.asarray([row[name] for row in rows]), higher)
        for row, rank in zip(rows, ranks):
            row.setdefault("score", 0.0)
            row["score"] += weight * float(rank)

    eligible = [
        row
        for row in rows
        if row["plausible_volume"]
        and row["mean_pairwise_dice"] >= 0.75
        and row["largest_component_fraction"] >= 0.97
    ]
    eligible.sort(key=lambda row: row["score"], reverse=True)
    if len(eligible) < args.top_k:
        raise RuntimeError(f"Only {len(eligible)} eligible cases for top-{args.top_k}")
    selected_ids = {row["case_id"] for row in eligible[: args.top_k]}
    for row in rows:
        row["selected"] = row["case_id"] in selected_ids
    rows.sort(key=lambda row: row["score"], reverse=True)
    payload = {
        "top_k": args.top_k,
        "eligible_count": len(eligible),
        "volume_range_mm3": [volume_low, volume_high],
        "selected": [row for row in rows if row["selected"]],
        "all_cases": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"ELIGIBLE={len(eligible)} SELECTED={len(payload['selected'])}")
    print(f"OUTPUT={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
