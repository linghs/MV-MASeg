#!/usr/bin/env python3
"""Select the higher-Dice final/best checkpoint independently for each fold."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report_path = args.report or args.results_root / "checkpoint_selection.json"

    selections = []
    for fold in range(5):
        fold_dir = args.results_root / f"fold_{fold}"
        scores = {}
        for checkpoint in ("final", "best"):
            summary_path = fold_dir / f"validation_{checkpoint}" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            scores[checkpoint] = float(summary["foreground_mean"]["Dice"])
        selected = "final" if scores["final"] >= scores["best"] else "best"
        source = fold_dir / f"checkpoint_{selected}.pth"
        destination = fold_dir / "checkpoint_selected.pth"
        shutil.copy2(source, destination)
        selections.append(
            {
                "fold": fold,
                "final_dice": scores["final"],
                "best_dice": scores["best"],
                "selected": selected,
            }
        )

    report = {
        "selection_rule": "higher foreground-mean Dice; ties prefer final",
        "folds": selections,
        "mean_selected_dice": sum(
            max(row["final_dice"], row["best_dice"]) for row in selections
        )
        / 5,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
