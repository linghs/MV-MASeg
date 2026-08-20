from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReleaseSmokeTests(unittest.TestCase):
    def test_configs_match_paper_method(self):
        task1_plan = json.loads((ROOT / "configs/task1_nnunet_plans.json").read_text())
        task2_plan = json.loads((ROOT / "configs/task2_nnunet_plans.json").read_text())
        task1 = task1_plan["configurations"]["3d_fullres"]
        task2 = task2_plan["configurations"]["3d_fullres"]
        self.assertEqual(task1["patch_size"], [112, 128, 160])
        self.assertEqual(task1["spacing"], [0.5, 0.357421875, 0.357421875])
        self.assertEqual(task2["patch_size"], [96, 128, 160])
        self.assertEqual(
            task2["spacing"],
            [0.5395808815956116, 0.2322079986333847, 0.372658371925354],
        )

        for task, expected in (("task1", 27), ("task2", 105)):
            splits = json.loads((ROOT / f"configs/{task}_splits_final.json").read_text())
            self.assertEqual(len(splits), 5)
            validation = [case for fold in splits for case in fold["val"]]
            self.assertEqual(len(validation), expected)
            self.assertEqual(len(set(validation)), expected)

    def test_combined_foreground_lcc_preserves_classes(self):
        finalizer = load_module(
            "finalizer", ROOT / "common/finalize_predictions.py"
        )
        segmentation = np.zeros((8, 8, 8), dtype=np.uint8)
        segmentation[1:4, 1:4, 1:4] = 1
        segmentation[2:4, 2:4, 2:4] = 2
        segmentation[7, 7, 7] = 2
        result = finalizer.foreground_lcc(segmentation)
        self.assertEqual(set(np.unique(result)), {0, 1, 2})
        self.assertEqual(result[7, 7, 7], 0)

    def test_task2_checkpoint_selection(self):
        script = ROOT / "task2/select_checkpoints.py"
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            expected = []
            for fold in range(5):
                fold_dir = model / f"fold_{fold}"
                best = 0.9 if fold % 2 == 0 else 0.7
                final = 0.8
                expected.append("best" if best > final else "final")
                for name, score in (("best", best), ("final", final)):
                    validation = fold_dir / f"validation_{name}"
                    validation.mkdir(parents=True)
                    (validation / "summary.json").write_text(
                        json.dumps({"foreground_mean": {"Dice": score}})
                    )
                    (fold_dir / f"checkpoint_{name}.pth").write_bytes(name.encode())
            subprocess.run(
                [sys.executable, str(script), "--results-root", str(model)],
                check=True,
                capture_output=True,
                text=True,
            )
            for fold, name in enumerate(expected):
                selected = (model / f"fold_{fold}/checkpoint_selected.pth").read_bytes()
                self.assertEqual(selected, name.encode())

    def test_prediction_finalizer(self):
        script = ROOT / "common/finalize_predictions.py"
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            predictions = temporary / "predictions"
            output = temporary / "output"
            predictions.mkdir()
            cases = []
            for index in range(1, 21):
                case = f"MVAA_T2_val_{index:03d}"
                segmentation = np.zeros((6, 6, 6), dtype=np.uint8)
                segmentation[1:4, 1:4, 1:4] = 1
                segmentation[2:4, 2:4, 2:4] = 2
                segmentation[5, 5, 5] = 2
                sitk.WriteImage(
                    sitk.GetImageFromArray(segmentation),
                    str(predictions / f"{case}.nii.gz"),
                )
                original = f"val_{index:03d}"
                cases.append(
                    {
                        "nnunet_case": case,
                        "original_case": original,
                        "expected_submission": f"{original}-pred.nii.gz",
                    }
                )
            mapping = temporary / "mapping.json"
            mapping.write_text(json.dumps({"test": cases}))
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--task",
                    "2",
                    "--prediction-dir",
                    str(predictions),
                    "--mapping-json",
                    str(mapping),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(len(list(output.glob("*.nii.gz"))), 20)
            result = sitk.GetArrayFromImage(
                sitk.ReadImage(str(output / "val_001-pred.nii.gz"))
            )
            self.assertEqual(result[5, 5, 5], 0)
            manifest = json.loads((output / "task2_predictions.json").read_text())
            self.assertEqual(len(manifest["cases"]), 20)


if __name__ == "__main__":
    unittest.main()
