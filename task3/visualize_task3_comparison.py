#!/usr/bin/env python3
"""Create publication-ready qualitative comparisons for Task 3 segmentation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from dataset import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    Sample,
    discover_samples,
    read_binary_mask_from_label_tar,
    split_train_val_by_video,
)
from model_factory import get_model  # noqa: E402

DEFAULT_RUNS_DIR = THIS_DIR / "runs"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "t3_vid" / "train"
DEFAULT_OUTPUT = THIS_DIR / "figures" / "task3_qualitative_comparison.png"


@dataclass
class ModelSpec:
    title: str
    checkpoint: Path
    requested_encoder: str
    fallback_note: str = ""


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: torch.nn.Module
    args: Dict[str, Any]
    threshold: float


@dataclass
class CaseResult:
    sample: Sample
    image: np.ndarray
    gt: np.ndarray
    predictions: List[np.ndarray]
    dices: List[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-tta", action="store_true", help="Disable flip TTA.")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--seed", type=int, default=None, help="Override checkpoint split seed.")
    parser.add_argument("--val-video-count", type=int, default=None)
    parser.add_argument("--case", action="append", default=[], metavar="VIDEO_ID:FRAME_IDX",
                        help="Select a case explicitly; may be supplied repeatedly.")
    parser.add_argument("--selection", choices=("representative", "improvement", "hard", "spread"),
                        default="representative")
    parser.add_argument("--max-eval-cases", type=int, default=0,
                        help="Limit evaluated foreground validation cases; 0 means all.")
    parser.add_argument("--overlay-mode", choices=("mask", "error"), default="mask",
                        help="mask reproduces the example; error shows TP/FP/FN.")
    return parser.parse_args()


def checkpoint_args(checkpoint: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object: {checkpoint}")
    args = obj.get("args", {})
    return obj, args if isinstance(args, dict) else {}


def checkpoint_candidates(runs_dir: Path) -> Iterable[Path]:
    preferred_names = ("bestmodel.pth", "best.pt", "best.pth")
    found: List[Path] = []
    for name in preferred_names:
        found.extend(runs_dir.rglob(name))
    return sorted(set(found))


def find_checkpoint(
    candidates: Sequence[Path], arch: str, encoder_names: Sequence[str], preferred_runs: Sequence[str] = ()
) -> Tuple[Path, Dict[str, Any], bool]:
    matches: List[Tuple[int, Path, Dict[str, Any]]] = []
    for path in candidates:
        try:
            _, cfg = checkpoint_args(path)
        except Exception:
            continue
        if str(cfg.get("arch", "")).lower() != arch.lower():
            continue
        encoder = str(cfg.get("encoder_name", "")).lower()
        if encoder not in {name.lower() for name in encoder_names}:
            continue
        score = 0
        for rank, run_name in enumerate(preferred_runs):
            if run_name.lower() in path.as_posix().lower():
                score += 100 - rank
        if "bestmodel" in path.name:
            score += 2
        matches.append((score, path, cfg))
    if not matches:
        raise FileNotFoundError(f"No checkpoint for arch={arch}, encoders={list(encoder_names)}")
    matches.sort(key=lambda item: (-item[0], str(item[1])))
    selected = matches[0]
    exact = str(selected[2].get("encoder_name", "")).lower() == encoder_names[0].lower()
    return selected[1], selected[2], exact


def resolve_specs(runs_dir: Path) -> List[ModelSpec]:
    candidates = list(checkpoint_candidates(runs_dir))
    if not candidates:
        raise FileNotFoundError(f"No best checkpoints found under {runs_dir}")

    ours_path, _, _ = find_checkpoint(
        candidates, "dinov2_unet", ("efficientnet-b4",), ("task3_search_v07_conservative_semi",)
    )
    b4_path, _, _ = find_checkpoint(
        candidates, "unetplusplus", ("efficientnet-b4", "tu-efficientnet_b4"), ("task3_baseline",)
    )
    b3_path, b3_cfg, _ = find_checkpoint(
        candidates, "unetplusplus", ("tu-efficientnet_b3", "timm-efficientnet-b3", "efficientnet-b3"),
        ("task3_effb3_img",)
    )

    try:
        b5_path, b5_cfg, exact_b5 = find_checkpoint(
            candidates, "unetplusplus",
            ("tu-efficientnet_b5", "timm-efficientnet-b5", "efficientnet-b5"),
        )
    except FileNotFoundError:
        b5_path, b5_cfg, exact_b5 = find_checkpoint(
            candidates, "unetplusplus", ("efficientnet-b4", "tu-efficientnet_b4"),
            ("task3_pretrain", "task3_baseline"),
        )

    b5_actual = str(b5_cfg.get("encoder_name", "unknown"))
    b3_actual = str(b3_cfg.get("encoder_name", "unknown"))
    return [
        ModelSpec("DINOv2-UNet (Ours)", ours_path, "dinov2_vitl14"),
        ModelSpec("UNet++ + EfficientNet-B4", b4_path, "efficientnet-b4"),
        ModelSpec(
            "UNet++ + Timm-EfficientNet-B5" if exact_b5 else f"UNet++ + {b5_actual} (B5 fallback)",
            b5_path, "timm-efficientnet-b5",
            "Exact B5 checkpoint not found; nearest available UNet++ EfficientNet checkpoint used." if not exact_b5 else "",
        ),
        ModelSpec("UNet++ + Timm-EfficientNet-B3", b3_path, "timm-efficientnet-b3",
                  "" if "b3" in b3_actual.lower() else f"Using available encoder {b3_actual}."),
    ]


def build_model_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    none_value = lambda x: None if x is None or str(x).strip().lower() in {"", "none", "null"} else x
    return {
        "arch": str(cfg.get("arch", "unetplusplus")),
        "encoder_name": str(cfg.get("encoder_name", "resnet34")),
        "encoder_weights": none_value(cfg.get("encoder_weights")),
        "in_channels": 3,
        "classes": 1,
        "dinov2_name": str(cfg.get("dinov2_name", "dinov2_vitl14")),
        "dinov2_pretrained": False,
        "dinov2_decoder_channels": int(cfg.get("dinov2_decoder_channels", 256)),
        "freeze_dinov2": False,
        "dinov2_checkpoint": None,
        "dinov2_lora_rank": int(cfg.get("dinov2_lora_rank", 0)),
        "dinov2_lora_alpha": float(cfg.get("dinov2_lora_alpha", 16.0)),
        "dinov2_lora_dropout": float(cfg.get("dinov2_lora_dropout", 0.0)),
        "dinov2_lora_target": str(cfg.get("dinov2_lora_target", "qkv")),
        "dinov2_train_lora_only": bool(cfg.get("dinov2_train_lora_only", False)),
        "dinov3_name": str(cfg.get("dinov3_name", "dinov3_vitl16")),
        "dinov3_pretrained": False,
        "dinov3_decoder_channels": int(cfg.get("dinov3_decoder_channels", 256)),
        "freeze_dinov3": False,
        "dinov3_checkpoint": None,
        "dinov3_lora_rank": int(cfg.get("dinov3_lora_rank", 8)),
        "dinov3_lora_alpha": float(cfg.get("dinov3_lora_alpha", 16.0)),
        "dinov3_lora_dropout": float(cfg.get("dinov3_lora_dropout", 0.0)),
        "dinov3_lora_target": str(cfg.get("dinov3_lora_target", "qkv")),
        "dinov3_train_lora_only": bool(cfg.get("dinov3_train_lora_only", True)),
        "nnunet_model_size": str(cfg.get("nnunet_model_size", "base")),
        "nnunet_deep_supervision": bool(cfg.get("nnunet_deep_supervision", False)),
        "mae_encoder_ckpt": none_value(cfg.get("mae_encoder_ckpt")),
    }


def state_dict_from_checkpoint(obj: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ("model_state", "model_state_dict", "state_dict"):
        if key in obj:
            return obj[key]
    return obj


def load_models(specs: Sequence[ModelSpec], device: torch.device) -> List[LoadedModel]:
    loaded: List[LoadedModel] = []
    for spec in specs:
        obj, cfg = checkpoint_args(spec.checkpoint)
        model = get_model(**build_model_kwargs(cfg)).to(device)
        model.load_state_dict(state_dict_from_checkpoint(obj), strict=True)
        model.eval()
        threshold = float(obj.get("val_metrics", {}).get("val_threshold", obj.get("best_threshold", 0.5)))
        loaded.append(LoadedModel(spec, model, cfg, threshold))
        print(f"[model] {spec.title}: {spec.checkpoint} (threshold={threshold:.3f})")
        if spec.fallback_note:
            print(f"[fallback] {spec.fallback_note}")
    return loaded


def read_mask(sample: Sample, target_label: int) -> np.ndarray:
    if sample.label_kind == "bin_png":
        return (np.asarray(Image.open(sample.label_path).convert("L")) > 127).astype(np.uint8)
    return read_binary_mask_from_label_tar(sample.label_path, target_label).astype(np.uint8)


def make_input(image: np.ndarray, cfg: Dict[str, Any], device: torch.device) -> torch.Tensor:
    size = tuple(int(x) for x in cfg.get("image_size", (448, 800)))
    x = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    if bool(cfg.get("use_imagenet_norm", True)):
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        x = (x - mean) / std
    return x


@torch.inference_mode()
def predict(loaded: LoadedModel, image: np.ndarray, device: torch.device, tta: bool) -> np.ndarray:
    x = make_input(image, loaded.args, device)
    amp = device.type == "cuda"
    flips: Sequence[Tuple[int, ...]] = ((), (3,), (2,), (2, 3)) if tta else ((),)
    total = None
    for dims in flips:
        inp = torch.flip(x, dims=dims) if dims else x
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            prob = torch.sigmoid(loaded.model(inp))
        if dims:
            prob = torch.flip(prob, dims=dims)
        total = prob if total is None else total + prob
    prob = total / len(flips)
    prob = F.interpolate(prob, size=image.shape[:2], mode="bilinear", align_corners=False)
    return (prob[0, 0].float().cpu().numpy() > loaded.threshold).astype(np.uint8)


def dice(gt: np.ndarray, pred: np.ndarray) -> float:
    denom = float(gt.sum() + pred.sum())
    return 1.0 if denom == 0 else float(2.0 * np.logical_and(gt, pred).sum() / denom)


def parse_cases(values: Sequence[str], samples: Sequence[Sample]) -> List[Sample]:
    lookup = {(s.video_id, s.frame_idx): s for s in samples}
    selected = []
    for value in values:
        video, sep, frame = value.rpartition(":")
        if not sep:
            raise ValueError(f"Invalid --case '{value}'; expected VIDEO_ID:FRAME_IDX")
        key = (video, int(frame))
        if key not in lookup:
            raise KeyError(f"Validation case not found: {value}")
        selected.append(lookup[key])
    return selected


def choose_results(results: Sequence[CaseResult], count: int, mode: str) -> List[CaseResult]:
    if count >= len(results):
        return list(results)
    if mode == "improvement":
        ranked = sorted(results, key=lambda r: r.dices[0] - max(r.dices[1:]), reverse=True)
        return ranked[:count]
    if mode == "hard":
        return sorted(results, key=lambda r: r.dices[0])[:count]
    if mode == "spread":
        ordered = sorted(results, key=lambda r: (r.sample.video_id, r.sample.frame_idx))
        indices = np.linspace(0, len(ordered) - 1, count).round().astype(int)
        return [ordered[i] for i in indices]

    ours = np.asarray([r.dices[0] for r in results])
    quantiles = np.linspace(0.2, 0.8, count)
    chosen: List[CaseResult] = []
    for target in np.quantile(ours, quantiles):
        available = [r for r in results if r not in chosen]
        available.sort(key=lambda r: (abs(r.dices[0] - target), -int(r.gt.sum())))
        different_video = [r for r in available if r.sample.video_id not in {x.sample.video_id for x in chosen}]
        chosen.append((different_video or available)[0])
    return chosen


def rgba_mask(mask: np.ndarray, color: Tuple[float, float, float], alpha: float) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., :3] = color
    rgba[..., 3] = mask.astype(np.float32) * alpha
    return rgba


def draw_mask(ax: plt.Axes, image: np.ndarray, mask: np.ndarray) -> None:
    blue = (0.05, 0.45, 0.95)
    ax.imshow(image)
    ax.imshow(rgba_mask(mask, blue, 0.48))
    if mask.any():
        ax.contour(mask, levels=[0.5], colors=[blue], linewidths=0.9)


def draw_error(ax: plt.Axes, image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> None:
    tp = np.logical_and(gt, pred)
    fp = np.logical_and(~gt.astype(bool), pred)
    fn = np.logical_and(gt, ~pred.astype(bool))
    ax.imshow(image)
    ax.imshow(rgba_mask(tp, (0.12, 0.75, 0.25), 0.48))
    ax.imshow(rgba_mask(fp, (1.00, 0.55, 0.05), 0.62))
    ax.imshow(rgba_mask(fn, (0.05, 0.65, 0.95), 0.62))


def save_figure(results: Sequence[CaseResult], models: Sequence[LoadedModel], output: Path,
                dpi: int, overlay_mode: str) -> None:
    titles = ["Input", "Ground Truth"] + [m.spec.title for m in models]
    rows, cols = len(results), len(titles)
    fig, axes = plt.subplots(rows, cols, figsize=(2.7 * cols, 2.45 * rows), squeeze=False)
    for row, result in enumerate(results):
        axes[row, 0].imshow(result.image)
        axes[row, 1].imshow(result.image)
        draw_mask(axes[row, 1], result.image, result.gt)
        for idx, pred in enumerate(result.predictions):
            ax = axes[row, idx + 2]
            if overlay_mode == "error":
                draw_error(ax, result.image, result.gt, pred)
            else:
                draw_mask(ax, result.image, pred)
        axes[row, 0].set_ylabel(
            f"{result.sample.video_id}\nFrame {result.sample.frame_idx:06d}",
            fontsize=8, fontweight="bold", rotation=90, labelpad=10,
        )
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=9.5, fontweight="bold", pad=7)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.93, bottom=0.02, wspace=0.035, hspace=0.09)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_manifest(results: Sequence[CaseResult], models: Sequence[LoadedModel], output: Path) -> None:
    payload = {
        "models": [{"title": m.spec.title, "checkpoint": str(m.spec.checkpoint),
                    "threshold": m.threshold, "fallback_note": m.spec.fallback_note} for m in models],
        "cases": [{"video_id": r.sample.video_id, "frame_idx": r.sample.frame_idx,
                   "image": str(r.sample.image_path), "dice": dict(zip([m.spec.title for m in models], r.dices))}
                  for r in results],
    }
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.num_cases < 1:
        raise ValueError("--num-cases must be positive")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    specs = resolve_specs(args.runs_dir)
    models = load_models(specs, device)
    primary_cfg = models[0].args
    all_samples = discover_samples(args.data_root)
    seed = int(primary_cfg.get("seed", 42) if args.seed is None else args.seed)
    val_count = int(primary_cfg.get("val_video_count", 2) if args.val_video_count is None else args.val_video_count)
    _, validation, _, val_videos = split_train_val_by_video(all_samples, val_count, seed)
    target_label = int(primary_cfg.get("target_label", 10))
    validation = [s for s in validation if read_mask(s, target_label).any()]
    if args.case:
        validation = parse_cases(args.case, validation)
    elif args.max_eval_cases > 0:
        step = max(1, math.floor(len(validation) / args.max_eval_cases))
        validation = validation[::step][:args.max_eval_cases]
    print(f"[data] validation videos={val_videos}; foreground cases={len(validation)}")

    results: List[CaseResult] = []
    for index, sample in enumerate(validation, 1):
        image = np.asarray(Image.open(sample.image_path).convert("RGB"), dtype=np.uint8)
        gt = read_mask(sample, target_label)
        predictions = [predict(model, image, device, not args.no_tta) for model in models]
        results.append(CaseResult(sample, image, gt, predictions, [dice(gt, p) for p in predictions]))
        print(f"[infer] {index}/{len(validation)} {sample.video_id}:{sample.frame_idx:06d}")
    if not results:
        raise RuntimeError("No validation cases available for visualization")
    selected = results if args.case else choose_results(results, args.num_cases, args.selection)
    save_figure(selected, models, args.output, args.dpi, args.overlay_mode)
    save_manifest(selected, models, args.output)
    print(f"[saved] {args.output}")
    print(f"[saved] {args.output.with_suffix('.pdf')}")
    print(f"[saved] {args.output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
