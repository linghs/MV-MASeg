#!/usr/bin/env python3
"""Model and loss factory for task3 2D segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
except Exception as e:  # pragma: no cover
    smp = None
    _SMP_IMPORT_ERROR = e
else:
    _SMP_IMPORT_ERROR = None


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class ConvINLeaky(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )


class ResidualBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvINLeaky(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
        )
        self.skip = nn.Identity()
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_ch, affine=True),
            )
        self.act = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class ResidualStage2D(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, blocks: int, stride: int) -> None:
        layers = [ResidualBlock2D(in_ch, out_ch, stride=stride)]
        layers.extend(ResidualBlock2D(out_ch, out_ch, stride=1) for _ in range(max(0, blocks - 1)))
        super().__init__(*layers)


class NnUNetV2ResidualEncoder2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        features: tuple[int, ...] = (32, 64, 128, 256, 320),
        blocks_per_stage: tuple[int, ...] = (1, 2, 3, 4, 4),
        strides: tuple[int, ...] = (1, 2, 2, 2, 2),
    ) -> None:
        super().__init__()
        if not (len(features) == len(blocks_per_stage) == len(strides)):
            raise ValueError("features, blocks_per_stage and strides must have the same length")
        stages = []
        cur_ch = in_channels
        for out_ch, blocks, stride in zip(features, blocks_per_stage, strides):
            stages.append(ResidualStage2D(cur_ch, int(out_ch), int(blocks), int(stride)))
            cur_ch = int(out_ch)
        self.stages = nn.ModuleList(stages)
        self.out_channels = tuple(int(x) for x in features)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class NnUNetV2ResEncUNet2D(nn.Module):
    """2D nnU-Net v2 residual-encoder UNet with optional MAE encoder initialization."""

    def __init__(
        self,
        in_channels: int = 3,
        classes: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256, 320),
        blocks_per_stage: tuple[int, ...] = (1, 2, 3, 4, 4),
        strides: tuple[int, ...] = (1, 2, 2, 2, 2),
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        self.deep_supervision = bool(deep_supervision)
        self.encoder = NnUNetV2ResidualEncoder2D(in_channels, features, blocks_per_stage, strides)
        rev_features = list(reversed(self.encoder.out_channels))
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.seg_heads = nn.ModuleList()
        cur_ch = rev_features[0]
        for skip_ch in rev_features[1:]:
            self.upconvs.append(nn.ConvTranspose2d(cur_ch, skip_ch, kernel_size=2, stride=2))
            self.decoders.append(ResidualBlock2D(skip_ch * 2, skip_ch, stride=1))
            self.seg_heads.append(nn.Conv2d(skip_ch, classes, kernel_size=1))
            cur_ch = skip_ch
        self.final_head = nn.Conv2d(cur_ch, classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        input_size = x.shape[-2:]
        skips = self.encoder(x)
        x = skips[-1]
        aux = []
        for up, dec, head, skip in zip(self.upconvs, self.decoders, self.seg_heads, reversed(skips[:-1])):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
            if self.training and self.deep_supervision:
                aux.append(F.interpolate(head(x), size=input_size, mode="bilinear", align_corners=False))
        logits = self.final_head(x)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        if self.training and self.deep_supervision:
            return logits, aux
        return logits


def _load_encoder_weights(model: nn.Module, checkpoint_path: str, strict: bool = False) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state = checkpoint.get("encoder_state_dict") or checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    else:
        state = checkpoint
    cleaned = {}
    for key, value in state.items():
        clean_key = str(key)
        for prefix in ("module.", "model.", "student.", "teacher.", "encoder."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        cleaned[clean_key] = value
    missing, unexpected = model.encoder.load_state_dict(cleaned, strict=strict)
    print(
        f"Loaded MAE encoder checkpoint from {checkpoint_path} "
        f"(missing_keys={len(missing)}, unexpected_keys={len(unexpected)})"
    )


def build_nnunetv2_resenc_model_2d(
    in_channels: int = 3,
    classes: int = 1,
    model_size: str = "base",
    deep_supervision: bool = False,
    pretrained_encoder: str | None = None,
) -> NnUNetV2ResEncUNet2D:
    model_size = str(model_size).lower()
    if model_size == "small":
        features = (24, 48, 96, 192)
        blocks = (1, 2, 2, 2)
        strides = (1, 2, 2, 2)
    elif model_size == "large":
        features = (32, 64, 128, 256, 320, 320)
        blocks = (1, 3, 4, 6, 6, 6)
        strides = (1, 2, 2, 2, 2, 2)
    else:
        features = (32, 64, 128, 256, 320)
        blocks = (1, 2, 3, 4, 4)
        strides = (1, 2, 2, 2, 2)
    model = NnUNetV2ResEncUNet2D(
        in_channels=in_channels,
        classes=classes,
        features=features,
        blocks_per_stage=blocks,
        strides=strides,
        deep_supervision=deep_supervision,
    )
    if pretrained_encoder:
        _load_encoder_weights(model, pretrained_encoder, strict=False)
    return model


class LoRALinear(nn.Module):
    """LoRA adapter for an existing nn.Linear layer."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.linear = linear
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(linear.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class DINOSegmentationModel(nn.Module):
    """DINO ViT encoder with LoRA adapters and a lightweight CNN decoder."""

    def __init__(
        self,
        model_name: str = "vit_large_patch16_dinov3",
        pretrained: bool = True,
        in_channels: int = 3,
        classes: int = 1,
        decoder_channels: int = 256,
        freeze_encoder: bool = False,
        checkpoint_path: str | None = None,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target: str = "qkv",
        train_lora_only: bool = False,
    ) -> None:
        super().__init__()
        try:
            import timm
        except Exception as e:  # pragma: no cover
            raise ImportError("timm is required for DINO models") from e

        if in_channels != 3:
            raise ValueError("DINO encoder currently expects 3 input channels")

        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=None,
            dynamic_img_size=True,
        )
        if checkpoint_path:
            self._load_encoder_checkpoint(checkpoint_path)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_target = str(lora_target).lower()
        self.lora_layers = self._inject_lora(
            self.encoder,
            self.lora_rank,
            self.lora_alpha,
            self.lora_dropout,
            self.lora_target,
        )
        self.patch_size = self._infer_patch_size(self.encoder)
        embed_dim = int(getattr(self.encoder, "num_features", 0))
        if embed_dim <= 0:
            raise ValueError(f"Could not infer DINO embed dim for {model_name}")

        self.proj = ConvBNAct(embed_dim, decoder_channels, kernel_size=1)
        self.low_stem = nn.Sequential(
            ConvBNAct(in_channels, 64),
            ConvBNAct(64, 64),
        )
        self.decode = nn.Sequential(
            ConvBNAct(decoder_channels + 64, decoder_channels),
            ConvBNAct(decoder_channels, decoder_channels // 2),
            nn.Conv2d(decoder_channels // 2, classes, kernel_size=1),
        )
        if train_lora_only and self.lora_rank <= 0:
            raise ValueError("train_lora_only=True requires lora_rank > 0")
        if train_lora_only:
            self.freeze_encoder(train_lora=True)
        elif freeze_encoder:
            self.freeze_encoder()

    @staticmethod
    def _infer_patch_size(encoder: nn.Module) -> int:
        patch_embed = getattr(encoder, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", 14)
        if isinstance(patch_size, tuple):
            return int(patch_size[0])
        return int(patch_size)

    def _load_encoder_checkpoint(self, checkpoint_path: str) -> None:
        path = str(checkpoint_path)
        if path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except Exception as e:  # pragma: no cover
                raise ImportError("safetensors is required to load .safetensors DINO checkpoints") from e
            state_dict = load_file(path, device="cpu")
        else:
            checkpoint = torch.load(path, map_location="cpu")
            if isinstance(checkpoint, dict):
                state_dict = (
                    checkpoint.get("model")
                    or checkpoint.get("state_dict")
                    or checkpoint.get("teacher")
                    or checkpoint.get("student")
                    or checkpoint
                )
            else:
                state_dict = checkpoint

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            clean_key = str(key)
            for prefix in ("module.", "backbone.", "encoder.", "model."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
            cleaned_state_dict[clean_key] = value

        missing, unexpected = self.encoder.load_state_dict(cleaned_state_dict, strict=False)
        print(
            f"Loaded DINO checkpoint from {path} "
            f"(missing_keys={len(missing)}, unexpected_keys={len(unexpected)})"
        )

    @staticmethod
    def _is_lora_target(module_name: str, child_name: str, target: str) -> bool:
        if target not in {"qkv", "qkv_ffn"}:
            raise ValueError(f"Unsupported LoRA target: {target}. Use 'qkv' or 'qkv_ffn'.")
        if child_name == "qkv":
            return True
        if target == "qkv_ffn" and child_name in {"fc1", "fc2"}:
            module_parts = module_name.split(".")
            return "mlp" in module_parts
        return False

    @classmethod
    def _inject_lora(cls, encoder: nn.Module, rank: int, alpha: float, dropout: float, target: str = "qkv") -> list[str]:
        if rank <= 0:
            return []

        lora_layers = []
        for module_name, module in encoder.named_modules():
            for child_name, child in list(module.named_children()):
                if not isinstance(child, nn.Linear) or not cls._is_lora_target(module_name, child_name, target):
                    continue
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                full_name = f"{module_name}.{child_name}" if module_name else child_name
                lora_layers.append(full_name)

        if not lora_layers:
            raise ValueError(f"No DINO Linear layers found for LoRA target: {target}")
        print(f"Injected LoRA into {len(lora_layers)} DINO layers (target={target})")
        return lora_layers

    def freeze_encoder(self, train_lora: bool = False) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        if train_lora:
            for module in self.encoder.modules():
                if isinstance(module, LoRALinear):
                    for p in module.lora_A.parameters():
                        p.requires_grad_(True)
                    for p in module.lora_B.parameters():
                        p.requires_grad_(True)

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(True)

    def get_param_groups(self, lr: float, encoder_lr: float | None = None, weight_decay: float = 1e-5):
        encoder_params = list(self.encoder.parameters())
        decoder_params = [p for n, p in self.named_parameters() if not n.startswith("encoder.")]
        groups = []
        if encoder_params:
            groups.append({"params": encoder_params, "lr": float(lr if encoder_lr is None else encoder_lr), "weight_decay": weight_decay})
        if decoder_params:
            groups.append({"params": decoder_params, "lr": float(lr), "weight_decay": weight_decay})
        return groups

    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, "forward_features"):
            features = self.encoder.forward_features(x)
        else:  # pragma: no cover
            features = self.encoder(x)
        if isinstance(features, dict):
            tokens = features.get("x_norm_patchtokens")
            if tokens is None:
                tokens = features.get("x")
        else:
            tokens = features
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected DINO tokens with shape B,N,C, got {tuple(tokens.shape)}")
        expected_tokens = (x.shape[-2] // self.patch_size) * (x.shape[-1] // self.patch_size)
        if tokens.shape[1] == expected_tokens + 1:
            tokens = tokens[:, 1:, :]
        elif tokens.shape[1] != expected_tokens:
            tokens = tokens[:, -expected_tokens:, :]
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_hw = x.shape[-2:]
        pad_h = (self.patch_size - orig_hw[0] % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - orig_hw[1] % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_in = x

        tokens = self._forward_tokens(x_in)
        grid_h = x_in.shape[-2] // self.patch_size
        grid_w = x_in.shape[-1] // self.patch_size
        feat = tokens.transpose(1, 2).reshape(x.shape[0], -1, grid_h, grid_w)
        feat = self.proj(feat)
        feat = F.interpolate(feat, size=x_in.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_stem(x_in)
        logits = self.decode(torch.cat([feat, low], dim=1))
        if pad_h or pad_w:
            logits = logits[..., : orig_hw[0], : orig_hw[1]]
        return logits


DINO_MODEL_ALIASES = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2_vitg14": "vit_giant_patch14_dinov2.lvd142m",
    "dinov3_vits16": "vit_small_patch16_dinov3",
    "dinov3_vitb16": "vit_base_patch16_dinov3",
    "dinov3_vitl16": "vit_large_patch16_dinov3",
    "dinov3_vith16": "vit_huge_plus_patch16_dinov3",
}

# Backward-compatible class/alias names.
DINOv2SegmentationModel = DINOSegmentationModel
DINOV2_MODEL_ALIASES = DINO_MODEL_ALIASES


def get_model(
    arch: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: str | None = None,
    in_channels: int = 3,
    classes: int = 1,
    dinov2_name: str = "dinov2_vitl14",
    dinov2_pretrained: bool = True,
    dinov2_decoder_channels: int = 256,
    freeze_dinov2: bool = False,
    dinov2_checkpoint: str | None = None,
    dinov2_lora_alpha: float = 16.0,
    dinov2_lora_dropout: float = 0.0,
    dinov2_lora_rank: int = 8, #loRA配置
    dinov2_lora_target: str = "qkv_ffn",
    dinov2_train_lora_only: bool = True,
    dinov3_name: str = "dinov3_vitl16",
    dinov3_pretrained: bool | None = None,
    dinov3_decoder_channels: int | None = None,
    freeze_dinov3: bool | None = None,
    dinov3_checkpoint: str | None = None,
    dinov3_lora_alpha: float | None = None,
    dinov3_lora_dropout: float | None = None,
    dinov3_lora_rank: int | None = None,
    dinov3_lora_target: str | None = None,
    dinov3_train_lora_only: bool | None = None,
    nnunet_model_size: str = "base",
    nnunet_deep_supervision: bool = False,
    mae_encoder_ckpt: str | None = None,
):
    arch = arch.lower()
    if arch in {"nnunet", "nnunetv2", "nnunetv2_resenc", "resenc_nnunet", "nnunet_resenc"}:
        return build_nnunetv2_resenc_model_2d(
            in_channels=in_channels,
            classes=classes,
            model_size=nnunet_model_size,
            deep_supervision=nnunet_deep_supervision,
            pretrained_encoder=mae_encoder_ckpt,
        )

    if arch in {"dinov2_unet", "dinov3_unet"}:
        use_dinov3 = arch == "dinov3_unet"
        if use_dinov3:
            dino_name = dinov3_name
            pretrained = True if dinov3_pretrained is None else bool(dinov3_pretrained)
            decoder_channels = dinov3_decoder_channels if dinov3_decoder_channels is not None else dinov2_decoder_channels
            freeze_encoder = False if freeze_dinov3 is None else bool(freeze_dinov3)
            checkpoint_path = dinov3_checkpoint
            lora_rank = dinov3_lora_rank if dinov3_lora_rank is not None else dinov2_lora_rank
            lora_alpha = dinov3_lora_alpha if dinov3_lora_alpha is not None else dinov2_lora_alpha
            lora_dropout = dinov3_lora_dropout if dinov3_lora_dropout is not None else dinov2_lora_dropout
            lora_target = dinov3_lora_target if dinov3_lora_target is not None else dinov2_lora_target
            train_lora_only = dinov3_train_lora_only if dinov3_train_lora_only is not None else dinov2_train_lora_only
        else:
            dino_name = dinov2_name
            pretrained = bool(dinov2_pretrained)
            decoder_channels = dinov2_decoder_channels
            freeze_encoder = bool(freeze_dinov2)
            checkpoint_path = dinov2_checkpoint
            lora_rank = dinov2_lora_rank
            lora_alpha = dinov2_lora_alpha
            lora_dropout = dinov2_lora_dropout
            lora_target = dinov2_lora_target
            train_lora_only = dinov2_train_lora_only

        model_name = DINO_MODEL_ALIASES.get(dino_name, dino_name)
        return DINOSegmentationModel(
            model_name=model_name,
            pretrained=pretrained,
            in_channels=in_channels,
            classes=classes,
            decoder_channels=int(decoder_channels),
            freeze_encoder=freeze_encoder,
            checkpoint_path=checkpoint_path,
            lora_rank=int(lora_rank),
            lora_alpha=float(lora_alpha),
            lora_dropout=float(lora_dropout),
            lora_target=str(lora_target),
            train_lora_only=bool(train_lora_only),
        )

    if smp is None:
        raise ImportError(
            "segmentation_models_pytorch is required but not installed. "
            f"Original import error: {_SMP_IMPORT_ERROR!r}"
        )

    if arch == "unet":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    if arch == "unetplusplus":
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    if arch == "fpn":
        return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    if arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    raise ValueError(f"Unsupported architecture: {arch}")


class DiceBCELoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.7,
        bce_weight: float = 0.3,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        if smp is None:
            raise ImportError(
                "segmentation_models_pytorch is required but not installed. "
                f"Original import error: {_SMP_IMPORT_ERROR!r}"
            )
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.dice = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
        if pos_weight is None:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + self.bce_weight * self.bce(logits, targets)


class BinaryFocalWithLogitsLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = 0.75,
        pos_weight: float | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        self.pos_weight = None if pos_weight is None else float(pos_weight)
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        loss = ((1.0 - pt).clamp_min(1e-6) ** self.gamma) * bce

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = loss * alpha_t

        if self.pos_weight is not None and self.pos_weight != 1.0:
            pos_w = 1.0 + (self.pos_weight - 1.0) * targets
            loss = loss * pos_w

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DiceFocalLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.7,
        focal_weight: float = 0.3,
        focal_gamma: float = 2.0,
        focal_alpha: float | None = 0.75,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        if smp is None:
            raise ImportError(
                "segmentation_models_pytorch is required but not installed. "
                f"Original import error: {_SMP_IMPORT_ERROR!r}"
            )
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.dice = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
        self.focal = BinaryFocalWithLogitsLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
            pos_weight=pos_weight,
            reduction="mean",
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + self.focal_weight * self.focal(logits, targets)


def get_loss_fn(
    loss_type: str = "dice_focal",
    dice_weight: float = 0.7,
    bce_weight: float = 0.3,
    focal_weight: float = 0.3,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = 0.75,
    pos_weight: float | None = None,
) -> nn.Module:
    loss_type = str(loss_type).lower()
    if loss_type == "dice_bce":
        return DiceBCELoss(dice_weight=dice_weight, bce_weight=bce_weight, pos_weight=pos_weight)
    if loss_type == "dice_focal":
        return DiceFocalLoss(
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            pos_weight=pos_weight,
        )
    raise ValueError(f"Unsupported loss_type: {loss_type}")
