"""nnU-Net trainer used for Task 1 filtered pseudo-label fine-tuning."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerPseudoTop200(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.initial_lr = 1e-3

    def do_split(self):
        train_keys, val_keys = super().do_split()
        real = [key for key in train_keys if str(key).startswith("L")]
        pseudo = [key for key in train_keys if str(key).startswith("P")]
        if not real or not pseudo:
            raise RuntimeError(
                f"Expected real and pseudo cases, found real={len(real)} pseudo={len(pseudo)}"
            )
        repeat = max(1, math.ceil(len(pseudo) / len(real)))
        balanced_train = real * repeat + pseudo
        self.print_to_log_file(
            f"PseudoTop200 sampling: real_unique={len(real)}, real_repeat={repeat}, "
            f"real_effective={len(real) * repeat}, pseudo={len(pseudo)}, validation={len(val_keys)}"
        )
        return balanced_train, val_keys

    def _pseudo_weight(self) -> float:
        progress = self.current_epoch / max(self.num_epochs - 1, 1)
        if progress < 0.15:
            return 0.10 + 0.20 * progress / 0.15
        if progress < 0.70:
            return 0.30
        return max(0.0, 0.30 * (1.0 - progress) / 0.30)

    @staticmethod
    def _slice(value, indices):
        if isinstance(value, (list, tuple)):
            return [item[indices] for item in value]
        return value[indices]

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = (
            [item.to(self.device, non_blocking=True) for item in target]
            if isinstance(target, list)
            else target.to(self.device, non_blocking=True)
        )
        keys = [str(key) for key in batch["keys"]]
        real_indices = [index for index, key in enumerate(keys) if key.startswith("L")]
        pseudo_indices = [index for index, key in enumerate(keys) if key.startswith("P")]
        if len(real_indices) + len(pseudo_indices) != len(keys):
            raise RuntimeError(f"Unexpected training keys: {keys}")

        self.optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with context:
            output = self.network(data)
            components = []
            if real_indices:
                components.append(
                    self.loss(
                        self._slice(output, real_indices),
                        self._slice(target, real_indices),
                    )
                )
            pseudo_weight = self._pseudo_weight()
            if pseudo_indices:
                pseudo_loss = self.loss(
                    self._slice(output, pseudo_indices),
                    self._slice(target, pseudo_indices),
                )
                components.append(pseudo_weight * pseudo_loss)
            denominator = 1.0 + pseudo_weight if real_indices and pseudo_indices else 1.0
            loss = sum(components) / denominator

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy(), "pseudo_weight": pseudo_weight}
