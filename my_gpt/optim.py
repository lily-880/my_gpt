"""AdamW 工厂：二维 Linear 权重做 weight decay；Embedding / RMSNorm / bias 不做。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _is_no_decay_param(name: str, param: torch.nn.Parameter) -> bool:
    if param.ndim < 2:
        return True
    if name.endswith("wte.weight") or name.endswith("wpe.weight"):
        return True
    return False


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_no_decay_param(name, param):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float):
    groups = param_groups(model, weight_decay)
    return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95))


def cosine_lr(step: int, *, base_lr: float, min_lr: float, schedule_steps: int, warmup_steps: int = 0) -> float:
    """按全局 step 算学习率，分段 resume 时不会因为本段 max_steps 被压扁。"""
    if step <= 0:
        return min_lr if warmup_steps <= 0 else base_lr * (1 / max(warmup_steps, 1))
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * step / warmup_steps
    if schedule_steps <= warmup_steps:
        return min_lr
    t = min(max(step - warmup_steps, 0) / (schedule_steps - warmup_steps), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
