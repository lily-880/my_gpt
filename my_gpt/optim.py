"""AdamW 工厂：二维 Linear 权重做 weight decay；Embedding / RMSNorm / bias 不做。"""

from __future__ import annotations

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
