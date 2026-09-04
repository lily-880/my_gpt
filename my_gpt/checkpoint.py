"""训练快照：模型权重、优化器状态、步数、超参。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch

from my_gpt.gpt import GPTConfig


def save_checkpoint(path, model, optimizer=None, step: int = 0, config=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if config is None:
        config = getattr(model, "config", None)
    if is_dataclass(config):
        config = asdict(config)
    payload = {
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "step": int(step),
        "config": config,
    }
    torch.save(payload, path)


def load_checkpoint(path, model=None, optimizer=None, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def gpt_config_from_checkpoint(path, map_location="cpu") -> GPTConfig:
    payload = load_checkpoint(path, map_location=map_location)
    raw = payload.get("config") or {}
    fields = {k: raw[k] for k in GPTConfig.__dataclass_fields__ if k in raw}
    return GPTConfig(**fields)
