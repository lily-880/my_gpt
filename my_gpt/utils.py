"""训练循环用的小工具：AMP、PPL、验证、贪心生成、单步更新。"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


def perplexity_from_loss(loss: torch.Tensor | float) -> float:
    """PPL = exp(平均交叉熵)。loss 必须是自然对数下的 CE（PyTorch 默认）。"""
    return math.exp(float(loss))


def amp_dtype_and_scaler(device: torch.device):
    """CPU 不用 AMP。CUDA 上优先 bf16（不用 GradScaler）；否则 fp16 + GradScaler。"""
    if device.type != "cuda":
        return None, None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, None
    scaler = torch.amp.GradScaler("cuda")
    return torch.float16, scaler


def autocast_context(device: torch.device, amp_dtype):
    if amp_dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    scaler=None,
    grad_clip: float = 1.0,
    autocast_ctx=None,
) -> float:
    if autocast_ctx is None:
        autocast_ctx = nullcontext()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with autocast_ctx:
        _, loss = model(x, y)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader, device: torch.device, autocast_ctx=None, max_batches: int | None = None) -> float:
    if autocast_ctx is None:
        autocast_ctx = nullcontext()
    model.eval()
    total = 0.0
    n = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            _, loss = model(x, y)
        total += float(loss)
        n += 1
    model.train()
    if n == 0:
        raise ValueError("验证集为空")
    return total / n


@torch.no_grad()
def greedy_generate(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, block_size: int) -> torch.Tensor:
    """每步取概率最大的下一个 token。容易复读，报告样例请用 sample_generate。"""
    return sample_generate(model, idx, max_new_tokens, block_size, temperature=0.0)


@torch.no_grad()
def sample_generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    *,
    temperature: float = 0.8,
    top_k: int | None = 50,
) -> torch.Tensor:
    """按概率抽样。temperature=0 退化为 greedy；top_k 限制每步只看分数最高的 k 个。"""
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block_size:])
        logits = logits[:, -1, :]
        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < thresh, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    if was_training:
        model.train()
    return idx
