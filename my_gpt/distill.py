"""知识蒸馏损失：L = α T² KL(p_T^τ || p_S^τ) + (1-α) CE(y, p_S)。"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F


def distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    alpha: float,
) -> tuple[torch.Tensor, dict]:
    """student/teacher logits 形状 (B, T, V)，labels (B, T)。

    KL 在温度 τ 的软分布上算，再乘 τ²（Hinton：补回 softmax(z/τ) 把梯度缩小的 1/τ²）。
    CE 用 τ=1 的硬标签，不乘 τ²。
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Student/Teacher logits 形状不一致：{tuple(student_logits.shape)} vs "
            f"{tuple(teacher_logits.shape)}。词表必须都是 GPT-2 的 50257。"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必须在 [0, 1]")
    if temperature <= 0:
        raise ValueError("temperature 必须 > 0")

    tau = float(temperature)
    vocab = student_logits.size(-1)
    ce = F.cross_entropy(student_logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100)

    log_p_s = F.log_softmax(student_logits.float() / tau, dim=-1)
    p_t = F.softmax(teacher_logits.float() / tau, dim=-1)
    kl = F.kl_div(
        log_p_s.reshape(-1, vocab),
        p_t.reshape(-1, vocab),
        reduction="batchmean",
    )
    kd = (tau * tau) * kl
    loss = alpha * kd + (1.0 - alpha) * ce
    stats = {
        "loss": float(loss.detach()),
        "ce": float(ce.detach()),
        "kl": float(kl.detach()),
        "kd": float(kd.detach()),
    }
    return loss, stats


@torch.no_grad()
def evaluate_ce_from_logits_fn(
    logits_fn,
    loader,
    device: torch.device,
    autocast_ctx=None,
    max_batches: int | None = None,
) -> float:
    """对任意「x → logits」算硬标签 CE。用来对比 Student 和冻结 Teacher 的 val PPL。"""
    if autocast_ctx is None:
        autocast_ctx = nullcontext()
    total = 0.0
    n = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            logits = logits_fn(x)
            vocab = logits.size(-1)
            loss = F.cross_entropy(logits.float().reshape(-1, vocab), y.reshape(-1))
        total += float(loss)
        n += 1
    if n == 0:
        raise ValueError("验证集为空")
    return total / n


def load_gpt2_teacher(name_or_path: str, device: torch.device, dtype):
    """加载冻结的 HF GPT-2。优先本地目录 / 缓存，避免离线时空词表。"""
    from transformers import GPT2LMHeadModel

    path = Path(name_or_path)
    kwargs = {"torch_dtype": dtype} if dtype is not None else {}
    if path.exists():
        model = GPT2LMHeadModel.from_pretrained(path, local_files_only=True, **kwargs)
    else:
        try:
            model = GPT2LMHeadModel.from_pretrained(name_or_path, local_files_only=True, **kwargs)
        except Exception:
            model = GPT2LMHeadModel.from_pretrained(name_or_path, **kwargs)
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)
