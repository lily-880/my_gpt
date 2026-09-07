"""CPU 动态 INT8 PTQ：只量化 nn.Linear。嵌入、RMSNorm、注意力矩阵乘保持浮点。

动态量化按每次 forward 的激活范围缩放，所以 INT8 + KV-Cache 能跑，但 greedy
不必和「每步整窗重算」逐 token 相同。fp32 上两者对齐。
"""

from __future__ import annotations

import copy
import io
import time

import torch
import torch.nn as nn

from my_gpt.utils import sample_generate


def n_plain_linears(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if type(m) is nn.Linear)


def quantize_linear_dynamic(model: nn.Module) -> nn.Module:
    """返回新模型；原模型不动。只能放 CPU，不是 GPU Tensor Core INT8。"""
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("动态量化只支持 CPU，先 model.to('cpu')")
    quantized = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model),
        {nn.Linear},
        dtype=torch.qint8,
    )
    leftover = n_plain_linears(quantized)
    if leftover != 0:
        raise RuntimeError(f"仍有 {leftover} 个未量化的 nn.Linear")
    return quantized


def state_dict_serialized_bytes(module: nn.Module) -> int:
    """序列化 state_dict 的字节数。训练 ckpt 里的 optimizer 不算进去。"""
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.getbuffer().nbytes


def tensor_nbytes(module: nn.Module) -> int:
    """state_dict 里能数出来的 Tensor 字节。量化 packed 权重有时走这条，有时只出现在序列化里。"""
    total = 0
    for value in module.state_dict().values():
        if torch.is_tensor(value):
            total += int(value.numel()) * int(value.element_size())
    return total


@torch.no_grad()
def time_generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    *,
    temperature: float,
    top_k: int,
    warmup: int = 1,
    repeats: int = 1,
) -> float:
    """平均一次 generate 的墙钟秒数。先 warmup，再计时。"""
    was_training = model.training
    model.eval()
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        temperature=temperature,
        top_k=top_k,
    )
    for _ in range(warmup):
        sample_generate(model, idx.clone(), **kwargs)
    t0 = time.perf_counter()
    for _ in range(repeats):
        sample_generate(model, idx.clone(), **kwargs)
    elapsed = time.perf_counter() - t0
    if was_training:
        model.train()
    return elapsed / max(repeats, 1)


def time_generate_kv(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float,
    top_k: int,
    warmup: int = 1,
    repeats: int = 1,
) -> float:
    """同上，走 Engine / KV-Cache。量化后的 GPT 也能用。"""
    from my_gpt.engine import Engine

    was_training = model.training
    model.eval()
    engine = Engine(model, temperature=temperature, top_k=top_k)
    for _ in range(warmup):
        engine.generate(idx.clone(), max_new_tokens)
    t0 = time.perf_counter()
    for _ in range(repeats):
        engine.generate(idx.clone(), max_new_tokens)
    elapsed = time.perf_counter() - t0
    if was_training:
        model.train()
    return elapsed / max(repeats, 1)
