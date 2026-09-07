"""Decoder-only GPT：RoPE 或可学习位置、RMSNorm、因果注意力。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 256
    vocab_size: int = 50257
    dropout: float = 0.0
    pos_encoding: str = "rope"  # "rope" 或 "learned"


def build_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """下三角为 True：位置 i 只能看见 j <= i。形状 (1, 1, T, T)，方便广播到 (B, H, T, T)。"""
    ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    return torch.tril(ones)[None, None, :, :]


class RMSNorm(nn.Module):
    """按最后一维做 RMS 归一化，再乘可学习的缩放。nanochat 的 rms_norm 没有这组 weight。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 用 float32 算方差，避免半精度下 underflow
        x_f = x.float()
        rms = torch.sqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * (x_f / rms)).type_as(x)


def precompute_rope(seq_len: int, head_dim: int, base: float = 10000.0, device=None):
    """预先算好每个位置、每个偶数通道的 cos/sin。返回形状 (1, 1, T, D/2)。"""
    if head_dim % 2 != 0:
        raise ValueError("RoPE 要求 head_dim 为偶数")
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """把 RoPE 加到 q 或 k 上。x 形状 (B, H, T, D)，把最后一维拆成两半再旋转。"""
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    # 教科书方向：角度为 +theta。nanochat 用了 -theta，相对位置仍然成立。
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1)


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float = 0.0) -> torch.Tensor:
    """q 是 (B, H, Tq, D)，k/v 是 (B, H, Tk, D)。Tq==Tk 是整段；解码时 Tq=1、Tk=已有长度。

    假定 q 是 k 的后缀：第 i 个 query 的绝对位置是 (Tk-Tq+i)，只能看见 <= 该位置的 key。
    """
    tq, tk = q.size(-2), k.size(-2)
    if tk < tq:
        raise ValueError(f"k 长度 {tk} 短于 q 长度 {tq}")
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = (q @ k.transpose(-2, -1)) * scale
    offset = tk - tq
    q_pos = torch.arange(tq, device=q.device) + offset
    k_pos = torch.arange(tk, device=q.device)
    allow = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    scores = scores.masked_fill(~allow, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    weights = F.dropout(weights, p=dropout_p, training=dropout_p > 0.0)
    return weights @ v


class CausalSelfAttention(nn.Module):
    """多头因果注意力；n_head=1 时退化为单头。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd 必须能被 n_head 整除")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_rope = config.pos_encoding == "rope"

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        cos, sin = precompute_rope(config.block_size, self.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        b, t, c = x.shape
        past_len = 0 if kv_cache is None else kv_cache[0].size(2)
        total = past_len + t
        if total > self.cos.size(2):
            raise ValueError(f"序列长度 {total} 超过 block_size={self.cos.size(2)}")

        qkv = self.c_attn(x)
        q, k, v = qkv.split(c, dim=-1)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos = self.cos[:, :, past_len:total, :]
            sin = self.sin[:, :, past_len:total, :]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        y = causal_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.c_proj(y)
        if use_cache:
            return y, (k, v)
        return y


class FeedForward(nn.Module):
    """两层 MLP，中间放大 4 倍。激活用 GELU（GPT-2 常见）；nanochat 用的是 ReLU²。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class TransformerBlock(nn.Module):
    """Pre-Norm 残差块：先归一化再算注意力 / FFN，再加回原输入。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        if use_cache:
            attn_out, kv = self.attn(self.ln_1(x), kv_cache=kv_cache, use_cache=True)
            x = x + attn_out
            x = x + self.ffn(self.ln_2(x))
            return x, kv
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class GPT(nn.Module):
    """词嵌入 → N 层 TransformerBlock → 最终 RMSNorm → 词表上的 logits。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = (
            nn.Embedding(config.block_size, config.n_embd)
            if config.pos_encoding == "learned"
            else None
        )
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, labels: torch.Tensor | None = None):
        _, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(f"序列长度 {t} 超过 block_size={self.config.block_size}")

        x = self.wte(idx)
        if self.wpe is not None:
            pos = torch.arange(t, device=idx.device)
            x = x + self.wpe(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if labels is not None:
            # labels 与 idx 同形状：第 t 位的 label 是「下一个该出现的 token」。
            # 填 -100 的位置不计入 loss（给以后 padding 用）。
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def forward_kv(
        self,
        idx: torch.Tensor,
        cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """只算新 token。cache 每层是 (k, v)，形状 (B, H, T_past, head_dim)。解码时 idx 通常是 (B, 1)。"""
        t = idx.shape[1]
        past_len = 0 if cache is None else cache[0][0].size(2)
        total = past_len + t
        if total > self.config.block_size:
            raise ValueError(f"序列长度 {total} 超过 block_size={self.config.block_size}")

        x = self.wte(idx)
        if self.wpe is not None:
            pos = torch.arange(past_len, total, device=idx.device)
            x = x + self.wpe(pos)
        x = self.drop(x)
        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            layer_kv = None if cache is None else cache[i]
            x, layer_kv = block(x, kv_cache=layer_kv, use_cache=True)
            new_cache.append(layer_kv)
        logits = self.lm_head(self.ln_f(x))
        return logits, new_cache
