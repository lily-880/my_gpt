"""Day 2：积木的形状和因果遮罩。不需要 GPU。"""

import torch

from my_gpt.gpt import (
    GPTConfig,
    RMSNorm,
    CausalSelfAttention,
    FeedForward,
    apply_rope,
    build_causal_mask,
    causal_attention,
    precompute_rope,
)


def _tiny_cfg(**kwargs) -> GPTConfig:
    cfg = dict(n_layer=2, n_head=4, n_embd=32, block_size=16, vocab_size=100, dropout=0.0)
    cfg.update(kwargs)
    return GPTConfig(**cfg)


def test_causal_mask_hides_future():
    t = 8
    mask = build_causal_mask(t)
    scores = torch.zeros(2, 4, t, t)
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)

    future = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    assert probs[:, :, future].abs().max() < 1e-6
    # 每个位置对自己及以前的 token 概率和为 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 4, t), atol=1e-5)
    # 位置 0 只能看自己
    assert torch.allclose(probs[:, :, 0, 0], torch.ones(2, 4), atol=1e-5)


def test_rmsnorm_shape_and_unit_rms():
    x = torch.randn(2, 5, 32)
    y = RMSNorm(32)(x)
    assert y.shape == x.shape
    rms = torch.sqrt(y.float().pow(2).mean(dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_rope_keeps_shape_and_length():
    b, h, t, d = 2, 4, 8, 16
    x = torch.randn(b, h, t, d)
    cos, sin = precompute_rope(t, d)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    # 旋转不改变每个向量的欧氏长度
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_causal_attention_shapes_single_and_multi_head():
    t, d = 8, 16
    for n_head in (1, 4):
        q = torch.randn(2, n_head, t, d)
        k = torch.randn(2, n_head, t, d)
        v = torch.randn(2, n_head, t, d)
        out = causal_attention(q, k, v)
        assert out.shape == (2, n_head, t, d)


def test_mha_and_ffn_shapes():
    cfg = _tiny_cfg()
    x = torch.randn(3, 8, cfg.n_embd)
    y = CausalSelfAttention(cfg)(x)
    z = FeedForward(cfg)(x)
    assert y.shape == x.shape
    assert z.shape == x.shape


def test_attention_without_rope_still_same_shape():
    # learned PE 加在词嵌入上（Day 3），这里只关 RoPE，避免每层重复加位置
    cfg = _tiny_cfg(pos_encoding="learned")
    x = torch.randn(1, 8, cfg.n_embd)
    y = CausalSelfAttention(cfg)(x)
    assert y.shape == x.shape
