"""KV-Cache：形状、与整段前向对齐。不加载莎士比亚权重。"""

import torch

from my_gpt.engine import Engine
from my_gpt.gpt import GPT, GPTConfig, causal_attention
from my_gpt.utils import sample_generate


def _tiny(**kwargs) -> GPT:
    cfg = dict(
        n_layer=2,
        n_head=4,
        n_embd=32,
        block_size=16,
        vocab_size=64,
        dropout=0.0,
        pos_encoding="rope",
    )
    cfg.update(kwargs)
    return GPT(GPTConfig(**cfg)).eval()


def test_causal_attention_decode_step_shape():
    q = torch.randn(2, 4, 1, 8)
    k = torch.randn(2, 4, 5, 8)
    v = torch.randn(2, 4, 5, 8)
    out = causal_attention(q, k, v)
    assert out.shape == (2, 4, 1, 8)


def test_prefill_logits_match_full_forward():
    torch.manual_seed(0)
    model = _tiny()
    idx = torch.randint(0, 64, (2, 7))
    full, _ = model(idx)
    cached, cache = model.forward_kv(idx)
    assert torch.allclose(full, cached, atol=1e-5)
    assert len(cache) == model.config.n_layer
    k, v = cache[0]
    assert k.shape == (2, 4, 7, 8)
    assert v.shape == k.shape


def test_decode_step_matches_full_forward():
    torch.manual_seed(1)
    model = _tiny()
    idx = torch.randint(0, 64, (1, 6))
    full, _ = model(idx)
    _, cache = model.forward_kv(idx[:, :-1])
    step, cache = model.forward_kv(idx[:, -1:], cache=cache)
    assert torch.allclose(step[:, -1, :], full[:, -1, :], atol=1e-5)
    assert cache[0][0].shape[2] == 6


def test_learned_pe_decode_matches():
    torch.manual_seed(2)
    model = _tiny(pos_encoding="learned")
    idx = torch.randint(0, 64, (1, 5))
    full, _ = model(idx)
    _, cache = model.forward_kv(idx[:, :-1])
    step, _ = model.forward_kv(idx[:, -1:], cache=cache)
    assert torch.allclose(step[:, -1, :], full[:, -1, :], atol=1e-5)


def test_engine_greedy_matches_no_cache():
    torch.manual_seed(3)
    model = _tiny()
    prompt = torch.randint(0, 64, (1, 4))
    a = sample_generate(model, prompt.clone(), 5, model.config.block_size, temperature=0.0)
    b = Engine(model, temperature=0.0, top_k=None).generate(prompt.clone(), 5)
    assert torch.equal(a, b)


def test_cache_rejects_over_block():
    model = _tiny()
    idx = torch.randint(0, 64, (1, 16))
    _, cache = model.forward_kv(idx)
    try:
        model.forward_kv(torch.zeros(1, 1, dtype=torch.long), cache=cache)
    except ValueError as e:
        assert "block_size" in str(e)
    else:
        raise AssertionError("应当拒绝超过 block_size")


def test_engine_generates_when_prompt_fills_block():
    torch.manual_seed(4)
    model = _tiny()
    prompt = torch.randint(0, 64, (1, model.config.block_size))
    a = sample_generate(model, prompt.clone(), 5, model.config.block_size, temperature=0.0)
    b = Engine(model, temperature=0.0, top_k=None).generate(prompt.clone(), 5)
    assert a.shape[1] == model.config.block_size + 5
    assert torch.equal(a, b)


def test_engine_keeps_long_prompt_and_matches_no_cache():
    torch.manual_seed(5)
    model = _tiny()
    prompt = torch.randint(0, 64, (1, model.config.block_size + 3))
    a = sample_generate(model, prompt.clone(), 4, model.config.block_size, temperature=0.0)
    b = Engine(model, temperature=0.0, top_k=None).generate(prompt.clone(), 4)
    assert b.shape[1] == prompt.size(1) + 4
    assert torch.equal(a, b)
