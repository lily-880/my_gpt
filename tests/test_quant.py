"""Day 9：CPU 动态量化。不加载莎士比亚权重。"""

import torch
import torch.nn as nn

from my_gpt.engine import Engine
from my_gpt.gpt import GPT, GPTConfig
from my_gpt.quant import n_plain_linears, quantize_linear_dynamic, state_dict_serialized_bytes
from my_gpt.utils import sample_generate


def _tiny() -> GPT:
    cfg = GPTConfig(
        n_layer=2,
        n_head=4,
        n_embd=32,
        block_size=16,
        vocab_size=64,
        dropout=0.0,
        pos_encoding="rope",
    )
    return GPT(cfg)


def test_quantize_replaces_all_linears():
    model = _tiny().eval()
    assert n_plain_linears(model) > 0
    q = quantize_linear_dynamic(model)
    assert n_plain_linears(q) == 0
    assert n_plain_linears(model) > 0


def test_quantized_forward_finite():
    torch.manual_seed(0)
    model = _tiny().eval()
    idx = torch.randint(0, 64, (2, 8))
    labels = torch.randint(0, 64, (2, 8))
    q = quantize_linear_dynamic(model)
    logits, loss = q(idx, labels)
    assert logits.shape == (2, 8, 64)
    assert torch.isfinite(logits).all()
    assert loss is not None and torch.isfinite(loss)


def test_quantized_state_dict_smaller_than_fp32():
    model = _tiny().eval()
    q = quantize_linear_dynamic(model)
    fp32_n = state_dict_serialized_bytes(model)
    int8_n = state_dict_serialized_bytes(q)
    assert int8_n < fp32_n


def test_quantize_rejects_cuda():
    if not torch.cuda.is_available():
        return
    model = _tiny().cuda()
    try:
        quantize_linear_dynamic(model)
    except ValueError as e:
        assert "CPU" in str(e)
    else:
        raise AssertionError("CUDA 模型应当被拒绝")
    finally:
        model.cpu()


def test_embedding_stays_float():
    model = _tiny().eval()
    q = quantize_linear_dynamic(model)
    assert isinstance(q.wte, nn.Embedding)
    assert q.wte.weight.dtype == torch.float32


def test_quantized_kvcache_runs_and_right_length():
    """动态量化按次缩放，不要求和无 cache 逐 token 相同。能跑、长度对即可。"""
    torch.manual_seed(0)
    model = _tiny().eval()
    q = quantize_linear_dynamic(model)
    prompt = torch.randint(0, 64, (1, 6))
    a = sample_generate(q, prompt.clone(), 5, q.config.block_size, temperature=0.0)
    b = Engine(q, temperature=0.0, top_k=None).generate(prompt.clone(), 5)
    assert a.shape == b.shape
    assert b.shape[1] == 11
    assert torch.isfinite(q(prompt)[0]).all()
