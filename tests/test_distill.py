"""Day 8：蒸馏损失。不加载真实 GPT-2。"""

import math

import torch
import torch.nn.functional as F

from my_gpt.distill import distill_loss


def test_identical_logits_kl_near_zero():
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 32)
    labels = torch.randint(0, 32, (2, 8))
    loss, stats = distill_loss(logits, logits.clone(), labels, temperature=4.0, alpha=1.0)
    assert stats["kl"] < 1e-5
    assert abs(stats["kd"] - 16.0 * stats["kl"]) < 1e-5
    assert torch.isfinite(loss)


def test_alpha_zero_equals_hard_ce():
    torch.manual_seed(0)
    s = torch.randn(2, 8, 32)
    t = torch.randn(2, 8, 32)
    labels = torch.randint(0, 32, (2, 8))
    loss, stats = distill_loss(s, t, labels, temperature=4.0, alpha=0.0)
    ce = F.cross_entropy(s.reshape(-1, 32), labels.reshape(-1))
    assert abs(stats["ce"] - float(ce)) < 1e-5
    assert abs(float(loss) - float(ce)) < 1e-5


def test_alpha_one_is_scaled_kl():
    torch.manual_seed(0)
    s = torch.randn(2, 8, 32)
    t = torch.randn(2, 8, 32)
    labels = torch.randint(0, 32, (2, 8))
    tau = 4.0
    loss, stats = distill_loss(s, t, labels, temperature=tau, alpha=1.0)
    log_s = F.log_softmax(s.float() / tau, dim=-1).reshape(-1, 32)
    p_t = F.softmax(t.float() / tau, dim=-1).reshape(-1, 32)
    kl = F.kl_div(log_s, p_t, reduction="batchmean")
    assert abs(stats["kl"] - float(kl)) < 1e-5
    assert abs(float(loss) - float(tau * tau * kl)) < 1e-4


def test_shape_mismatch_raises():
    s = torch.randn(1, 4, 16)
    t = torch.randn(1, 4, 8)
    labels = torch.zeros(1, 4, dtype=torch.long)
    try:
        distill_loss(s, t, labels, temperature=1.0, alpha=0.5)
    except ValueError as e:
        assert "50257" in str(e) or "形状" in str(e)
    else:
        raise AssertionError("应当拒绝词表不一致")


def test_perplexity_from_ce_finite():
    from my_gpt.utils import perplexity_from_loss

    assert abs(perplexity_from_loss(math.log(2)) - 2.0) < 1e-6
