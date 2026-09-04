"""Day 5：PPL、贪心生成、单步训练。"""

import math

import torch

from my_gpt.gpt import GPT, GPTConfig
from my_gpt.optim import build_optimizer
from my_gpt.utils import (
    amp_dtype_and_scaler,
    greedy_generate,
    perplexity_from_loss,
    train_step,
)


def _tiny() -> GPTConfig:
    return GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=8, vocab_size=32, dropout=0.0)


def test_perplexity_from_ce():
    assert abs(perplexity_from_loss(0.0) - 1.0) < 1e-6
    assert abs(perplexity_from_loss(math.log(2)) - 2.0) < 1e-6


def test_amp_cpu_disables_scaler():
    dtype, scaler = amp_dtype_and_scaler(torch.device("cpu"))
    assert dtype is None and scaler is None


def test_greedy_appends_tokens():
    model = GPT(_tiny())
    idx = torch.randint(0, 32, (1, 3))
    out = greedy_generate(model, idx, max_new_tokens=5, block_size=8)
    assert out.shape == (1, 8)


def test_run_train_writes_checkpoint(tmp_path):
    from scripts.base_train import run_train

    class FakeTok:
        eos_id = 0
        vocab_size = 64

        def encode(self, text: str) -> list[int]:
            return [(i % 50) + 1 for i in range(800)]

        def decode(self, ids) -> str:
            return "ok"

    cfg = {
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 16,
        "block_size": 16,
        "vocab_size": 64,
        "dropout": 0.0,
        "pos_encoding": "rope",
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "batch_size": 2,
        "max_steps": 3,
        "grad_clip": 1.0,
        "log_every": 1,
        "eval_every": 3,
        "save_every": 3,
        "sample_every": 3,
        "max_new_tokens": 4,
        "eval_batches": 1,
    }
    metrics = run_train(
        cfg,
        text="ignored",
        tokenizer=FakeTok(),
        device=torch.device("cpu"),
        ckpt_dir=tmp_path,
        prompt="a",
    )
    assert (tmp_path / "last.pt").exists()
    assert math.isfinite(metrics["train_loss"])


def test_run_train_resume_continues_step(tmp_path):
    from scripts.base_train import run_train

    class FakeTok:
        eos_id = 0
        vocab_size = 64

        def encode(self, text: str) -> list[int]:
            return [(i % 50) + 1 for i in range(800)]

        def decode(self, ids) -> str:
            return "ok"

    cfg = {
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 16,
        "block_size": 16,
        "vocab_size": 64,
        "dropout": 0.0,
        "pos_encoding": "rope",
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "batch_size": 2,
        "max_steps": 2,
        "grad_clip": 1.0,
        "log_every": 2,
        "eval_every": 0,
        "save_every": 0,
        "sample_every": 0,
        "max_new_tokens": 2,
        "eval_batches": 1,
    }
    first_dir = tmp_path / "a"
    run_train(
        cfg,
        text="ignored",
        tokenizer=FakeTok(),
        device=torch.device("cpu"),
        ckpt_dir=first_dir,
        prompt="a",
    )
    cfg["max_steps"] = 4
    later = tmp_path / "b"
    metrics = run_train(
        cfg,
        text="ignored",
        tokenizer=FakeTok(),
        device=torch.device("cpu"),
        ckpt_dir=later,
        prompt="a",
        resume=first_dir / "last.pt",
    )
    assert metrics["step"] == 4
    payload = torch.load(later / "last.pt", map_location="cpu", weights_only=False)
    assert payload["step"] == 4


def test_train_step_finite_and_changes_weights():
    torch.manual_seed(0)
    model = GPT(_tiny())
    opt = build_optimizer(model, learning_rate=1e-2, weight_decay=0.0)
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    before = model.lm_head.weight.detach().clone()
    loss = train_step(model, opt, x, y, grad_clip=1.0)
    assert math.isfinite(loss)
    assert not torch.equal(before, model.lm_head.weight.detach())
