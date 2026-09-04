"""Day 4：在 CPU 上过拟合一个 batch，确认反向传播和优化器真的在干活。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_gpt.dataloader import TokenChunkDataset, download_tiny_shakespeare
from my_gpt.gpt import GPT, GPTConfig
from my_gpt.optim import build_optimizer
from my_gpt.tokenizer import GPT2TokenizerWrapper


def _tiny_config() -> GPTConfig:
    return GPTConfig(
        n_layer=2,
        n_head=4,
        n_embd=64,
        block_size=32,
        vocab_size=50257,
        dropout=0.0,
        pos_encoding="rope",
    )


def run_overfit(steps: int = 40, device: str = "cpu") -> tuple[float, float]:
    tok = GPT2TokenizerWrapper()
    try:
        text = download_tiny_shakespeare().read_text(encoding="utf-8")
    except OSError:
        text = ("To be, or not to be, that is the question. " * 80)

    tokens = torch.tensor(tok.encode(text), dtype=torch.long)
    ds = TokenChunkDataset(tokens, block_size=32)
    x, y = ds[0]
    x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)

    model = GPT(_tiny_config()).to(device)
    model.train()
    opt = build_optimizer(model, learning_rate=3e-4, weight_decay=0.1)

    first = last = None
    for step in range(steps):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        value = float(loss.detach())
        if step == 0:
            first = value
        last = value
        if step == 0 or (step + 1) % 10 == 0:
            print(f"step {step + 1:03d}  loss={value:.4f}")
    return first, last


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    first, last = run_overfit(steps=args.steps)
    print(f"overfit: {first:.4f} -> {last:.4f}")
    if last >= first:
        raise SystemExit("loss 没有下降，检查模型或学习率")
