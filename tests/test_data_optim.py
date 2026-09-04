"""Day 4：错位标签、AdamW 分组、过拟合。"""

import torch

from my_gpt.dataloader import (
    TokenChunkDataset,
    assemble_shakespeare_complete,
    build_dataloaders,
    split_tokens,
)
from my_gpt.gpt import GPT, GPTConfig, RMSNorm
from my_gpt.optim import build_optimizer, param_groups


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [(i % 50) + 1 for i in range(max(len(text), 400))]


def test_token_shift_labels():
    tokens = torch.arange(0, 21)
    ds = TokenChunkDataset(tokens, block_size=8)
    x, y = ds[0]
    assert x.tolist() == list(range(0, 8))
    assert y.tolist() == list(range(1, 9))
    assert torch.equal(x[1:], y[:-1])


def test_split_tokens_90_10():
    tokens = torch.arange(100)
    train, val = split_tokens(tokens, val_ratio=0.1)
    assert len(train) == 90
    assert len(val) == 10
    assert torch.equal(torch.cat([train, val]), tokens)


def test_assemble_shakespeare_complete(tmp_path, monkeypatch):
    import my_gpt.dataloader as dl

    plays = tmp_path / "plays"
    plays.mkdir()
    (plays / "b.txt").write_text("second play")
    (plays / "a.txt").write_text("first play")
    monkeypatch.setattr(dl, "PLAYS_DIR", plays)
    out = dl.assemble_shakespeare_complete(tmp_path / "all.txt")
    text = out.read_text(encoding="utf-8")
    assert text.index("first play") < text.index("second play")


def test_build_dataloaders_shapes():
    train_loader, val_loader = build_dataloaders(
        _FakeTokenizer(), "Hello world. " * 200, block_size=16, batch_size=2, val_ratio=0.1
    )
    x, y = next(iter(train_loader))
    assert x.shape[1] == 16
    assert y.shape == x.shape
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert len(val_loader) >= 1


def test_adamw_decay_skips_embed_and_norm():
    cfg = GPTConfig(
        n_layer=1, n_head=2, n_embd=16, block_size=8, vocab_size=50, dropout=0.0
    )
    model = GPT(cfg)
    groups = param_groups(model, weight_decay=0.1)
    decay_ids = {id(p) for p in groups[0]["params"]}
    no_decay_ids = {id(p) for p in groups[1]["params"]}
    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0
    assert id(model.wte.weight) in no_decay_ids
    assert id(model.lm_head.weight) in decay_ids
    for module in model.modules():
        if isinstance(module, RMSNorm):
            assert id(module.weight) in no_decay_ids
    assert decay_ids.isdisjoint(no_decay_ids)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_grouped = sum(p.numel() for g in groups for p in g["params"])
    assert n_params == n_grouped


def test_overfit_one_batch():
    cfg = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=8,
        vocab_size=64,
        dropout=0.0,
        pos_encoding="rope",
    )
    torch.manual_seed(0)
    model = GPT(cfg)
    opt = build_optimizer(model, learning_rate=1e-2, weight_decay=0.0)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    losses = []
    for _ in range(60):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.5
