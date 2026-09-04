"""Day 3：整模型前向、checkpoint、分词器。"""

from pathlib import Path

import torch

from my_gpt.checkpoint import gpt_config_from_checkpoint, load_checkpoint, save_checkpoint
from my_gpt.gpt import GPT, GPTConfig
from my_gpt.tokenizer import GPT2TokenizerWrapper


def _tiny(**kwargs) -> GPTConfig:
    cfg = dict(
        n_layer=2,
        n_head=4,
        n_embd=32,
        block_size=16,
        vocab_size=100,
        dropout=0.0,
        pos_encoding="rope",
    )
    cfg.update(kwargs)
    return GPTConfig(**cfg)


def test_gpt_forward_logits_and_loss():
    cfg = _tiny()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (3, 8))
    labels = torch.randint(0, cfg.vocab_size, (3, 8))
    logits, loss = model(idx, labels)
    assert logits.shape == (3, 8, cfg.vocab_size)
    assert loss is not None and loss.ndim == 0
    assert torch.isfinite(loss)
    logits_only, no_loss = model(idx)
    assert no_loss is None
    assert logits_only.shape == logits.shape


def test_gpt_learned_pe_forward():
    cfg = _tiny(pos_encoding="learned")
    model = GPT(cfg)
    assert model.wpe is not None
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, loss = model(idx)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert loss is None


def test_checkpoint_roundtrip(tmp_path: Path):
    cfg = _tiny()
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    idx = torch.randint(0, cfg.vocab_size, (2, 4))
    _, loss = model(idx, idx)
    loss.backward()
    opt.step()

    ckpt = tmp_path / "step10.pt"
    save_checkpoint(ckpt, model, optimizer=opt, step=10, config=cfg)

    restored = GPT(gpt_config_from_checkpoint(ckpt))
    opt2 = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    payload = load_checkpoint(ckpt, model=restored, optimizer=opt2)
    assert payload["step"] == 10
    for a, b in zip(model.parameters(), restored.parameters()):
        assert torch.equal(a, b)


def test_gpt2_tokenizer_roundtrip():
    import pytest

    tok = GPT2TokenizerWrapper()
    assert tok.vocab_size == 50257
    assert tok.eos_id == tok.pad_id
    text = "hello world"
    ids = tok.encode(text)
    assert ids == [31373, 995]
    assert tok.decode(ids).strip() == text
    stream = tok.encode_documents(["hi", "bye"])
    assert stream[-1] == tok.eos_id
    assert stream.count(tok.eos_id) == 2
