"""文本 → token 长流 → 固定长度窗口。train/val 按 token 切，不按剧本幕次切。"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import torch
from torch.utils.data import DataLoader, Dataset

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_DATA_PATH = DATA_DIR / "tiny_shakespeare.txt"
PLAYS_DIR = DATA_DIR / "shakespeare_plays"
COMPLETE_PATH = DATA_DIR / "shakespeare_complete.txt"


class TokenChunkDataset(Dataset):
    """非重叠窗口。每条样本长度 block_size；y 是 x 向右错开 1 个 token。"""

    def __init__(self, tokens: torch.Tensor, block_size: int):
        if tokens.ndim != 1:
            raise ValueError("tokens 必须是一维 LongTensor")
        if len(tokens) < block_size + 1:
            raise ValueError(f"至少需要 {block_size + 1} 个 token，实际 {len(tokens)}")
        self.tokens = tokens.long()
        self.block_size = block_size
        self.n_chunks = (len(self.tokens) - 1) // block_size

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = i * self.block_size
        chunk = self.tokens[start : start + self.block_size + 1]
        return chunk[:-1].clone(), chunk[1:].clone()


def download_tiny_shakespeare(dest: Path | None = None) -> Path:
    dest = Path(dest) if dest is not None else DEFAULT_DATA_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with urlopen(TINY_SHAKESPEARE_URL, timeout=60) as resp:
        dest.write_bytes(resp.read())
    return dest


def assemble_shakespeare_complete(dest: Path | None = None) -> Path:
    """把 shakespeare_plays/ 里各篇公有领域文本拼成一条长河，篇与篇之间空三行。"""
    dest = Path(dest) if dest is not None else COMPLETE_PATH
    files = sorted(p for p in PLAYS_DIR.glob("*.txt") if p.is_file())
    if not files:
        raise FileNotFoundError(f"没有找到剧本：{PLAYS_DIR}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    parts = [p.read_text(encoding="utf-8").strip() for p in files]
    dest.write_text("\n\n\n".join(parts) + "\n", encoding="utf-8")
    return dest


def load_pretrain_text() -> str:
    """优先用莎士比亚全集；没有再退回 tiny-shakespeare。"""
    if COMPLETE_PATH.exists() and COMPLETE_PATH.stat().st_size > 0:
        return COMPLETE_PATH.read_text(encoding="utf-8")
    if PLAYS_DIR.exists() and any(PLAYS_DIR.glob("*.txt")):
        return assemble_shakespeare_complete().read_text(encoding="utf-8")
    return download_tiny_shakespeare().read_text(encoding="utf-8")


def split_tokens(tokens: torch.Tensor, val_ratio: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio 必须在 (0, 1)")
    n = int(len(tokens) * (1.0 - val_ratio))
    if n < 1 or len(tokens) - n < 1:
        raise ValueError("切分后 train/val 不能为空，把文本加长或减小 val_ratio")
    return tokens[:n], tokens[n:]


def build_dataloaders(
    tokenizer,
    text: str,
    block_size: int,
    batch_size: int,
    val_ratio: float = 0.1,
    num_workers: int = 0,
):
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        raise ValueError(
            f"分词后只有 {len(ids)} 个 token（文本 {len(text)} 字符）。"
            "多半是 tokenizer 词表没加载成功，而不是语料太短。"
        )
    tokens = torch.tensor(ids, dtype=torch.long)
    train_tok, val_tok = split_tokens(tokens, val_ratio=val_ratio)
    train_ds = TokenChunkDataset(train_tok, block_size)
    val_ds = TokenChunkDataset(val_tok, block_size)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader
