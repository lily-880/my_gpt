"""文本 → token。默认按篇切 train/val，避免验证全集中在拼接长河的尾巴。"""

from __future__ import annotations

import hashlib
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

SPLIT_PER_DOC = "per_doc"
SPLIT_PER_DOC_TAIL = "per_doc_tail"
SPLIT_TOKEN_TAIL = "token_tail"
_VALID_SPLITS = {SPLIT_PER_DOC, SPLIT_PER_DOC_TAIL, SPLIT_TOKEN_TAIL}


class TokenChunkDataset(Dataset):
    """固定长度窗口。多篇时窗口不跨篇；interleave=True 时按篇轮转，验证前几批就能扫到各篇。"""

    def __init__(
        self,
        tokens: torch.Tensor | list[torch.Tensor],
        block_size: int,
        *,
        random_windows: bool = False,
        interleave: bool = False,
    ):
        if isinstance(tokens, torch.Tensor):
            docs = [tokens]
        else:
            docs = list(tokens)
        if not docs:
            raise ValueError("至少需要一篇 token 序列")
        self.docs = []
        self.chunk_counts: list[int] = []
        self.max_starts: list[int] = []
        for t in docs:
            if t.ndim != 1:
                raise ValueError("tokens 必须是一维 LongTensor")
            t = t.long()
            if len(t) < block_size + 1:
                raise ValueError(f"至少需要 {block_size + 1} 个 token，实际 {len(t)}")
            self.docs.append(t)
            self.chunk_counts.append((len(t) - 1) // block_size)
            self.max_starts.append(len(t) - block_size - 1)
        self.block_size = block_size
        self.random_windows = random_windows
        self.index = _window_index(self.chunk_counts, interleave=interleave)
        self._start_weight = torch.tensor(
            [m + 1 for m in self.max_starts], dtype=torch.double
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.random_windows:
            doc_i = int(torch.multinomial(self._start_weight, 1).item())
            start = int(torch.randint(0, self.max_starts[doc_i] + 1, (1,)).item())
        else:
            doc_i, chunk_i = self.index[i]
            start = chunk_i * self.block_size
        chunk = self.docs[doc_i][start : start + self.block_size + 1]
        return chunk[:-1].clone(), chunk[1:].clone()


def _window_index(chunk_counts: list[int], *, interleave: bool) -> list[tuple[int, int]]:
    if not interleave:
        return [(d, k) for d, n in enumerate(chunk_counts) for k in range(n)]
    index: list[tuple[int, int]] = []
    max_n = max(chunk_counts, default=0)
    for k in range(max_n):
        for d, n in enumerate(chunk_counts):
            if k < n:
                index.append((d, k))
    return index


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


def load_pretrain_documents() -> list[tuple[str, str]]:
    """优先按篇返回莎士比亚文本；没有分篇文件再退回拼好的长河或 tiny-shakespeare。"""
    if PLAYS_DIR.exists():
        files = sorted(p for p in PLAYS_DIR.glob("*.txt") if p.is_file())
        if files:
            return [(p.name, p.read_text(encoding="utf-8").strip()) for p in files]
    if COMPLETE_PATH.exists() and COMPLETE_PATH.stat().st_size > 0:
        return [(COMPLETE_PATH.name, COMPLETE_PATH.read_text(encoding="utf-8"))]
    return [(DEFAULT_DATA_PATH.name, download_tiny_shakespeare().read_text(encoding="utf-8"))]


def load_pretrain_text() -> str:
    """拼成一条字符串。训练默认应走 load_pretrain_documents，避免再按长河尾巴切。"""
    docs = load_pretrain_documents()
    return "\n\n\n".join(text for _, text in docs) + "\n"


def split_tokens(tokens: torch.Tensor, val_ratio: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio 必须在 (0, 1)")
    n = int(len(tokens) * (1.0 - val_ratio))
    if n < 1 or len(tokens) - n < 1:
        raise ValueError("切分后 train/val 不能为空，把文本加长或减小 val_ratio")
    return tokens[:n], tokens[n:]


def _encode_docs(tokenizer, documents: list[tuple[str, str]]) -> list[tuple[str, torch.Tensor]]:
    encoded: list[tuple[str, torch.Tensor]] = []
    for name, text in documents:
        ids = tokenizer.encode(text)
        encoded.append((name, torch.tensor(ids, dtype=torch.long)))
    return encoded


def _doc_generator(seed: int, name: str) -> torch.Generator:
    digest = hashlib.sha256(f"{int(seed)}:{name}".encode()).digest()
    g = torch.Generator()
    g.manual_seed(int.from_bytes(digest[:8], "little") % (2**31))
    return g


def _merge_window_runs(
    flags: list[bool],
    tokens: torch.Tensor,
    block_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """连续同属 train/val 的窗口拼成一段，避免在挖洞处伪造衔接。"""
    train_segs: list[torch.Tensor] = []
    val_segs: list[torch.Tensor] = []
    n = len(flags)
    i = 0
    while i < n:
        j = i + 1
        while j < n and flags[j] == flags[i]:
            j += 1
        start = i * block_size
        end = (j - 1) * block_size + block_size + 1
        piece = tokens[start:end]
        if flags[i]:
            val_segs.append(piece)
        else:
            train_segs.append(piece)
        i = j
    rem = tokens[n * block_size + 1 :]
    if len(rem) > 0 and n > 0 and not flags[-1]:
        train_segs[-1] = torch.cat([train_segs[-1], rem])
    elif len(rem) >= block_size + 1:
        train_segs.append(rem)
    return train_segs, val_segs


def _split_one_doc_random(
    tokens: torch.Tensor,
    val_ratio: float,
    block_size: int,
    generator: torch.Generator,
) -> tuple[list[torch.Tensor], list[torch.Tensor]] | None:
    n_windows = (len(tokens) - 1) // block_size
    if n_windows < 2:
        return None
    n_val = int(round(n_windows * val_ratio))
    n_val = min(max(n_val, 1), n_windows - 1)
    perm = torch.randperm(n_windows, generator=generator)
    val_ids = {int(i) for i in perm[:n_val].tolist()}
    flags = [i in val_ids for i in range(n_windows)]
    return _merge_window_runs(flags, tokens, block_size)


def split_document_tokens(
    encoded: list[tuple[str, torch.Tensor]],
    val_ratio: float,
    min_tokens: int,
    *,
    split: str = SPLIT_PER_DOC,
    seed: int = 42,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict]:
    """per_doc：每篇随机抽约 val_ratio 的不重叠窗口进验证（种子固定，可复现）。
    per_doc_tail：每篇末尾 val_ratio。token_tail：先拼成长河再切一次。"""
    split = str(split).lower()
    if split not in _VALID_SPLITS:
        raise ValueError(f"split 只能是 {sorted(_VALID_SPLITS)}，收到 {split!r}")

    if split == SPLIT_TOKEN_TAIL:
        tokens = torch.cat([t for _, t in encoded]) if encoded else torch.tensor([], dtype=torch.long)
        if len(tokens) < 2:
            raise ValueError(
                f"分词后只有 {len(tokens)} 个 token。"
                "多半是 tokenizer 词表没加载成功，而不是语料太短。"
            )
        train_tok, val_tok = split_tokens(tokens, val_ratio=val_ratio)
        info = {
            "split": SPLIT_TOKEN_TAIL,
            "n_docs": len(encoded),
            "train_tokens": int(len(train_tok)),
            "val_tokens": int(len(val_tok)),
            "train_docs": [n for n, _ in encoded],
            "val_docs": [encoded[-1][0]] if encoded else [],
            "skipped_val": [],
            "seed": int(seed),
        }
        return [train_tok], [val_tok], info

    block_size = min_tokens - 1
    if block_size < 1:
        raise ValueError("min_tokens 必须 >= 2")

    train_docs: list[torch.Tensor] = []
    val_docs: list[torch.Tensor] = []
    train_names: list[str] = []
    val_names: list[str] = []
    skipped_val: list[str] = []
    skipped_all: list[str] = []
    for name, tokens in encoded:
        if len(tokens) < min_tokens:
            skipped_all.append(name)
            continue
        if split == SPLIT_PER_DOC_TAIL:
            train_tok, val_tok = split_tokens(tokens, val_ratio=val_ratio)
            if len(train_tok) < min_tokens or len(val_tok) < min_tokens:
                train_docs.append(tokens)
                train_names.append(name)
                skipped_val.append(name)
                continue
            train_segs, val_segs = [train_tok], [val_tok]
        else:
            parts = _split_one_doc_random(
                tokens, val_ratio, block_size, _doc_generator(seed, name)
            )
            if parts is None:
                train_docs.append(tokens)
                train_names.append(name)
                skipped_val.append(name)
                continue
            train_segs, val_segs = parts
        train_docs.extend(train_segs)
        val_docs.extend(val_segs)
        train_names.append(name)
        val_names.append(name)

    if not train_docs or not val_docs:
        raise ValueError("按篇切分后 train/val 为空，把文本加长、减小 block_size 或减小 val_ratio")

    info = {
        "split": split,
        "seed": int(seed),
        "n_docs": len(encoded),
        "train_tokens": int(sum(len(t) for t in train_docs)),
        "val_tokens": int(sum(len(t) for t in val_docs)),
        "train_docs": train_names,
        "val_docs": val_names,
        "skipped_val": skipped_val,
        "skipped_all": skipped_all,
    }
    return train_docs, val_docs, info


def format_split_info(info: dict) -> str:
    skipped = info.get("skipped_val") or []
    skip_txt = f"  no_val_window={len(skipped)}"
    if skipped:
        skip_txt += f"({', '.join(skipped)})"
    seed_txt = f"  seed={info['seed']}" if info.get("seed") is not None else ""
    return (
        f"split={info['split']}  docs={info['n_docs']}  "
        f"val_docs={len(info['val_docs'])}  "
        f"train_tok={info['train_tokens']}  val_tok={info['val_tokens']}"
        f"{seed_txt}{skip_txt}"
    )


def build_dataloaders(
    tokenizer,
    text: str | None = None,
    block_size: int = 256,
    batch_size: int = 8,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    random_windows: bool = False,
    documents: list[tuple[str, str]] | None = None,
    split: str = SPLIT_PER_DOC,
    split_seed: int = 42,
):
    if documents is None:
        if text is None:
            raise ValueError("需要 text 或 documents")
        documents = [("text", text)]
    encoded = _encode_docs(tokenizer, documents)
    n_ids = sum(len(t) for _, t in encoded)
    if n_ids < 2:
        n_chars = sum(len(t) for _, t in documents)
        raise ValueError(
            f"分词后只有 {n_ids} 个 token（文本 {n_chars} 字符）。"
            "多半是 tokenizer 词表没加载成功，而不是语料太短。"
        )
    train_docs, val_docs, info = split_document_tokens(
        encoded,
        val_ratio=val_ratio,
        min_tokens=block_size + 1,
        split=split,
        seed=split_seed,
    )
    print(format_split_info(info), flush=True)
    train_ds = TokenChunkDataset(train_docs, block_size, random_windows=random_windows)
    val_ds = TokenChunkDataset(val_docs, block_size, random_windows=False, interleave=True)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader
