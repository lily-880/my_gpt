"""GPT-2 BPE。词表文件放在仓库里，不连 HuggingFace（transformers 5 在离线时会静默给出空词表）。"""

from __future__ import annotations

from pathlib import Path

import regex as re

_TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "assets" / "gpt2_tokenizer"
_MERGES_FILE = _TOKENIZER_DIR / "merges.txt"
_GPT2_SPLIT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


class GPT2TokenizerWrapper:
    def __init__(self, name: str = "gpt2"):
        if name != "gpt2":
            raise ValueError("目前只内置 gpt2 BPE")
        if not _MERGES_FILE.exists():
            raise FileNotFoundError(f"缺少 GPT-2 merges：{_MERGES_FILE}")

        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges: list[tuple[str, str]] = []
        for line in _MERGES_FILE.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#version"):
                continue
            parts = line.split(" ")
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
        if len(merges) != 50000:
            raise ValueError(f"GPT-2 merges 应为 50000 条，实际 {len(merges)}")

        encoder = {ch: i for i, ch in enumerate(self.byte_encoder.values())}
        for a, b in merges:
            encoder[a + b] = len(encoder)
        encoder["<|endoftext|>"] = len(encoder)
        self.encoder = encoder
        self.decoder = {i: t for t, i in encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self._cache: dict[str, str] = {}

        self.eos_id = encoder["<|endoftext|>"]
        self.bos_id = self.eos_id
        self.pad_id = self.eos_id
        self.vocab_size = len(encoder)
        if self.vocab_size != 50257:
            raise ValueError(f"GPT-2 词表应为 50257，实际 {self.vocab_size}")

    def _bpe(self, token: str) -> str:
        if token in self._cache:
            return self._cache[token]
        word = tuple(token)
        pairs = _get_pairs(word)
        if not pairs:
            self._cache[token] = token
            return token
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        out = " ".join(word)
        self._cache[token] = out
        return out

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        for token in _GPT2_SPLIT.findall(text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            for piece in self._bpe(token).split(" "):
                ids.append(self.encoder[piece])
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        text = "".join(self.decoder[int(i)] for i in ids)
        return bytearray(self.byte_decoder[c] for c in text).decode("utf-8", errors="replace")

    def encode_documents(self, texts: list[str]) -> list[int]:
        """多篇文档拼成一条 token 流，每篇末尾加 eos，方便语言模型学会「一篇结束」。"""
        out: list[int] = []
        for text in texts:
            out.extend(self.encode(text, add_eos=True))
        return out
