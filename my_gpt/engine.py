"""带 KV-Cache 的生成：prefill 一次，之后每步只送新 token。满窗后整窗重算，与无 cache 滑窗对齐。"""

from __future__ import annotations

import torch
import torch.nn as nn

from my_gpt.utils import sample_next_id


def _cache_len(cache: list[tuple[torch.Tensor, torch.Tensor]] | None) -> int:
    if not cache:
        return 0
    return int(cache[0][0].size(2))


class Engine:
    """cache 形状每层 (B, n_head, T, head_dim)，按步 cat，不是 vLLM 的分页。

    注意力窗口最长 block_size。超过以后丢掉 cache、对最近 block_size 个 token 再 prefill，
    和 `sample_generate` 的 `idx[:, -block_size:]` 一致（RoPE 按新窗口从 0 算）。
    返回序列保留完整 prompt，即使 prompt 长于 block_size。
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
    ):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def _sample(self, logits: torch.Tensor, past_ids: torch.Tensor) -> torch.Tensor:
        return sample_next_id(
            logits[:, -1, :],
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            past_ids=past_ids,
        )

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        block = int(self.model.config.block_size)
        n = int(max_new_tokens)
        if n <= 0 or idx.size(1) == 0:
            if was_training:
                self.model.train()
            return idx

        out = idx
        logits, cache = self.model.forward_kv(out[:, -block:])
        next_id = self._sample(logits, out)
        out = torch.cat([out, next_id], dim=1)
        for _ in range(n - 1):
            if _cache_len(cache) >= block:
                logits, cache = self.model.forward_kv(out[:, -block:])
            else:
                logits, cache = self.model.forward_kv(next_id, cache=cache)
            next_id = self._sample(logits, out)
            out = torch.cat([out, next_id], dim=1)
        if was_training:
            self.model.train()
        return out
