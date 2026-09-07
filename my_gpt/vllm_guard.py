"""vLLM Demo 只允许官方 HF 模型名，防止误加载自研 .pt。"""

from __future__ import annotations

from pathlib import Path


def assert_official_hf_model(name: str) -> str:
    raw = name.strip()
    if not raw:
        raise ValueError("模型名不能为空")
    lower = raw.lower()
    path = Path(raw)
    if path.suffix in {".pt", ".pth"}:
        raise ValueError("vLLM Demo 不加载本地 checkpoint，请用官方模型名，例如 gpt2")
    if "shakespeare" in lower or "perdoc" in lower:
        raise ValueError("不要把莎士比亚 / 自研权重喂给 vLLM；只用官方 gpt2")
    if path.exists() and path.is_file():
        raise ValueError(f"{raw} 是本地文件。本 Demo 只接受 HuggingFace 模型 id 或官方 snapshot 目录")
    if path.exists() and path.is_dir():
        if not (path / "config.json").is_file():
            raise ValueError(f"{raw} 不是带 config.json 的 HF 目录")
        cfg = (path / "config.json").read_text(encoding="utf-8")
        if '"GPT2LMHeadModel"' not in cfg and '"gpt2"' not in cfg.lower():
            raise ValueError(f"{raw} 看起来不是官方 GPT-2 snapshot")
    return raw
