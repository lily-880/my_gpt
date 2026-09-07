"""从 checkpoint 抽样生成。训练日志里的 greedy 复读不能代表模型上限。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_gpt.checkpoint import gpt_config_from_checkpoint, load_checkpoint
from my_gpt.engine import Engine
from my_gpt.gpt import GPT
from my_gpt.tokenizer import GPT2TokenizerWrapper
from my_gpt.utils import pick_device, sample_generate


def main() -> None:
    parser = argparse.ArgumentParser(description="从 .pt 抽样生成")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--n", type=int, default=3, help="生成几条")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cache", action="store_true", help="用 KV-Cache，只算新 token")
    args = parser.parse_args()

    device = pick_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    cfg = gpt_config_from_checkpoint(args.ckpt, map_location=device)
    model = GPT(cfg).to(device)
    load_checkpoint(args.ckpt, model=model, map_location=device)
    model.eval()
    tokenizer = GPT2TokenizerWrapper()
    prompt_ids = tokenizer.encode(args.prompt) or [tokenizer.eos_id]

    print(f"ckpt={args.ckpt}  temp={args.temperature}  top_k={args.top_k}  cache={args.cache}")
    engine = Engine(model, temperature=args.temperature, top_k=args.top_k) if args.cache else None
    for i in range(args.n):
        idx = torch.tensor([prompt_ids], device=device)
        if engine is not None:
            out = engine.generate(idx, args.max_new_tokens)
        else:
            out = sample_generate(
                model,
                idx,
                args.max_new_tokens,
                cfg.block_size,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        print(f"\n===== sample {i + 1} =====")
        print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
