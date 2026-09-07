"""莎士比亚专家多轮续写。把上文拼进同一条上下文，生成内部走 KV-Cache。"""

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
from my_gpt.utils import pick_device


def main() -> None:
    parser = argparse.ArgumentParser(description="多轮续写（历史拼进上下文）")
    parser.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "shakespeare_perdoc" / "best.pt")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = pick_device(args.device)
    cfg = gpt_config_from_checkpoint(args.ckpt, map_location=device)
    model = GPT(cfg).to(device)
    load_checkpoint(args.ckpt, model=model, map_location=device)
    model.eval()
    tokenizer = GPT2TokenizerWrapper()
    engine = Engine(model, temperature=args.temperature, top_k=args.top_k)
    history: list[int] = []
    print(
        f"ckpt={args.ckpt}  device={device}  cache=on  temp={args.temperature}  "
        f"ctx={cfg.block_size}  /reset 清空  空行退出",
        flush=True,
    )
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            break
        if not text:
            break
        if text == "/reset":
            history = []
            print("(cleared)", flush=True)
            continue
        user_ids = tokenizer.encode(text)
        if not user_ids:
            continue
        if history:
            history.extend(tokenizer.encode("\n") or [])
        history.extend(user_ids)
        idx = torch.tensor([history], device=device)
        out = engine.generate(idx, args.max_new_tokens)
        new_ids = out[0, idx.size(1) :].tolist()
        history.extend(new_ids)
        print(tokenizer.decode(new_ids), flush=True)
        if len(history) > cfg.block_size:
            print(f"(history={len(history)} tokens，注意力只看最近 {cfg.block_size})", flush=True)


if __name__ == "__main__":
    main()
