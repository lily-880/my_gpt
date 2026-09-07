"""对比无 cache 整段重算 vs KV-Cache。默认打在莎士比亚专家上，不改权重。"""

from __future__ import annotations

import argparse
import json
import sys
import time
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


def timed(fn, *, sync: bool) -> float:
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if sync:
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="KV-Cache vs 整段重算")
    parser.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "shakespeare_perdoc" / "best.pt")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "assets" / "experiments" / "day10_kvcache")
    args = parser.parse_args()

    device = pick_device(args.device)
    cfg = gpt_config_from_checkpoint(args.ckpt, map_location=device)
    model = GPT(cfg).to(device)
    load_checkpoint(args.ckpt, model=model, map_location=device)
    model.eval()
    tokenizer = GPT2TokenizerWrapper()
    prompt_ids = tokenizer.encode(args.prompt) or [tokenizer.eos_id]
    idx = torch.tensor([prompt_ids], device=device)
    n = args.max_new_tokens
    sync = device.type == "cuda"

    print(f"ckpt={args.ckpt}  device={device}  new_tokens={n}  greedy 对照", flush=True)

    # 正确性：greedy 必须一致
    no_cache = sample_generate(model, idx.clone(), n, cfg.block_size, temperature=0.0)
    cached = Engine(model, temperature=0.0, top_k=None).generate(idx.clone(), n)
    match = bool(torch.equal(no_cache, cached))
    print(f"greedy_match={match}", flush=True)
    if not match:
        raise RuntimeError("KV-Cache greedy 与整段重算不一致")

    def run_no_cache():
        sample_generate(model, idx.clone(), n, cfg.block_size, temperature=0.0)

    def run_cache():
        Engine(model, temperature=0.0, top_k=None).generate(idx.clone(), n)

    warmup = timed(run_no_cache, sync=sync)
    no_s = timed(run_no_cache, sync=sync)
    _ = timed(run_cache, sync=sync)
    yes_s = timed(run_cache, sync=sync)
    print(
        f"no_cache={no_s:.3f}s  ({n / no_s:.1f} tok/s)  "
        f"kvcache={yes_s:.3f}s  ({n / yes_s:.1f} tok/s)  "
        f"speedup={no_s / max(yes_s, 1e-9):.2f}x  warmup_no_cache={warmup:.3f}s",
        flush=True,
    )
    print("sample:", tokenizer.decode(cached[0].tolist()), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "ckpt": str(args.ckpt),
        "device": str(device),
        "max_new_tokens": n,
        "greedy_match": match,
        "no_cache_s": no_s,
        "kvcache_s": yes_s,
        "speedup": no_s / max(yes_s, 1e-9),
        "no_cache_tok_s": n / no_s,
        "kvcache_tok_s": n / yes_s,
        "note": "单条 greedy。cache 是 (B,H,T,D) 按步 cat，不是 PagedAttention。",
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
