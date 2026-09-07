"""CPU 动态 INT8，打在莎士比亚 best.pt 上。"""

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
from my_gpt.dataloader import build_dataloaders, load_pretrain_documents
from my_gpt.gpt import GPT
from my_gpt.quant import (
    n_plain_linears,
    quantize_linear_dynamic,
    state_dict_serialized_bytes,
    tensor_nbytes,
    time_generate,
    time_generate_kv,
)
from my_gpt.tokenizer import GPT2TokenizerWrapper
from my_gpt.engine import Engine
from my_gpt.utils import evaluate_loss, perplexity_from_loss, sample_generate


def load_train_cfg(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def mib(n: int) -> float:
    return n / (1024**2)


def eval_one(
    name: str,
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    *,
    decode,
    max_batches: int,
    prompt_ids: list[int],
    block_size: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    n_samples: int,
    seed: int,
) -> dict:
    t_eval = time.perf_counter()
    val_loss = evaluate_loss(model, val_loader, device, max_batches=max_batches)
    val_ppl = perplexity_from_loss(val_loss)
    eval_s = time.perf_counter() - t_eval
    print(f"{name}  val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}  eval_s={eval_s:.1f}", flush=True)

    idx = torch.tensor([prompt_ids], device=device)
    gen_s = time_generate(
        model,
        idx,
        max_new_tokens,
        block_size,
        temperature=temperature,
        top_k=top_k,
        warmup=1,
        repeats=1,
    )
    tok_s = max_new_tokens / max(gen_s, 1e-9)
    print(f"{name}  gen_s={gen_s:.2f}  tok/s={tok_s:.2f}  (warmup 后 1 次, {max_new_tokens} tokens)", flush=True)

    torch.manual_seed(seed)
    samples = []
    for i in range(n_samples):
        out = sample_generate(
            model,
            idx.clone(),
            max_new_tokens,
            block_size,
            temperature=temperature,
            top_k=top_k,
        )
        text = decode(out[0].tolist())
        samples.append(text)
        print(f"\n===== {name} sample {i + 1} =====", flush=True)
        print(text, flush=True)

    ser = state_dict_serialized_bytes(model)
    ten = tensor_nbytes(model)
    print(
        f"{name}  state_dict={mib(ser):.1f} MiB  tensor_bytes={mib(ten):.1f} MiB  "
        f"plain_linear={n_plain_linears(model)}",
        flush=True,
    )
    return {
        "name": name,
        "val_loss": float(val_loss),
        "val_ppl": float(val_ppl),
        "eval_s": eval_s,
        "gen_s": gen_s,
        "tok_per_s": tok_s,
        "state_dict_mib": mib(ser),
        "tensor_mib": mib(ten),
        "plain_linear": n_plain_linears(model),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="莎士比亚专家 CPU INT8 PTQ 对比")
    parser.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "shakespeare_perdoc" / "best.pt")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "shakespeare_perdoc.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "assets" / "experiments" / "day9_int8")
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-eval", action="store_true", help="只量化并抽样，不算 PPL")
    args = parser.parse_args()

    if not args.ckpt.is_file():
        raise FileNotFoundError(f"找不到 {args.ckpt}，不要用别的目录覆盖专家权重")

    cfg = load_train_cfg(args.config)
    eval_batches = int(args.eval_batches if args.eval_batches is not None else cfg.get("eval_batches", 20))
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 8))

    device = torch.device("cpu")
    gpt_cfg = gpt_config_from_checkpoint(args.ckpt, map_location=device)
    model = GPT(gpt_cfg).to(device)
    load_checkpoint(args.ckpt, model=model, map_location=device)
    model.eval()
    print(
        f"ckpt={args.ckpt}  device=cpu  linears={n_plain_linears(model)}  "
        f"（动态量化 nn.Linear；不是 QAT，不是 GPU INT8）",
        flush=True,
    )

    tokenizer = GPT2TokenizerWrapper()
    prompt_ids = tokenizer.encode(args.prompt) or [tokenizer.eos_id]

    val_loader = None
    if not args.skip_eval:
        documents = load_pretrain_documents()
        _, val_loader = build_dataloaders(
            tokenizer,
            documents=documents,
            block_size=gpt_cfg.block_size,
            batch_size=batch_size,
            random_windows=bool(cfg.get("random_windows", False)),
            split=str(cfg.get("split", "per_doc")),
            split_seed=int(cfg.get("split_seed", 42)),
        )
        print(
            f"val  split={cfg.get('split', 'per_doc')}  seed={cfg.get('split_seed', 42)}  "
            f"batch={batch_size}  eval_batches={eval_batches}",
            flush=True,
        )

    common = dict(
        val_loader=val_loader,
        device=device,
        decode=tokenizer.decode,
        max_batches=eval_batches,
        prompt_ids=prompt_ids,
        block_size=gpt_cfg.block_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        n_samples=args.n,
        seed=args.seed,
    )

    if args.skip_eval:
        fp32 = {
            "name": "fp32",
            "state_dict_mib": mib(state_dict_serialized_bytes(model)),
            "tensor_mib": mib(tensor_nbytes(model)),
            "plain_linear": n_plain_linears(model),
            "samples": [],
        }
        print(f"fp32  skip_eval  state_dict={fp32['state_dict_mib']:.1f} MiB", flush=True)
        torch.manual_seed(args.seed)
        idx = torch.tensor([prompt_ids], device=device)
        for i in range(args.n):
            out = sample_generate(
                model, idx.clone(), args.max_new_tokens, gpt_cfg.block_size,
                temperature=args.temperature, top_k=args.top_k,
            )
            fp32["samples"].append(tokenizer.decode(out[0].tolist()))
    else:
        fp32 = eval_one("fp32", model, **common)

    model.eval()
    t_q = time.perf_counter()
    qmodel = quantize_linear_dynamic(model)
    qmodel.eval()
    print(f"quantized  in {time.perf_counter() - t_q:.1f}s  leftover_linear={n_plain_linears(qmodel)}", flush=True)

    if args.skip_eval:
        int8 = {
            "name": "int8",
            "state_dict_mib": mib(state_dict_serialized_bytes(qmodel)),
            "tensor_mib": mib(tensor_nbytes(qmodel)),
            "plain_linear": n_plain_linears(qmodel),
            "samples": [],
        }
        torch.manual_seed(args.seed)
        idx = torch.tensor([prompt_ids], device=device)
        for i in range(args.n):
            out = sample_generate(
                qmodel, idx.clone(), args.max_new_tokens, gpt_cfg.block_size,
                temperature=args.temperature, top_k=args.top_k,
            )
            int8["samples"].append(tokenizer.decode(out[0].tolist()))
            print(f"\n===== int8 sample {i + 1} =====", flush=True)
            print(int8["samples"][-1], flush=True)
    else:
        int8 = eval_one("int8", qmodel, **common)

    summary = {
        "ckpt": str(args.ckpt),
        "note": "CPU 动态 PTQ，只量化 Linear。不覆盖专家 ckpt。不是 GPU Tensor Core INT8。",
        "fp32": {k: v for k, v in fp32.items() if k != "samples"},
        "int8": {k: v for k, v in int8.items() if k != "samples"},
        "size_ratio_fp32_over_int8": fp32["state_dict_mib"] / max(int8["state_dict_mib"], 1e-9),
    }
    if "val_ppl" in fp32 and "val_ppl" in int8:
        summary["ppl_delta"] = int8["val_ppl"] - fp32["val_ppl"]
        summary["gen_speedup"] = fp32.get("gen_s", 0) / max(int8.get("gen_s") or 1e-9, 1e-9)

    print("\n===== summary =====", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"fp32", "int8"}}, indent=2), flush=True)
    print(
        f"size  fp32={fp32['state_dict_mib']:.1f} MiB  int8={int8['state_dict_mib']:.1f} MiB  "
        f"ratio={summary['size_ratio_fp32_over_int8']:.2f}x",
        flush=True,
    )
    if "val_ppl" in fp32:
        print(
            f"ppl   fp32={fp32['val_ppl']:.2f}  int8={int8['val_ppl']:.2f}  "
            f"delta={summary['ppl_delta']:+.2f}",
            flush=True,
        )
        print(
            f"gen   fp32={fp32['gen_s']:.2f}s  int8={int8['gen_s']:.2f}s  "
            f"speedup={summary['gen_speedup']:.2f}x",
            flush=True,
        )

    idx = torch.tensor([prompt_ids], device=device)
    torch.manual_seed(args.seed)
    no_cache_q = sample_generate(
        qmodel, idx.clone(), args.max_new_tokens, gpt_cfg.block_size, temperature=0.0
    )
    torch.manual_seed(args.seed)
    cache_q = Engine(qmodel, temperature=0.0, top_k=None).generate(idx.clone(), args.max_new_tokens)
    kv_match = bool(torch.equal(no_cache_q, cache_q))
    print(f"int8  kvcache_greedy_match={kv_match}", flush=True)
    if not kv_match:
        print(
            "note: 动态量化每次 forward 按当前激活范围缩放，"
            "逐步 decode 和整窗重算不必逐 token 相同。这是 PTQ 行为，不是 cache 写错。",
            flush=True,
        )
    fp32_kv_s = time_generate_kv(
        model, idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, warmup=1, repeats=1
    )
    int8_kv_s = time_generate_kv(
        qmodel, idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, warmup=1, repeats=1
    )
    summary["int8_kvcache_greedy_match"] = kv_match
    summary["fp32_kv_s"] = fp32_kv_s
    summary["int8_kv_s"] = int8_kv_s
    print(
        f"kv    fp32={fp32_kv_s:.2f}s  int8={int8_kv_s:.2f}s  "
        f"(Engine，{args.max_new_tokens} tokens)",
        flush=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    samples_md = [
        f"# INT8 对照样例（prompt={args.prompt!r}, temp={args.temperature}, top_k={args.top_k}, seed={args.seed})",
        "",
        "同一 seed 各抽。量化会改变 logits，文本不必相同。不要 greedy。",
        "",
    ]
    for name, blob in (("fp32", fp32), ("int8", int8)):
        for i, text in enumerate(blob.get("samples") or [], start=1):
            samples_md.append(f"## {name} sample {i}")
            samples_md.append("")
            samples_md.append("```")
            samples_md.append(text)
            samples_md.append("```")
            samples_md.append("")
    (args.out_dir / "samples.md").write_text("\n".join(samples_md), encoding="utf-8")
    print(f"wrote {args.out_dir / 'metrics.json'}  {args.out_dir / 'samples.md'}", flush=True)
    print("专家权重未改动：checkpoints/shakespeare_perdoc/best.pt", flush=True)


if __name__ == "__main__":
    main()
