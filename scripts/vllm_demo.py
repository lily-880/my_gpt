"""用 vLLM 跑官方 gpt2。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_gpt.vllm_guard import assert_official_hf_model

DEFAULT_PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "In a small village,",
    "Once upon a time",
    "The meaning of life is",
    "Python is a",
    "To be or not to be",
    "ROMEO:",
]


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def count_new_tokens(outputs) -> int:
    n = 0
    for o in outputs:
        n += len(o.outputs[0].token_ids)
    return n


def run_vllm(model: str, prompts: list[str], max_tokens: int, temperature: float, top_k: int, gpu_mem: float):
    from vllm import LLM, SamplingParams

    print(f"vllm_model={model}", flush=True)
    llm, load_s = timed(
        lambda: LLM(
            model=model,
            dtype="float16",
            gpu_memory_utilization=gpu_mem,
            max_model_len=1024,
            swap_space=1,
            enforce_eager=True,
            trust_remote_code=False,
        )
    )
    print(f"vllm_load_s={load_s:.1f}", flush=True)
    params = SamplingParams(temperature=temperature, top_k=top_k, max_tokens=max_tokens)

    # warmup
    llm.generate(prompts[:1], params, use_tqdm=False)

    single, single_s = timed(lambda: llm.generate(prompts[:1], params, use_tqdm=False))
    single_tok = count_new_tokens(single)
    print(
        f"vllm_single  prompts=1  new_tok={single_tok}  s={single_s:.3f}  tok/s={single_tok / max(single_s, 1e-9):.1f}",
        flush=True,
    )

    batch, batch_s = timed(lambda: llm.generate(prompts, params, use_tqdm=False))
    batch_tok = count_new_tokens(batch)
    print(
        f"vllm_batch   prompts={len(prompts)}  new_tok={batch_tok}  s={batch_s:.3f}  "
        f"tok/s={batch_tok / max(batch_s, 1e-9):.1f}",
        flush=True,
    )
    texts = [o.outputs[0].text for o in batch]
    return {
        "backend": "vllm",
        "load_s": load_s,
        "single_s": single_s,
        "single_tok": single_tok,
        "single_tok_s": single_tok / max(single_s, 1e-9),
        "batch_prompts": len(prompts),
        "batch_s": batch_s,
        "batch_tok": batch_tok,
        "batch_tok_s": batch_tok / max(batch_s, 1e-9),
        "samples": [p + t for p, t in zip(prompts, texts)],
    }


def run_hf(model: str, prompts: list[str], max_tokens: int, temperature: float, top_k: int, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hf, load_s = timed(lambda: AutoModelForCausalLM.from_pretrained(model).to(device).eval())
    print(f"hf_load_s={load_s:.1f}", flush=True)

    def generate_one(text: str):
        ids = tok(text, return_tensors="pt").to(device)
        out = hf.generate(
            **ids,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_k=top_k,
            pad_token_id=tok.eos_token_id,
        )
        return out

    generate_one(prompts[0])
    _, single_s = timed(lambda: generate_one(prompts[0]))
    print(f"hf_single  prompts=1  s={single_s:.3f}  tok/s~={max_tokens / max(single_s, 1e-9):.1f}", flush=True)

    def generate_seq():
        for p in prompts:
            generate_one(p)

    _, batch_s = timed(generate_seq)
    print(
        f"hf_seq     prompts={len(prompts)}  s={batch_s:.3f}  "
        f"tok/s~={len(prompts) * max_tokens / max(batch_s, 1e-9):.1f}",
        flush=True,
    )
    return {
        "backend": "hf_generate",
        "load_s": load_s,
        "single_s": single_s,
        "single_tok_s_approx": max_tokens / max(single_s, 1e-9),
        "batch_prompts": len(prompts),
        "seq_s": batch_s,
        "seq_tok_s_approx": len(prompts) * max_tokens / max(batch_s, 1e-9),
        "note": "HF 按条串行 generate，用来对照；不是公平的 kernel 对 kernel。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM 官方 gpt2 Demo")
    parser.add_argument(
        "--model",
        type=str,
        default=str(ROOT / "data" / "hf_gpt2") if (ROOT / "data" / "hf_gpt2" / "config.json").is_file() else "gpt2",
        help="官方 gpt2 的 HF id，或本地 snapshot 目录",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--no-hf", action="store_true", help="只跑 vLLM，不对照 HF")
    parser.add_argument("--hf-only", action="store_true", help="装不上 vLLM 时只跑 HF，并写明降级")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "assets" / "experiments" / "day11_vllm")
    args = parser.parse_args()

    model = assert_official_hf_model(args.model)
    prompts = list(DEFAULT_PROMPTS)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "model": model,
        "note": "官方 gpt2。未加载 shakespeare_perdoc/best.pt。",
        "max_tokens": args.max_tokens,
    }

    if not args.hf_only:
        try:
            import vllm  # noqa: F401
        except ImportError as e:
            print(f"vllm_import_failed: {e}", flush=True)
            print("降级：加 --hf-only 只跑 transformers.generate，或先 pip install vllm", flush=True)
            raise SystemExit(2) from e
        summary["vllm"] = run_vllm(
            model,
            prompts,
            args.max_tokens,
            args.temperature,
            args.top_k,
            args.gpu_memory_utilization,
        )

    if args.hf_only or not args.no_hf:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        summary["hf"] = run_hf(model, prompts, args.max_tokens, args.temperature, args.top_k, device)

    if "vllm" in summary and "hf" in summary:
        summary["batch_speedup_vs_hf_seq"] = summary["hf"]["seq_s"] / max(summary["vllm"]["batch_s"], 1e-9)

    print("\n===== summary =====", flush=True)
    printable = {k: v for k, v in summary.items() if k not in {"vllm", "hf"}}
    if "vllm" in summary:
        printable["vllm_batch_tok_s"] = summary["vllm"]["batch_tok_s"]
        printable["vllm_single_s"] = summary["vllm"]["single_s"]
    if "hf" in summary:
        printable["hf_seq_s"] = summary["hf"]["seq_s"]
    print(json.dumps(printable, indent=2), flush=True)

    samples = (summary.get("vllm") or {}).get("samples") or []
    dump = dict(summary)
    if "vllm" in dump and "samples" in dump["vllm"]:
        dump["vllm"] = {k: v for k, v in dump["vllm"].items() if k != "samples"}
    (args.out_dir / "metrics.json").write_text(json.dumps(dump, indent=2) + "\n", encoding="utf-8")
    if samples:
        lines = ["# vLLM 官方 gpt2 样例", "", "不是莎士比亚专家。", ""]
        for i, text in enumerate(samples, start=1):
            lines += [f"## {i}", "", "```", text, "```", ""]
        (args.out_dir / "samples.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
