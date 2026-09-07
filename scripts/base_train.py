"""单卡预训练：AMP、梯度裁剪、验证 PPL、checkpoint、贪心采样。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import cycle
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_gpt.checkpoint import gpt_config_from_checkpoint, load_checkpoint, save_checkpoint
from my_gpt.dataloader import build_dataloaders, load_pretrain_documents
from my_gpt.gpt import GPT, GPTConfig
from my_gpt.optim import build_optimizer, cosine_lr, set_optimizer_lr
from my_gpt.tokenizer import GPT2TokenizerWrapper
from my_gpt.utils import (
    amp_dtype_and_scaler,
    autocast_context,
    evaluate_loss,
    perplexity_from_loss,
    pick_device,
    sample_generate,
    train_step,
)


def load_train_cfg(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def gpt_config_from_train_cfg(cfg: dict) -> GPTConfig:
    fields = {k: cfg[k] for k in GPTConfig.__dataclass_fields__ if k in cfg}
    return GPTConfig(**fields)


def load_train_state(ckpt_dir: Path) -> dict:
    path = ckpt_dir / "train_state.json"
    if not path.exists():
        return {"best_val_loss": None, "best_step": None, "bad_evals": 0, "history": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_train_state(ckpt_dir: Path, state: dict) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "train_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_train(
    cfg: dict,
    *,
    documents: list[tuple[str, str]],
    tokenizer,
    device: torch.device,
    ckpt_dir: Path,
    prompt: str = "ROMEO:",
    resume: Path | None = None,
    use_amp: bool = True,
) -> dict:
    gpt_cfg = gpt_config_from_checkpoint(resume) if resume is not None else gpt_config_from_train_cfg(cfg)
    train_loader, val_loader = build_dataloaders(
        tokenizer,
        documents=documents,
        block_size=gpt_cfg.block_size,
        batch_size=int(cfg["batch_size"]),
        random_windows=bool(cfg.get("random_windows", False)),
        split=str(cfg.get("split", "per_doc")),
        split_seed=int(cfg.get("split_seed", 42)),
    )
    if len(train_loader) == 0:
        raise ValueError("训练集为空，把文本加长或减小 block_size")

    model = GPT(gpt_cfg).to(device)
    optimizer = build_optimizer(model, float(cfg["learning_rate"]), float(cfg["weight_decay"]))
    start_step = 0
    if resume is not None:
        payload = load_checkpoint(resume, model=model, optimizer=optimizer, map_location=device)
        start_step = int(payload.get("step", 0))
        print(f"resume {resume}  from_step={start_step}")
    if use_amp:
        amp_dtype, scaler = amp_dtype_and_scaler(device)
    else:
        amp_dtype, scaler = None, None
    autocast_ctx = autocast_context(device, amp_dtype)
    print(f"device={device}  amp={amp_dtype}  scaler={scaler is not None}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    batches = cycle(train_loader)
    last_metrics: dict = {}
    max_steps = int(cfg["max_steps"])
    if start_step >= max_steps:
        raise ValueError(
            f"checkpoint 已是 step={start_step}，--max-steps={max_steps} 不会再往前走。"
            "把 --max-steps 设成比当前步数更大的目标总步数，例如 2000。"
        )
    grad_clip = float(cfg.get("grad_clip", 1.0))
    base_lr = float(cfg["learning_rate"])
    min_lr = float(cfg.get("min_lr", base_lr))
    schedule_steps = int(cfg.get("lr_schedule_steps", max_steps))
    warmup_steps = int(cfg.get("warmup_steps", 0))
    use_cosine = str(cfg.get("lr_decay", "none")).lower() == "cosine"
    patience = int(cfg.get("patience", 0))
    min_eval_step = int(cfg.get("min_eval_step", 0))
    state = load_train_state(ckpt_dir)
    stopped_early = False

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    for step in range(start_step + 1, max_steps + 1):
        if use_cosine:
            set_optimizer_lr(
                optimizer,
                cosine_lr(
                    step,
                    base_lr=base_lr,
                    min_lr=min_lr,
                    schedule_steps=schedule_steps,
                    warmup_steps=warmup_steps,
                ),
            )
        x, y = next(batches)
        x, y = x.to(device), y.to(device)
        train_loss = train_step(
            model,
            optimizer,
            x,
            y,
            scaler=scaler,
            grad_clip=grad_clip,
            autocast_ctx=autocast_ctx,
        )

        if step == 1 or step % int(cfg.get("log_every", 20)) == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"step {step:05d}  train_loss={train_loss:.4f}  "
                f"train_ppl={perplexity_from_loss(train_loss):.2f}  lr={lr_now:.2e}",
                flush=True,
            )

        if int(cfg.get("eval_every", 0)) > 0 and step % int(cfg["eval_every"]) == 0:
            val_loss = evaluate_loss(
                model,
                val_loader,
                device,
                autocast_ctx,
                max_batches=int(cfg.get("eval_batches", 20)),
            )
            val_ppl = perplexity_from_loss(val_loss)
            last_metrics = {"step": step, "train_loss": train_loss, "val_loss": val_loss, "val_ppl": val_ppl}
            print(f"step {step:05d}  val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}", flush=True)
            state["history"].append(
                {"step": step, "train_loss": float(train_loss), "val_loss": float(val_loss), "val_ppl": float(val_ppl)}
            )
            best = state.get("best_val_loss")
            if best is None or val_loss < float(best) - 1e-4:
                state["best_val_loss"] = float(val_loss)
                state["best_step"] = step
                state["bad_evals"] = 0
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, step=step, config=gpt_cfg)
                print(f"saved best.pt  val_loss={val_loss:.4f}  step={step}", flush=True)
            else:
                state["bad_evals"] = int(state.get("bad_evals") or 0) + 1
                print(f"no improve  bad_evals={state['bad_evals']}/{patience or 'off'}", flush=True)
            save_train_state(ckpt_dir, state)
            if (
                patience > 0
                and step >= min_eval_step
                and int(state["bad_evals"]) >= patience
            ):
                print(f"early_stop  step={step}  best_step={state['best_step']}", flush=True)
                stopped_early = True
                max_steps = step
                break

        if int(cfg.get("sample_every", 0)) > 0 and step % int(cfg["sample_every"]) == 0:
            ids = torch.tensor([tokenizer.encode(prompt) or [tokenizer.eos_id]], device=device)
            temperature = float(cfg.get("sample_temperature", 0.8))
            top_k = int(cfg.get("sample_top_k", 50))
            out = sample_generate(
                model,
                ids,
                int(cfg.get("max_new_tokens", 40)),
                gpt_cfg.block_size,
                temperature=temperature,
                top_k=top_k,
            )
            print("sample:", tokenizer.decode(out[0].tolist()), flush=True)

        if int(cfg.get("save_every", 0)) > 0 and step % int(cfg["save_every"]) == 0:
            save_checkpoint(ckpt_dir / f"step_{step:06d}.pt", model, optimizer, step=step, config=gpt_cfg)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    n_done = max_steps - start_step
    print(f"wall={elapsed:.1f}s  steps={n_done}  step/s={n_done / max(elapsed, 1e-9):.2f}")
    if device.type == "cuda":
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"peak_mem_MiB={peak_mib:.1f}")

    save_checkpoint(ckpt_dir / "last.pt", model, optimizer, step=max_steps, config=gpt_cfg)
    last_metrics.setdefault("step", max_steps)
    last_metrics.setdefault("train_loss", train_loss)
    last_metrics["wall_s"] = elapsed
    last_metrics["steps_per_s"] = n_done / max(elapsed, 1e-9)
    last_metrics["early_stop"] = stopped_early
    last_metrics["best_step"] = state.get("best_step")
    last_metrics["best_val_loss"] = state.get("best_val_loss")
    save_train_state(ckpt_dir, state)
    return last_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="单卡预训练（Day 5）")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "shakespeare_perdoc.json")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--ckpt-dir", type=Path, default=ROOT / "checkpoints" / "base")
    parser.add_argument("--data-file", type=Path, default=None, help="覆盖默认莎士比亚全集")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 config 里的 batch_size")
    parser.add_argument("--resume", type=Path, default=None, help="从已有 .pt 接着练；max_steps 是目标总步数")
    parser.add_argument("--no-amp", action="store_true", help="关掉 autocast，全程 fp32（Day 7 AMP 消融）")
    args = parser.parse_args()

    cfg = load_train_cfg(args.config)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    if args.data_file is not None:
        documents = [(args.data_file.name, args.data_file.read_text(encoding="utf-8"))]
    else:
        documents = load_pretrain_documents()

    tokenizer = GPT2TokenizerWrapper()
    cfg["vocab_size"] = tokenizer.vocab_size
    device = pick_device(args.device)
    run_train(
        cfg,
        documents=documents,
        tokenizer=tokenizer,
        device=device,
        ckpt_dir=args.ckpt_dir,
        resume=args.resume,
        use_amp=not args.no_amp,
    )
    


if __name__ == "__main__":
    main()
