"""单卡预训练：AMP、梯度裁剪、验证 PPL、checkpoint、贪心采样。"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import cycle
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_gpt.checkpoint import gpt_config_from_checkpoint, load_checkpoint, save_checkpoint
from my_gpt.dataloader import build_dataloaders, load_pretrain_text
from my_gpt.gpt import GPT, GPTConfig
from my_gpt.optim import build_optimizer
from my_gpt.tokenizer import GPT2TokenizerWrapper
from my_gpt.utils import (
    amp_dtype_and_scaler,
    autocast_context,
    evaluate_loss,
    greedy_generate,
    perplexity_from_loss,
    train_step,
)


def load_train_cfg(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def gpt_config_from_train_cfg(cfg: dict) -> GPTConfig:
    fields = {k: cfg[k] for k in GPTConfig.__dataclass_fields__ if k in cfg}
    return GPTConfig(**fields)


def pick_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 cuda 但当前没有 GPU")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_train(
    cfg: dict,
    *,
    text: str,
    tokenizer,
    device: torch.device,
    ckpt_dir: Path,
    prompt: str = "ROMEO:",
    resume: Path | None = None,
) -> dict:
    gpt_cfg = gpt_config_from_checkpoint(resume) if resume is not None else gpt_config_from_train_cfg(cfg)
    train_loader, val_loader = build_dataloaders(
        tokenizer,
        text,
        block_size=gpt_cfg.block_size,
        batch_size=int(cfg["batch_size"]),
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
    amp_dtype, scaler = amp_dtype_and_scaler(device)
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

    for step in range(start_step + 1, max_steps + 1):
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
            print(f"step {step:05d}  train_loss={train_loss:.4f}  train_ppl={perplexity_from_loss(train_loss):.2f}")

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
            print(f"step {step:05d}  val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}")

        if int(cfg.get("sample_every", 0)) > 0 and step % int(cfg["sample_every"]) == 0:
            ids = torch.tensor([tokenizer.encode(prompt) or [tokenizer.eos_id]], device=device)
            out = greedy_generate(model, ids, int(cfg.get("max_new_tokens", 40)), gpt_cfg.block_size)
            print("sample:", tokenizer.decode(out[0].tolist()))

        if int(cfg.get("save_every", 0)) > 0 and step % int(cfg["save_every"]) == 0:
            save_checkpoint(ckpt_dir / f"step_{step:06d}.pt", model, optimizer, step=step, config=gpt_cfg)

    save_checkpoint(ckpt_dir / "last.pt", model, optimizer, step=max_steps, config=gpt_cfg)
    last_metrics.setdefault("step", max_steps)
    last_metrics.setdefault("train_loss", train_loss)
    return last_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="单卡预训练（Day 5）")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.json")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--ckpt-dir", type=Path, default=ROOT / "checkpoints" / "base")
    parser.add_argument("--data-file", type=Path, default=None, help="覆盖默认莎士比亚全集")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="从已有 .pt 接着练；max_steps 是目标总步数")
    args = parser.parse_args()

    cfg = load_train_cfg(args.config)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps

    if args.data_file is not None:
        text = args.data_file.read_text(encoding="utf-8")
    else:
        text = load_pretrain_text()

    tokenizer = GPT2TokenizerWrapper()
    cfg["vocab_size"] = tokenizer.vocab_size
    device = pick_device(args.device)
    run_train(
        cfg,
        text=text,
        tokenizer=tokenizer,
        device=device,
        ckpt_dir=args.ckpt_dir,
        resume=args.resume,
    )
    


if __name__ == "__main__":
    main()
