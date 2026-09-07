"""冻结 HF gpt2 → 本仓库 GPT。ckpt 写到单独目录。"""

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
from my_gpt.distill import distill_loss, evaluate_ce_from_logits_fn, load_gpt2_teacher
from my_gpt.gpt import GPT
from my_gpt.optim import build_optimizer
from my_gpt.tokenizer import GPT2TokenizerWrapper
from my_gpt.utils import (
    amp_dtype_and_scaler,
    autocast_context,
    perplexity_from_loss,
    pick_device,
    sample_generate,
)


def load_train_cfg(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_train_state(ckpt_dir: Path) -> dict:
    path = ckpt_dir / "train_state.json"
    if not path.exists():
        return {"best_val_loss": None, "best_step": None, "bad_evals": 0, "history": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_train_state(ckpt_dir: Path, state: dict) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "train_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_distill(
    cfg: dict,
    *,
    documents: list[tuple[str, str]],
    tokenizer,
    device: torch.device,
    ckpt_dir: Path,
    student_ckpt: Path,
    teacher_name: str,
    resume: Path | None = None,
    eval_only: bool = False,
    use_amp: bool = True,
    prompt: str = "ROMEO:",
) -> dict:
    gpt_cfg = gpt_config_from_checkpoint(student_ckpt)
    if int(gpt_cfg.vocab_size) != int(tokenizer.vocab_size):
        raise ValueError(
            f"Student vocab={gpt_cfg.vocab_size} 与 tokenizer {tokenizer.vocab_size} 不一致，无法对 GPT-2 做 KL。"
        )
    train_loader, val_loader = build_dataloaders(
        tokenizer,
        documents=documents,
        block_size=gpt_cfg.block_size,
        batch_size=int(cfg["batch_size"]),
        random_windows=bool(cfg.get("random_windows", False)),
        split=str(cfg.get("split", "per_doc")),
        split_seed=int(cfg.get("split_seed", 42)),
    )

    student = GPT(gpt_cfg).to(device)
    load_checkpoint(student_ckpt, model=student, optimizer=None, map_location=device)
    print(f"student_init {student_ckpt}")

    if use_amp:
        amp_dtype, scaler = amp_dtype_and_scaler(device)
    else:
        amp_dtype, scaler = None, None
    autocast_ctx = autocast_context(device, amp_dtype)
    teacher_dtype = amp_dtype if amp_dtype is not None else torch.float32
    teacher = load_gpt2_teacher(teacher_name, device, teacher_dtype)
    print(f"teacher={teacher_name}  frozen  amp={amp_dtype}  scaler={scaler is not None}")

    def student_logits_fn(x):
        logits, _ = student(x)
        return logits

    def teacher_logits_fn(x):
        return teacher(input_ids=x, use_cache=False).logits

    max_eval = int(cfg.get("eval_batches", 20))
    student.eval()
    student_ce = evaluate_ce_from_logits_fn(
        student_logits_fn, val_loader, device, autocast_ctx, max_eval
    )
    teacher_ce = evaluate_ce_from_logits_fn(
        teacher_logits_fn, val_loader, device, autocast_ctx, max_eval
    )
    print(
        f"before  student_val_ce={student_ce:.4f}  student_ppl={perplexity_from_loss(student_ce):.2f}  "
        f"teacher_val_ce={teacher_ce:.4f}  teacher_ppl={perplexity_from_loss(teacher_ce):.2f}",
        flush=True,
    )
    if teacher_ce > student_ce:
        print(
            "note: Teacher val 比 Student 差。ckpt 写在本目录，不动 shakespeare_perdoc/best.pt。",
            flush=True,
        )

    if eval_only:
        return {
            "student_val_ce": student_ce,
            "teacher_val_ce": teacher_ce,
            "student_ppl": perplexity_from_loss(student_ce),
            "teacher_ppl": perplexity_from_loss(teacher_ce),
        }

    optimizer = build_optimizer(student, float(cfg["learning_rate"]), float(cfg["weight_decay"]))
    start_step = 0
    if resume is not None:
        payload = load_checkpoint(resume, model=student, optimizer=optimizer, map_location=device)
        start_step = int(payload.get("step", 0))
        print(f"resume {resume}  from_step={start_step}")

    tau = float(cfg["temperature"])
    alpha = float(cfg["alpha"])
    print(f"distill  temperature={tau}  alpha={alpha}  L=α T² KL + (1-α) CE")
    if alpha == 0.0:
        print("alpha=0：本步不算老师，只继续硬标签 CE（对照「多训几步」）。", flush=True)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    batches = cycle(train_loader)
    max_steps = int(cfg["max_steps"])
    if start_step >= max_steps:
        raise ValueError(f"checkpoint 已是 step={start_step}，把 --max-steps 设得更大。")
    grad_clip = float(cfg.get("grad_clip", 1.0))
    patience = int(cfg.get("patience", 0))
    min_eval_step = int(cfg.get("min_eval_step", 0))
    state = load_train_state(ckpt_dir)
    stopped_early = False
    last_metrics: dict = {
        "student_ppl_before": perplexity_from_loss(student_ce),
        "teacher_ppl": perplexity_from_loss(teacher_ce),
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    for step in range(start_step + 1, max_steps + 1):
        x, y = next(batches)
        x, y = x.to(device), y.to(device)
        student.train()
        optimizer.zero_grad(set_to_none=True)
        if alpha == 0.0:
            with autocast_ctx:
                _, loss = student(x, y)
            stats = {"loss": float(loss.detach()), "ce": float(loss.detach()), "kl": 0.0, "kd": 0.0}
        else:
            with torch.no_grad():
                with autocast_ctx:
                    t_logits = teacher(input_ids=x, use_cache=False).logits
            with autocast_ctx:
                s_logits, _ = student(x)
                loss, stats = distill_loss(
                    s_logits, t_logits, y, temperature=tau, alpha=alpha
                )
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optimizer.step()

        if step == 1 or step % int(cfg.get("log_every", 20)) == 0:
            print(
                f"step {step:05d}  loss={stats['loss']:.4f}  ce={stats['ce']:.4f}  "
                f"kl={stats['kl']:.4f}  kd={stats['kd']:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

        if int(cfg.get("eval_every", 0)) > 0 and step % int(cfg["eval_every"]) == 0:
            student.eval()
            val_ce = evaluate_ce_from_logits_fn(
                student_logits_fn, val_loader, device, autocast_ctx, max_eval
            )
            val_ppl = perplexity_from_loss(val_ce)
            last_metrics.update({"step": step, "val_loss": val_ce, "val_ppl": val_ppl})
            print(f"step {step:05d}  val_ce={val_ce:.4f}  val_ppl={val_ppl:.2f}", flush=True)
            state["history"].append({"step": step, "val_loss": float(val_ce), "val_ppl": float(val_ppl)})
            best = state.get("best_val_loss")
            if best is None or val_ce < float(best) - 1e-4:
                state["best_val_loss"] = float(val_ce)
                state["best_step"] = step
                state["bad_evals"] = 0
                save_checkpoint(ckpt_dir / "best.pt", student, optimizer, step=step, config=gpt_cfg)
                print(f"saved best.pt  val_ce={val_ce:.4f}  step={step}", flush=True)
            else:
                state["bad_evals"] = int(state.get("bad_evals") or 0) + 1
                print(f"no improve  bad_evals={state['bad_evals']}/{patience or 'off'}", flush=True)
            save_train_state(ckpt_dir, state)
            if patience > 0 and step >= min_eval_step and int(state["bad_evals"]) >= patience:
                print(f"early_stop  step={step}  best_step={state['best_step']}", flush=True)
                stopped_early = True
                max_steps = step
                break

        if int(cfg.get("sample_every", 0)) > 0 and step % int(cfg["sample_every"]) == 0:
            ids = torch.tensor([tokenizer.encode(prompt) or [tokenizer.eos_id]], device=device)
            out = sample_generate(
                student,
                ids,
                int(cfg.get("max_new_tokens", 80)),
                gpt_cfg.block_size,
                temperature=float(cfg.get("sample_temperature", 0.8)),
                top_k=int(cfg.get("sample_top_k", 50)),
            )
            print("sample:", tokenizer.decode(out[0].tolist()), flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    n_done = max_steps - start_step
    print(f"wall={elapsed:.1f}s  steps={n_done}  step/s={n_done / max(elapsed, 1e-9):.2f}")
    if device.type == "cuda":
        print(f"peak_mem_MiB={torch.cuda.max_memory_allocated(device) / (1024**2):.1f}")

    save_checkpoint(ckpt_dir / "last.pt", student, optimizer, step=max_steps, config=gpt_cfg)
    last_metrics["wall_s"] = elapsed
    last_metrics["early_stop"] = stopped_early
    last_metrics["best_step"] = state.get("best_step")
    last_metrics["best_val_loss"] = state.get("best_val_loss")
    save_train_state(ckpt_dir, state)
    return last_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 8 知识蒸馏（不覆盖莎士比亚专家 ckpt）")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "distill_t4.json")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--ckpt-dir", type=Path, default=ROOT / "checkpoints" / "distill_t4")
    parser.add_argument(
        "--student-ckpt",
        type=Path,
        default=ROOT / "checkpoints" / "shakespeare_perdoc" / "best.pt",
        help="只读；蒸馏结果写到 --ckpt-dir",
    )
    parser.add_argument("--teacher", type=str, default="gpt2", help="HF 名或本地目录")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true", help="只报 Student/Teacher val PPL，不训练")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    cfg = load_train_cfg(args.config)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    tokenizer = GPT2TokenizerWrapper()
    device = pick_device(args.device)
    run_distill(
        cfg,
        documents=load_pretrain_documents(),
        tokenizer=tokenizer,
        device=device,
        ckpt_dir=args.ckpt_dir,
        student_ckpt=args.student_ckpt,
        teacher_name=args.teacher,
        resume=args.resume,
        eval_only=args.eval_only,
        use_amp=not args.no_amp,
    )


if __name__ == "__main__":
    main()
