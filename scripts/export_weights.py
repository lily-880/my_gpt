"""从训练 ckpt 抽出模型权重，去掉 AdamW 状态。不覆盖原文件。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_gpt.checkpoint import gpt_config_from_checkpoint, load_checkpoint, save_checkpoint
from my_gpt.gpt import GPT


def main() -> None:
    parser = argparse.ArgumentParser(description="导出无 optimizer 的权重 ckpt")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "checkpoints" / "shakespeare_perdoc" / "best.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="默认写到同一目录的 weights.pt",
    )
    args = parser.parse_args()
    if not args.ckpt.is_file():
        raise FileNotFoundError(f"找不到 {args.ckpt}")
    out = args.out if args.out is not None else args.ckpt.with_name("weights.pt")
    if out.resolve() == args.ckpt.resolve():
        raise ValueError("拒绝覆盖源 ckpt，换一个 --out")

    cfg = gpt_config_from_checkpoint(args.ckpt, map_location="cpu")
    model = GPT(cfg)
    payload = load_checkpoint(args.ckpt, model=model, map_location="cpu")
    save_checkpoint(out, model, optimizer=None, step=int(payload.get("step", 0)), config=cfg)
    src_mib = args.ckpt.stat().st_size / (1024**2)
    dst_mib = out.stat().st_size / (1024**2)
    print(f"wrote {out}  {dst_mib:.1f} MiB  (source {src_mib:.1f} MiB, optimizer stripped)")


if __name__ == "__main__":
    main()
