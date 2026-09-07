"""按 URL 下载对外权重。权重不进 git（GitHub 单文件 100MB 限制）。"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "checkpoints" / "shakespeare_perdoc" / "weights.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="下载莎士比亚专家权重")
    parser.add_argument(
        "--url",
        type=str,
        default=os.environ.get("MY_GPT_WEIGHTS_URL", ""),
        help="直链。也可设环境变量 MY_GPT_WEIGHTS_URL",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.url:
        raise SystemExit(
            "没有 URL。把权重放到网盘或 GitHub Release 后：\n"
            "  python scripts/fetch_weights.py --url <直链>\n"
            "本机已有 best.pt 时：\n"
            "  python scripts/export_weights.py"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.url} -> {args.out}", flush=True)
    urllib.request.urlretrieve(args.url, args.out)
    print(f"wrote {args.out}  {args.out.stat().st_size / (1024**2):.1f} MiB")


if __name__ == "__main__":
    main()
