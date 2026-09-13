"""训练性价比：val 越低、算力越省、模型越小，分数越高。"""

from __future__ import annotations

import math
from typing import Any


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def chance_ce(vocab_size: int) -> float:
    """均匀乱猜时的交叉熵，ln(词表)。从零预训练用它当起点。"""
    if vocab_size <= 1:
        raise ValueError("vocab_size 必须 > 1")
    return math.log(vocab_size)


def gpu_seconds_gb(wall_s: float, peak_mem_mib: float | None) -> float:
    """算力代理：墙钟 × 峰值显存（GPU·GiB·秒）。没有显存记录时退回墙钟。"""
    wall = float(wall_s)
    if wall <= 0:
        raise ValueError("wall_s 必须 > 0")
    if peak_mem_mib is None or float(peak_mem_mib) <= 0:
        return wall
    return wall * (float(peak_mem_mib) / 1024.0)


def value_score(
    val_loss: float,
    n_params: int,
    wall_s: float,
    peak_mem_mib: float | None = None,
    *,
    scale: float = 100.0,
) -> float:
    """综合性价比 V。越高越好。

    V = scale * (1/L * 1/C * 1/N)^{1/3}

    L = val CE，C = 墙钟×峰值显存（GiB），N = 参数量（百万）。
    三项等权几何平均：任一变大都会把分数压下去。
    """
    loss = float(val_loss)
    n_m = int(n_params) / 1e6
    compute = gpu_seconds_gb(wall_s, peak_mem_mib)
    if loss <= 0 or n_m <= 0:
        raise ValueError("val_loss 和 n_params 必须 > 0")
    return float(scale * (1.0 / loss * 1.0 / compute * 1.0 / n_m) ** (1.0 / 3.0))


def gain_score(
    val_loss: float,
    n_params: int,
    wall_s: float,
    peak_mem_mib: float | None = None,
    *,
    start_loss: float,
    scale: float = 100.0,
) -> float:
    """相对起点的提升效率。val 没有变好则记 0，避免「几乎没训」刷高分。"""
    loss = float(val_loss)
    start = float(start_loss)
    if start <= 0:
        raise ValueError("start_loss 必须 > 0")
    if loss >= start:
        return 0.0
    n_m = int(n_params) / 1e6
    compute = gpu_seconds_gb(wall_s, peak_mem_mib)
    if n_m <= 0:
        raise ValueError("n_params 必须 > 0")
    rel = (start - loss) / start
    return float(scale * (rel * 1.0 / compute * 1.0 / n_m) ** (1.0 / 3.0))


def efficiency_metrics(
    val_loss: float,
    n_params: int,
    wall_s: float,
    peak_mem_mib: float | None = None,
    *,
    start_loss: float | None = None,
    vocab_size: int | None = None,
) -> dict[str, float]:
    """一次算齐 value / gain / compute，方便写进日志和 train_state。"""
    start = start_loss
    if start is None and vocab_size is not None:
        start = chance_ce(vocab_size)
    out: dict[str, float] = {
        "value": value_score(val_loss, n_params, wall_s, peak_mem_mib),
        "n_params": float(n_params),
        "compute_gpu_gib_s": gpu_seconds_gb(wall_s, peak_mem_mib),
    }
    if start is not None:
        out["gain"] = gain_score(
            val_loss, n_params, wall_s, peak_mem_mib, start_loss=start
        )
        out["start_loss"] = float(start)
    return out


def print_efficiency(metrics: dict[str, Any]) -> None:
    parts = [f"value={metrics['value']:.3f}"]
    if "gain" in metrics:
        parts.append(f"gain={metrics['gain']:.3f}")
    parts.append(f"n_params={int(metrics['n_params'])}")
    parts.append(f"compute={metrics['compute_gpu_gib_s']:.1f} GPU-GiB-s")
    print("  ".join(parts), flush=True)
