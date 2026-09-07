# Day 10 KV-Cache

日期：2026-09-07  
机器：本机 RTX 4060 Laptop  
权重：`checkpoints/shakespeare_perdoc/best.pt`

无 cache 每步整段重算 vs `Engine`（先 prefill，之后只送新 token）。cache 每层 `(B, n_head, T, head_dim)`，按步拼接。满 `block_size=256` 之后丢掉 cache、对最近 256 token 再 prefill，和 `sample_generate` 滑窗一致，所以 prompt 已经 256 长时仍能继续写。满窗后每步代价回到整窗前向，cache 的加速发生在窗口未满时。

日志：`run.log`。`metrics.json`。

---

## 总表（80 个新 token，greedy）

| | 无 cache | KV-Cache |
| --- | ---: | ---: |
| 墙钟 | 0.402 s | **0.296 s** |
| 吞吐 | 199 tok/s | **270 tok/s** |
| 加速 | — | **1.36×** |
| 与无 cache 逐 token 一致 | — | **是** |

加速有限是因为模型小、上下文短、Python 逐步循环的开销占一截。省下的是「过去那些位置的 QKV 和注意力」，不是把 FFN 变成别的算法。再写长一点、窗口还没满时，差距会更大。

greedy 样例会复读 `FALSTAFF. O, thou art a man.`——这是 greedy 的老问题，不是 cache 算错。对外仍用 temperature 0.8 / top-k 50：

```bash
python scripts/sample.py \
  --ckpt checkpoints/shakespeare_perdoc/best.pt \
  --device cuda --cache --temperature 0.8 --top-k 50 --n 2
```

cache 形状：6 层 × 2 × `(B, 8, T, 64)`。RoPE 按绝对位置转。窗口内用 cache；超过 `block_size` 则整窗重算。`scripts/chat_cli.py` 会把多轮拼进同一条历史。
