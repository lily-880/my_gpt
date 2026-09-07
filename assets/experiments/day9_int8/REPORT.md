# Day 9 INT8：CPU 动态 PTQ

日期：2026-09-07  
机器：本机 CPU（动态量化不能上 GPU）  
权重：`checkpoints/shakespeare_perdoc/best.pt`（只读，未覆盖）  
方法：`torch.ao.quantization.quantize_dynamic`，只改 `nn.Linear` → `qint8`  
数据：与专家相同的 `split=per_doc`、`split_seed=42`、batch 8、`eval_batches=20`

训完再压，只改 `nn.Linear`。词嵌入仍是 fp32，体积不会变成四分之一。

日志：`run.log`。`metrics.json`。`samples.md`。

---

## 总表

| | fp32 | INT8 Linear | 变化 |
| --- | ---: | ---: | ---: |
| 权重 state_dict | 268.4 MiB | 140.8 MiB | **1.91× 更小** |
| val PPL | **34.53** | **34.64** | +0.12 |
| 生成 40 token | 0.58 s（70 tok/s） | 0.39 s（104 tok/s） | **1.49×** |
| 未量化 Linear | 25 | 0 | 全部打上 |

PPL 与训练日志里 best 的 34.53 对齐。量化 0.5 秒做完。

体积不是 4×：`wte`（50257×512）仍是 fp32；被压的是 25 个 Linear（含 `lm_head` 和各层投影）。训练 ckpt `best.pt` 有 806 MiB，里面还有 AdamW 状态，**不要拿文件体积当量化收益**。

延迟是同脚本、同 prompt、warmup 后各 1 次，不是多进程压测。CPU 上动态量化有加速，幅度有限。

量化后可以走同一套 `Engine`。动态量化按**这一次 forward 的激活范围**缩放，prefill 和逐步 decode 的 scale 不同，所以 INT8 上 cache 与整段重算不必逐 token 相同（fp32 上是对齐的）。本机 `ROMEO:` 40 token：INT8 无 cache 0.37s，INT8 + KV-Cache 0.25s。

---

## 样例

`ROMEO:`，温度 0.8，top-k 50，seed 0。量化会改 logits，两条不必相同。int8 里出现过 `LADY CAPTAIN` 这种串角。全文在 `samples.md`。
