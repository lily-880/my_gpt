# 莎士比亚预训练：总报告（2026-09-03 — 09-05）

机器：本机 RTX 4060 Laptop 8GB，bf16，无云。  
模型：6 层 / 8 头 / 宽 512 / ctx 256 / GPT-2 BPE，约 70M。  
语料：43 篇公有领域莎士比亚，约 195 万 token。

**对外权重：** `checkpoints/shakespeare_perdoc/best.pt`  
解码：`temperature=0.8`，`top-k=50`。不要 greedy，不要 `last.pt`。

抽样：

```bash
python scripts/sample.py \
  --ckpt checkpoints/shakespeare_perdoc/best.pt \
  --device cuda --temperature 0.8 --top-k 50 --n 2
```

---

## 两套 val，数字不能横比

| 协议 | 怎么切 | 用在哪 |
| --- | --- | --- |
| `token_tail` | 文件名排序拼成长河，后 10% | Day 6/7、shakespeare_best、v2 |
| `per_doc` | 每篇随机抽约 10% 窗口（种子 42） | shakespeare_perdoc |

`token_tail` 的 val 几乎全是 Titus 之后几部。`per_doc` 43 部都有窗口，数字会更低，**不是模型突然强了一倍**。

---

## 实验一览

同一架构。下表「变了什么」以外都固定。

| 轮次 | 变了什么 | val PPL | 结论 |
| --- | --- | ---: | --- |
| Day 6 首跑 | 打通 2000 步 | 273→76（尾巴） | 链路通；空词表修过一次 |
| Day 7 基线 | 复跑 Day 6 | **77.27** | 旧协议对照；`ablate_baseline/last.pt` |
| learned PE | 只改位置编码 | 89.28 | 短跑里 RoPE 更好 |
| n_head=4 | 只改头数 | 74.46 | 与 8 头打平，不能写 8 头更强 |
| AMP off | 只改 fp32 | 76.85 | 质量一样；bf16 约 1.5× 步速，少 ~0.4GB |
| dropout 0.1 | 只开 dropout | 77.59 | 2000 步无 val 收益 |
| 采样 vs greedy | 同一基线 ckpt | — | greedy 复读；0.8 / top-k 50 才有剧本格式 |
| shakespeare_best | batch 32、8000 步、dropout 0.2 | 88→**727** | ~37 epoch，背书。不是 NaN |
| shakespeare_v2 | dropout 0.15 + cosine + batch 16 | ~114 @1600 | 防过拟合药叠多了，不如基线。已删 |
| shakespeare_perdoc | 基线配方 + `per_doc` + early stop | 189→**34.53** @3200 | 新协议谷底；3600 早停 |

分报告：

- [Day 6 首跑](2026-09-03_local_shakespeare_full/REPORT.md)
- [Day 7 消融](2026-09-04_day7_ablation/REPORT.md)
- [采样 vs greedy](2026-09-04_sampling_vs_greedy/REPORT.md)
- [8000 步过拟合](2026-09-04_shakespeare_best_overfit/REPORT.md)
- 本轮日志：`shakespeare_perdoc/train.log`

---

## 现在采用的配方

`configs/shakespeare_perdoc.json`：dropout 0，batch 8，lr 恒定 3e-4，`split=per_doc`，`best.pt`，patience=4。

3200 步 val 最低（34.53），随后连续 4 次回升，3600 早停。train/val 仍有缝，但没有再背到 700。大约 3.7 个 epoch，再加步数就是 best 那条路。

---

## 面试可以讲的三句话

1. 小语料上加步数会背书：8000 步 train loss 0.46、val PPL 727；必须 `best.pt` + early stop。  
2. 消融一次只改一个开关：RoPE 优于 learned；4/8 头打平；bf16 不掉点。  
3. val 按长河尾巴会偏晚期作品；按篇随机窗口后数字不能和 77 比。解码必须看温度采样。

下一步按计划是 Day 8 蒸馏，不要再加长莎士比亚。
