# 本地小实验：莎士比亚全集预训练冒烟

日期：2026-09-03  
机器：笔记本 RTX 4060 Laptop 8GB  
2000 step，val PPL 273→76，大约 3 分钟。没有 NaN / OOM。生成开始像剧本，但仍会复读。

本目录数字来自这一次日志。

---

## 3. 环境

| 项 | 值 |
| --- | --- |
| 机器 | WSL2，`LAPTOP-6FQIF6OJ` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8192 MiB |
| 驱动 | 566.24 |
| 计算能力 | 8.9（Ada，支持 bf16） |
| PyTorch | 2.6.0+cu124 |
| 精度 | `torch.bfloat16` autocast；`scaler=False`（符合 `utils.amp_dtype_and_scaler`） |
| 代码 | `scripts/base_train.py` |
| 配置快照 | 同目录 `config.json`（与当时 `configs/default.json` 一致） |
| 数据 | `data/shakespeare_complete.txt`（43 篇公有领域文本拼成） |
| Checkpoint | 当时的 `shakespeare_full/` 已删；现用 `checkpoints/shakespeare_perdoc/best.pt` |

命令：

```bash
python scripts/base_train.py \
  --config configs/default.json \
  --device cuda \
  --ckpt-dir checkpoints/shakespeare_full
```

第一次启动失败：`ValueError: 切分后 train/val 不能为空`。原因是当时 tokenizer 词表没真正加载（HuggingFace 离线/SSL 问题），encode 几乎是空的。改成本地 `assets/gpt2_tokenizer/merges.txt` 的 GPT-2 BPE 后第二次跑通。

---

## 4. 模型与数据量

参数量实测 **70,344,192（70.34M）**。README 写「约 80M」是规划口径，简历若写精确数字用 70M。词嵌入和 `lm_head` **不共享**，各约 25.7M，两者加起来占了一大半。

| 项 | 值 |
| --- | --- |
| n_layer / n_head / n_embd | 6 / 8 / 512 |
| block_size | 256 |
| vocab | GPT-2 BPE 50257 |
| pos_encoding | rope |
| dropout | 0.0 |
| 优化器 | AdamW，lr=3e-4，wd=0.1，betas=(0.9, 0.95) |
| batch_size | 8 |
| max_steps | 2000 |
| grad_clip | 1.0 |
| 语料 | 6,016,262 字符，43 个文件 |
| token 总数 | 1,950,498 |
| train / val | 1,755,448 / 195,050（按 token 下标 90/10，不是按剧本切） |
| 非重叠窗口 | train 6857，val 761 |
| 本次见过的 token | 2000 × 8 × 256 = **4,096,000**（约 **2.33** 个 train epoch） |
| 验证 | 每 100 step 抽 **20 个 val batch**，不是全验证集（全量约 95 batch） |

没有学习率衰减，没有 warmup。单步 train loss 是 **一个 batch**，所以曲线很抖；val 是 20 batch 平均，所以更平滑。

墙钟（用 ckpt 时间戳反推）：step 500→2000 约 138 秒，约 **10.8 step/s**。整次训练大约 **3.1 分钟**，另加分词和建 loader。4060 8GB 跑这个配置没有 OOM。

---

## 5. 数字结果

完整表见 `metrics.csv`。曲线见 `loss_curve.png`。

| step | train loss | train PPL（单 batch） | val loss | val PPL |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10.88 | 53240 | — | — |
| 100 | 5.11 | 166 | 5.61 | 273 |
| 500 | 4.29 | 73 | 4.72 | 112 |
| 1000 | 3.94 | 51 | 4.48 | 89 |
| 1700 | 3.86 | 47 | 4.33 | 76 |
| 2000 | 3.44 | 31 | **4.33** | **76** |

解读：

1. **前 700 步是真正在学语言。** val PPL 273→97，下降最快。说明前向、causal mask、AMP、优化器分组都没有写反。
2. **700 步之后收益变小。** 1000→2000 只从 89 降到 76。1700 之后 val 在 76–79 之间晃，基本横盘。
3. **train/val 缝在拉大。** 末期约 3.44 vs 4.33。70M 模型、dropout=0、只有 1.76M train token、还扫了 2.3 个 epoch，过拟合是预期现象，不是实现错误。
4. **没有 NaN。** bf16 不用 GradScaler 在这张 4060 上是对的。
5. 单步 train 从 3.17 跳到 4.19 很常见，batch=8 噪声大，看 val 曲线即可。

最好的 val 点：step 2000，val loss 4.3295 / PPL 75.91。step 1700 几乎一样（4.3346 / 76.29）。两个 ckpt 都能当后续实验起点；2000 的 train 更低，略更背数据。

---

## 6. 生成（greedy，prompt=`ROMEO:`）

采样设置：每 200 step、贪心、`max_new_tokens=40`。完整摘录见 `samples.md`。

| step | 观感 |
| ---: | --- |
| 200 | 碎片、舞台提示乱堆 |
| 600 | 开始像英文句子，但 `And in the world` 复读 |
| 1000 | 有完整句和场次标题 |
| 1200 | `[_Exeunt._]`、`SCENE III. Another part of the Castle.` 格式对上了 |
| 1400 | 几乎全是空行（greedy 锁死） |
| 1800 | 最像诗的一次：`The sun is like a king...` |
| 2000 | 出现人物名 `PERICLES`，但仍 `nor she, nor she` / `love her love` |

这不能直接当「模型质量差」的证据。训练脚本里只有 greedy，一旦抽到空行或套话就会一直重复。Day 10 加上 temperature / top-k 后再看样例，会公平得多。

---

## 7. 这次说明了什么、没说明什么

已经能当作证据的：

- 手写 GPT + 单卡 AMP 预训练脚本在真 GPU 上可复现
- 本机 4060 8GB 跑得动默认配置，上云前不必先缩小模型
- 莎士比亚全集（约 2M token）对 70M 模型来说太小，2000 step 已经接近这条数据的天花板
- Checkpoint 格式可用：`last.pt` 以及 500/1000/1500/2000

还不能当作证据的：

- 不是云上 4090 / A10 结果，不能写进「云上预训练时长 / 费用」
- 没有消融（RoPE vs learned PE、头数、AMP on/off）
- 没有 tok/s、显存峰值的系统记录（只有 ckpt 时间戳反推的大约 11 step/s）
- 验证只抽 20 batch；PPL 和全验证集会有偏差
- 生成质量被 greedy 低估

---

## 8. 上云和写材料时怎么用

真正拉低 PPL 需要更多数据或蒸馏，而不是把莎士比亚再训 2 万步。消融用同一数据、同一 2000 step 即可对比。

这次用到了：`gpt.py`、`dataloader.py`、`optim.py`、`checkpoint.py`、`tokenizer.py`、`utils.py`、`scripts/base_train.py`。
