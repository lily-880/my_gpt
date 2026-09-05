# 复盘：`shakespeare_best` 把验证集训崩了

日期：2026-09-04  
配置：`configs/shakespeare_best.json`  
日志：本目录 `shakespeare_best.log`  
归档：中间权重已删。对外模型见 `assets/experiments/REPORT.md`。  
曲线：`val_ppl_curve.png`  
机器：RTX 4060 Laptop，bf16，墙钟 **3626 s（60.4 分钟）**，2.21 step/s，峰值显存 7425 MiB

**一句话：** 不是 AMP 炸了，也不是代码写反。是 **70M 模型在约 176 万 train token 上扫了太多个 epoch**，把训练集背下来，验证集从 PPL 88 一路升到 **727**。明天不要再加长莎士比亚，要加 **early stop / 保存最优 ckpt**，真正变强去 **更多数据（电商 CPT）**。

---

## 1. 发生了什么

| step | train loss（单 batch） | val PPL | 备注 |
| ---: | ---: | ---: | --- |
| 800 | 3.40 | **87.8** | 本轮最好；没有存盘 |
| 1200 | ~2.9 | 87.9 | 平台期结束 |
| 2000 | 更低 | 107 | 已过拟合；有 `step_002000.pt` |
| 4000 | ~1.2 | 223 | `step_004000.pt` |
| 6000 | ~0.7 | 436 | `step_006000.pt` |
| 8000 | **0.46** | **726.7** | `last.pt`，最差 |

Day 7 基线（batch 8，2000 step，约 2.3 epoch）val PPL **77**。本轮最好也只有 88，还不如那次短跑。

见过的 token：`8000 × 32 × 256 = 6550 万`。相对 175 万 train token ≈ **37 个 epoch**。基线只有约 **2.3 个 epoch**。多训 16 倍遍数，验证没有变好，只是在背书。

train 掉到 0.2–0.5、val 升到 6.5，是教科书过拟合，不是 NaN。优化器一直在更新。

---

## 2. 为什么会做成这样（决策错误，不是偶然）

当时的动机是「拉满 4060、样例更好看」。改动叠在一起：

1. **batch 32、随机窗口、8000 步** → 每步 4 倍 token，总曝光量爆炸。  
2. **dropout 0.2** 挡不住 37 个 epoch；短跑时 0.1 也没改善 val，加 dropout 不能当「可以多训」的许可证。  
3. **学习率一直 3e-4**，没有衰减。后期仍用预训练初期的步长，在已经拟合的数据上猛更新，val 掉得更快。  
4. **`save_every=2000`**，真正最好的 800–1200 步没有档。`last.pt` 还是最差的一份。  
5. **没有 early stopping**，val 连升 30 次仍跑到 8000。  
6. 样例用了温度采样，观感一度像诗，容易误判成「模型变强」。那是背训练集，val 已经在罚。

显存 7.4 / 8.2 GB 说明卡吃满了，**算力用上了，用在错误的目标上**。

---

## 3. 样例为什么仍不理想

后期生成（step 7600 附近）仍有 Romeo / Juliet / Nurse，但情节乱、会造词（`HENRYNARDOKE`）、套话堆叠。过拟合不会自动变成「更好的作家」：它更会复述训练片段，在验证风格的接续上更差。

对外展示仍应使用：

- 现用权重：`checkpoints/shakespeare_perdoc/best.pt`（按篇随机 val，PPL 34.5）  
- 旧协议对照：`checkpoints/ablate_baseline/last.pt`（长河尾巴 val，PPL 77）  
- 解码：temperature 0.8、top-k 50（见 `2026-09-04_sampling_vs_greedy/`）

本轮过拟合权重已删除，不要再训 8000 步莎士比亚。

---

## 4. 明天怎么改（按优先级）

莎士比亚这条线上，**再训不会得到更好的 val**。明天分两块：一小块工程补丁，然后回到计划的 Day 8。

### A. 必做：训练循环补安全网（先写代码和单测，再跑短实验）

1. **保存 `best.pt`**：每当 val_loss 创新低就覆盖写一份。  
2. **early stop**：val 连续恶化 `patience` 次（建议 4–6 次 eval，即 800–1200 步）就停，仍写 `last.pt` 但不覆盖 `best.pt`。  
3. **`save_every` 改密**（200 或 400），以免最佳点落在两个整数档之间。  
4. 日志：`PYTHONUNBUFFERED=1` 再 `| tee`，避免终端假死。

短跑验收：用 `shakespeare_best.json` 但 `--max-steps 1500`。预期：在 ~800–1200 停下或至少留下 `best.pt`，val PPL 应停在 90 附近而不是冲到 700。

### B. 不要做

- 不要再跑 8000 步莎士比亚。  
- 不要从 `last.pt` resume 接着训。  
- 不要指望更大 dropout / 更小 lr 在这份 2M token 上把 70M 训成「莎士比亚生成器」。数据天花板还在。

### C. 若还想碰莎士比亚（可选，1 小时内）

只改**一个**开关，对照 Day 7 基线：

- 方案：`max_steps=1200`，`batch_size=8` 或 16，`dropout=0.2`，cosine 或线性把 lr 收到 3e-5，有 `best.pt`。  
- 成功标准：val **不差于 77**，且样例用同一套 0.8 / top-k 50。  
- 达不到就停，采用 Day 7 基线为莎士比亚最终权重。

### D. 正路：换数据，不要换步数

计划里的 Day 8 蒸馏、Day 9 电商 CPT 才是「模型变强」：Teacher 分布或电商评论会提供**新 token**，而不是把同一份戏再背 30 遍。过拟合课已经上完，写进面试即可。

面试可说：加大 batch 和步数后 train PPL→1.6、val PPL→727，说明小语料上必须 early stop；对外权重停在短跑基线。

---

## 5. 数字备查

- 最佳本轮：step 800，val_loss 4.475，val_ppl 87.80  
- 终点：step 8000，train_loss 0.46，val_loss 6.59，val_ppl 726.73  
- 基线（应保留）：Day 7 `ablate_baseline`，val_ppl 77.27  
- 过拟合 ckpt 已删，数字以本报告和 `shakespeare_best.log` 为准
