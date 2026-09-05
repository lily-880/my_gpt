# my_gpt

从零手写 Decoder-only GPT 的训练与推理链路（参考 [nanochat](https://github.com/karpathy/nanochat) 的模块划分，**未复制其源码**），在公开电商评论文本上继续预训练，并做消融、知识蒸馏、INT8 量化 Demo，以及冻结 CLIP + 投影层的商品图文**概念验证**。模型约 80M 参数，用来把流程跑通，不是 SOTA。

实现程度会在文档里标成三类：**完整实现（小规模）** / **研究 Demo** / **只掌握原理**。

## 当前进度（莎士比亚收束；下一件是 Day 8 蒸馏）

- Day 1–5：骨架、算子、整模型、数据、过拟合、`base_train.py`。
- Day 6–7：4060 上跑通约 70M 预训练，并完成 RoPE / 头数 / AMP / dropout 消融。
- 2026-09-05：val 改为按篇随机窗口；采用 `checkpoints/shakespeare_perdoc/best.pt`（step 3200，该协议下 val PPL 34.5）。
- 总报告：`assets/experiments/REPORT.md`。旧协议（长河尾巴）基线仍是 `checkpoints/ablate_baseline/last.pt`，PPL 77，两套数字不能比。

## 默认超参

写在 `configs/default.json`，含义见 `notes/glossary.md`。

| 名字 | 值 | 人话 |
| --- | --- | --- |
| n_layer | 6 | Transformer 叠 6 层 |
| n_head | 8 | 注意力分 8 个头 |
| n_embd | 512 | 向量宽度 512 |
| block_size | 256 | 一次最多看 256 个 token |
| vocab_size | 50257 | GPT-2 词表，方便和 Teacher 蒸馏 |
| pos_encoding | rope | 位置用 RoPE（消融时可改 learned） |

## 本地环境

需要 Python 3.10+。在项目根目录：

```bash
cd my_gpt
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
# CPU 版 PyTorch（本机写代码用；体积远小于 CUDA 版）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest -q
```

本机**不要**执行 `pip install -r requirements-gpu.txt`（里面有 vLLM，留给云 GPU）。

Windows WSL 同样用上面的 `source` 命令。激活成功后，提示符前面通常会出现 `(.venv)`。

## 目录

```
my_gpt/                 # Python 包：模型、数据、推理
scripts/                # 命令行入口（按天实现）
configs/                # 超参
tests/                  # 单元测试
assets/                 # 曲线和样例图
notes/                  # 术语表、实验笔记
requirements.txt        # 本地
requirements-gpu.txt    # 云上额外依赖
```

## 数据怎么切（Day 4）

预训练默认用**莎士比亚全集**（公有领域，约 38 部戏剧 + 十四行诗等），不是 Karpathy 那份 1.1MB 的 tiny-shakespeare。单篇在 `data/shakespeare_plays/`，拼好的长河是 `data/shakespeare_complete.txt`。`overfit_check.py` 仍用 tiny，方便 CPU 冒烟。

切法（默认 `split=per_doc`，`split_seed=42`）：

1. 按 `data/shakespeare_plays/` 里每一篇单独分词，不先拼成长河再切。
2. **每篇**把不重叠窗口打乱，固定种子随机抽出约 10% 进验证，其余进训练。开头、中间、结尾都可能进 val，不是篇末一刀切。
3. 不能按单个 token 随机抽：相邻 token 会漏进 train 和 val。抽的是整段 `block_size` 窗口。
4. 验证窗口按篇轮转，所以 `eval_batches=20` 也会扫到多部作品。窗口不跨篇；挖洞后的片段不重新粘在一起。
5. 太短、凑不够两个窗口的篇只进训练。

旧实验（Day 6/7、shakespeare_best/v2）是整条长河后 10%（`token_tail`）。只要篇末 10% 用 `per_doc_tail`。新旧 PPL **不能直接比**。

莎士比亚全集大约是 tiny 的 5 倍，对 80M 模型仍偏小。以后电商数据按商品/评论 ID 切，和这里的按篇切是同一类做法。

指定其它文件：

```bash
python scripts/base_train.py --config configs/default.json --device cuda --data-file data/shakespeare_complete.txt
```

本机过拟合（确认 loss 能降，不必用完整 80M 模型）：

```bash
python scripts/overfit_check.py
pytest -q tests/test_data_optim.py
```

本机训练冒烟（小模型，CPU）：

```bash
python scripts/base_train.py --config configs/cpu_smoke.json --device cpu
```

云上预训练（默认约 80M）：

```bash
python scripts/base_train.py --config configs/default.json --device cuda
```

AMP：有 GPU 且支持 bf16 时用 bf16、不用 GradScaler；更老的 GPU 用 fp16 + GradScaler；CPU 自动关掉 AMP。nanochat 的自定义 `COMPUTE_DTYPE` 不在 14 天范围。

## 和 nanochat 的关系

只借鉴「小而全的训练—推理仓库怎么拆」。14 天不实现其自定义精度、`--depth` 自动超参、Muon、RL 和评测套件。当前 nanochat 的 smear / GQA / 滑窗等也刻意不做。

## 多模态说明

聚焦 NLP 底座。Day 13 用现成 CLIP 验证「图像向量可以进 Decoder」，不训练视觉模型、不做视频。
