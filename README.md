# my_gpt

从零手写 Decoder-only GPT 的训练与推理链路（参考 [nanochat](https://github.com/karpathy/nanochat) 的模块划分，**未复制其源码**），在公开电商评论文本上继续预训练，并做消融、知识蒸馏、INT8 量化 Demo，以及冻结 CLIP + 投影层的商品图文**概念验证**。模型约 80M 参数，用来把流程跑通，不是 SOTA。

实现程度会在文档里标成三类：**完整实现（小规模）** / **研究 Demo** / **只掌握原理**。

## 当前进度（Day 6 完成；下一件是 Day 7 消融）

- Day 1–5：骨架、算子、整模型、数据、过拟合、`base_train.py`。
- Day 6（2026-09-03）：本机 **RTX 4060 Laptop** 跑通默认约 70M：莎士比亚全集、2000 step、bf16、val PPL 273→76。报告：`assets/experiments/2026-09-03_local_shakespeare_full/`。计划已改为默认在 4060 上训，**不必为莎士比亚上云**。
- Day 7 起：本机做三组消融；电商 CPT / 蒸馏也优先 4060。

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

切法：

1. 用 GPT-2 tokenizer 把**全文**编成一条 token 长河。
2. **按 token 下标**切开：前 90% 训练、后 10% 验证（不是按剧本幕/场切）。
3. 再切成不重叠的窗口，长度 `block_size`。每条样本的 `y` 是 `x` 向右错开 1 位（猜下一个 token）。
4. 窗口拼不齐的尾巴丢掉。

这样验证集是文本的后 10%，和训练在时间上相邻，可能略有相似。莎士比亚全集大约是 tiny 的 5 倍，对 80M 模型仍偏小（Chinchilla 量级要十亿级 token）；再大就超出莎士比亚能提供的文本了。以后电商数据可以改成「按商品/按评论 ID 切」。

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
