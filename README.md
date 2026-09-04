# my_gpt

从零手写 Decoder-only GPT 的训练与推理链路（参考 [nanochat](https://github.com/karpathy/nanochat) 的模块划分，**未复制其源码**），在公开电商评论文本上继续预训练，并做消融、知识蒸馏、INT8 量化 Demo，以及冻结 CLIP + 投影层的商品图文**概念验证**。模型约 80M 参数，用来把流程跑通，不是 SOTA。

实现程度会在文档里标成三类：**完整实现（小规模）** / **研究 Demo** / **只掌握原理**。

## 当前进度（Day 1）

- 目录、默认超参、本地依赖已就绪。
- 核心代码仍是占位，调用会抛 `NotImplementedError`。
- Day 2 开始写模型算子和单元测试。

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

## 和 nanochat 的关系

只借鉴「小而全的训练—推理仓库怎么拆」。14 天不实现其自定义精度、`--depth` 自动超参、Muon、RL 和评测套件。当前 nanochat 的 smear / GQA / 滑窗等也刻意不做。

## 多模态说明

聚焦 NLP 底座。Day 13 用现成 CLIP 验证「图像向量可以进 Decoder」，不训练视觉模型、不做视频。
