# Day 6（2026-09-03）——已完成，在本机 4060，没上云

- 计划已更正：Day 6/7 默认就是 4060，不是「必须租 4090」。莎士比亚只打通 pipeline。
- 本机 RTX 4060 Laptop 8GB，`configs/default.json`，全集 2000 step。bf16，无 GradScaler，无 OOM / NaN。
- val PPL 273→76；train/val 末期约 3.44 vs 4.33，开始过拟合。墙钟约 3 分钟（~11 step/s）。
- 第一次失败：tokenizer 空词表。本地 GPT-2 merges 修好后重跑成功。
- 报告：`assets/experiments/2026-09-03_local_shakespeare_full/`。ckpt：`checkpoints/shakespeare_full/`。
- 下一件：Day 7 三组消融（仍在 4060；这次 run 当 RoPE/8head/AMP-on 对照，不必重跑）。

# Day 5

- `utils.py`：AMP 选择、train_step、val loss、PPL=exp(CE)、greedy_generate。
- `scripts/base_train.py`：循环 log / eval / sample / save。
- `configs/cpu_smoke.json` 本机小模型冒烟；`default.json` 原计划留给云上 80M，2026-09-03 已在本机 4060 上冒烟通过。

# Day 4

- `dataloader.py`：全文 encode 后按 token 90/10 切分，非重叠窗口，`y` 相对 `x` 错开 1。
- `optim.py`：AdamW；Embedding / 1 维参数无 decay，Linear 权重有 decay。
- `scripts/overfit_check.py` + `tests/test_data_optim.py`：单 batch 过拟合，loss 必须明显下降。

# Day 3

- `TransformerBlock` + 完整 `GPT.forward`（logits / 可选 CE loss）。
- `checkpoint.py`：保存/加载 model、optimizer、step、config。
- `tokenizer.py`：HF GPT-2；文档之间用 eos 拼接。learned PE 加在词嵌入上，不在每层注意力里。

# Day 2

- `my_gpt/gpt.py`：causal mask、RMSNorm、RoPE、单头/多头注意力、FFN。
- GPT 整模型仍留到 Day 3。
- `tests/test_gpt_ops.py`：shape + 未来 token 概率≈0。

# Day 1 完成记录

- 工程骨架已按 PLAN 建好；Python 文件目前是占位（调用会 NotImplementedError）。
- 本地：先 `pip install torch --index-url https://download.pytorch.org/whl/cpu`，再 `pip install -r requirements.txt`。
- 云 GPU 用 CUDA 版 torch + `requirements-gpu.txt`（含 vLLM，本机不装）。
- 默认超参见 `configs/default.json`。
- 术语说明见 `notes/glossary.md`。
