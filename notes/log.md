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
