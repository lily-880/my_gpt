# 实验总览

莎士比亚阶段已收束。数字、结论、现用权重：`assets/experiments/REPORT.md`。

对外：`checkpoints/shakespeare_perdoc/best.pt`，温度 0.8 / top-k 50。  
下一件：Day 8 蒸馏。

# Day 5

- `utils.py`：AMP 选择、train_step、val loss、PPL=exp(CE)、greedy_generate。
- `scripts/base_train.py`：循环 log / eval / sample / save。
- `configs/cpu_smoke.json` 本机小模型冒烟；`default.json` 原计划留给云上 80M，2026-09-03 已在本机 4060 上冒烟通过。

# Day 4

- `dataloader.py`：现为按篇随机窗口切分（`split=per_doc`）；旧长河 90/10 仍可用 `token_tail`。
- `optim.py`：AdamW；Embedding / 1 维参数无 decay，Linear 权重有 decay。
- `scripts/overfit_check.py` + `tests/test_data_optim.py`：单 batch 过拟合，loss 必须明显下降。

# Day 3

- `TransformerBlock` + 完整 `GPT.forward`（logits / 可选 CE loss）。
- `checkpoint.py`：保存/加载 model、optimizer、step、config。
- `tokenizer.py`：本地 GPT-2 BPE（`assets/gpt2_tokenizer/merges.txt`）。

# Day 2

- `my_gpt/gpt.py`：causal mask、RMSNorm、RoPE、单头/多头注意力、FFN。
- `tests/test_gpt_ops.py`：shape + 未来 token 概率≈0。

# Day 1

- 工程骨架按 PLAN 建好。默认超参见 `configs/default.json`。术语：`notes/glossary.md`。
