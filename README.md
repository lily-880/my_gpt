# my_gpt

手写的 Decoder-only GPT，大约 70M（6 层、8 头、宽 512、ctx 256，RoPE + RMSNorm）。在公有领域莎士比亚上预训练。词表用 GPT-2 的 50257，方便和现成 gpt2 做蒸馏。

参考了 nanochat 怎么拆目录，代码是自己写的。MIT 许可。

`lm_head` 没有和 `wte` 绑在一起，多大约 25M 参数。绑上会换一份对不齐现有 ckpt 的结构，所以保持分开。

对外权重：`checkpoints/shakespeare_perdoc/best.pt`（训练快照，含优化器，约 806 MiB）或同目录 `weights.pt`（只含模型）。生成用温度 0.8、top-k 50，不要 greedy。

```bash
python scripts/sample.py \
  --ckpt checkpoints/shakespeare_perdoc/best.pt \
  --temperature 0.8 --top-k 50 --n 2 --cache
```

没写 `--device` 时有 CUDA 用 GPU，否则 CPU。

数字和过程：`assets/experiments/REPORT.md`。

## 环境

Python 3.10+。有 GPU 的话用现成的 CUDA PyTorch（这边是 2.6.0+cu124）。

```bash
cd my_gpt
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

vLLM 单独装钉死的版本，别直接 `pip install vllm`（会把 torch 升掉）：

```bash
pip install -r requirements-gpu.txt
```

## 克隆下来缺权重

`checkpoints/` 不进 git（GitHub 单文件上限 100MB，完整 ckpt 约 806MB）。语料 `data/shakespeare_plays/*.txt` 在仓库里。

本机已有 `best.pt` 时导出只含模型的文件（约 268 MiB）：

```bash
python scripts/export_weights.py --ckpt checkpoints/shakespeare_perdoc/best.pt
```

别人需要权重时，把 `weights.pt` 放到网盘或 GitHub Release，再：

```bash
python scripts/fetch_weights.py --url <直链>
python scripts/sample.py --ckpt checkpoints/shakespeare_perdoc/weights.pt --cache
```

`sample.py` / `chat_cli.py` 也能读原来的 `best.pt`。

## 结果摘要

| 实验 | 结果 |
| --- | --- |
| 莎士比亚专家 | `per_doc` val PPL 34.53（3200 步早停） |
| 消融 | RoPE 比 learned PE 好；4 头和 8 头差不多；bf16 不掉点 |
| 过拟合 | 同一份语料训太久，val PPL 到 727。配方在 `configs/shakespeare_best.json`，不要当 demo |
| 蒸馏 | 网页 gpt2 在这域上更差（PPL 62）。T=4 → 30.5；α=0 只继续 CE 停在 33.5；T=1 → 36.5 |
| INT8 | Linear 动态量化，268→141 MiB，PPL 34.64；量化后也能走 KV-Cache（动态量化逐步 decode 不必和整窗逐 token 相同） |
| KV-Cache | greedy 和整段重算对得上；80 token 大约 1.36 倍。满 256 后整窗重算，还能继续写 |
| vLLM | 官方 gpt2，8 条请求相对 HF 串行大约 9.5 倍（主要是批在一起，单条大约 1.4 倍） |

配方：`configs/shakespeare_perdoc.json`（`configs/default.json` 与它相同）。`python scripts/base_train.py` 默认读这份。数据按篇切 train/val（`split=per_doc`），不要拿旧的 `token_tail` PPL 来比。

## 其它命令

```bash
python scripts/quant_demo.py --ckpt checkpoints/shakespeare_perdoc/best.pt
python scripts/kvcache_bench.py
python scripts/vllm_demo.py --model data/hf_gpt2
python scripts/chat_cli.py
```

`chat_cli.py` 会把每一轮的输入和模型输出拼进同一条上下文，不是每行独立续写。`/reset` 清空。注意力窗口仍是 256。

蒸馏：

```bash
python scripts/distill_train.py --config configs/distill_t4.json --ckpt-dir checkpoints/distill_t4
python scripts/distill_train.py --config configs/distill_alpha0.json --ckpt-dir checkpoints/distill_alpha0
```

复现说明在 `assets/experiments/day8_distill/`。

## 限制

模型小、上下文 256、语料是莎士比亚。INT8 是 CPU 动态量化。vLLM 跑的是官方 gpt2，加载不了这份 RoPE 权重。没有做电商续训、CLIP、多卡。
