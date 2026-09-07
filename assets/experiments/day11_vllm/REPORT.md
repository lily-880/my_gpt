# Day 11 vLLM（官方 gpt2）

日期：2026-09-07  
机器：本机 RTX 4060 Laptop 8GB，WSL2  
`vllm==0.8.5.post1`，配 torch 2.6.0+cu124。模型是官方 GPT-2（`data/hf_gpt2`），没有加载莎士比亚权重。

`enforce_eager=True`：本机没有给 inductor 用的 C 编译器。WSL 关掉了 pin_memory，FlashInfer 也没有。

日志：`run.log`。`metrics.json`。`samples.md`。

---

## 总表（每条最多 64 个新 token，8 条 prompt）

| | HF `generate` 串行 | vLLM |
| --- | ---: | ---: |
| 单条 | 0.52 s（~124 tok/s） | **0.37 s（173 tok/s）** |
| 8 条合计 | 3.92 s（~131 tok/s） | **0.41 s（1245 tok/s）** |
| 8 条墙钟比 | — | **9.5×** |

9.5 倍主要是一次跑 8 条，不是单条快 9 倍。加载大约 8.4 秒。

## 生成原文（温度 0.8 / top-k 50）

**Hello, my name is**

```
Hello, my name is Jack, and I'm from the New England Patriots. I was a special teamer for four years in the NFL, and I really love football. I started back when I was a rookie and I always stayed true to myself. I used to watch the games, and I knew I would be part of it.
```

**The capital of France is**

```
The capital of France is a centre of global trade and investment, and a place where companies can invest overseas. It also lies on a long, fertile land of trade, investment and tourism.

But what about the rest of the country?

With globalisation taking root, so does the way the French capital looks.

In
```

**ROMEO:**

```
ROMEO: I will let you in for a chat! (Applause) What did you say? (Applause) Did you just say that you liked the character and the story? (Applause) Did you just say you liked the character and the story? (Applause) Did you just say that
```

其余 5 条见 `samples.md`。`ROMEO:` 后面是掌声和现代英语，和莎士比亚专家不是同一个模型。

装的时候踩过：最新 vLLM 会换 torch 2.13；Transformers 5 和 0.8.5 对不上；镜像经常下不全 tokenizer，所以本地拼了 `data/hf_gpt2`。自研 RoPE 权重加载不进去。
