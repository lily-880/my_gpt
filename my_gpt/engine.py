"""推理引擎：KV-Cache + 采样。Day 10–11 实现。"""


class Engine:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Day 10：KV-Cache 贪心生成")

    def generate(self, *args, **kwargs):
        raise NotImplementedError("Day 11：Top-k / Top-p / repetition penalty")
