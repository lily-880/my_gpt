"""冻结 CLIP + 线性投影，把图像向量接到 GPT 前面。Day 13 概念验证。"""


class ClipPrefixProjector:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Day 13：CLIP 视觉塔冻结 + Linear(clip_dim, n_embd)")
