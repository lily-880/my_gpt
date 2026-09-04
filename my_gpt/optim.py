"""优化器工厂：调用 PyTorch 自带 AdamW，不手写更新公式。Day 4 实现。"""


def build_optimizer(model, learning_rate: float, weight_decay: float):
    raise NotImplementedError("Day 4：参数分组（有/无 weight decay）+ torch.optim.AdamW")
