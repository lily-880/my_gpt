"""性价比分数：loss / 算力 / 参数量任一变大，value 应变小。"""

import math

from my_gpt.efficiency import chance_ce, gain_score, value_score


def test_chance_ce_is_log_vocab():
    assert abs(chance_ce(2) - math.log(2)) < 1e-12


def test_value_penalizes_loss_time_and_size():
    base = value_score(4.0, 70_000_000, 200.0, 2048.0)
    assert value_score(5.0, 70_000_000, 200.0, 2048.0) < base
    assert value_score(4.0, 70_000_000, 400.0, 2048.0) < base
    assert value_score(4.0, 140_000_000, 200.0, 2048.0) < base


def test_gain_zero_when_val_does_not_improve():
    assert gain_score(3.6, 70_000_000, 40.0, 2048.0, start_loss=3.5) == 0.0
    assert gain_score(3.4, 70_000_000, 40.0, 2048.0, start_loss=3.5) > 0.0
