"""T-049~T-057: 时长预测器测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from model.duration_predictor import DurationPredictor, regulate_length
from config import DurationPredictorConfig


def test_duration_predictor_instantiation():
    """T-049: DurationPredictor(config) 能实例化"""
    config = DurationPredictorConfig()
    dp = DurationPredictor(config)
    assert dp is not None


def test_duration_predictor_forward():
    """T-051: forward(x, x_mask) 输入 (2, 10, 192)，输出 shape (2, 10)"""
    config = DurationPredictorConfig()
    dp = DurationPredictor(config)

    x = torch.randn(2, 10, config.in_channels)  # (batch, time, channels)
    x_mask = torch.ones(2, 1, 10)  # 全1表示无填充

    output = dp(x, x_mask)

    assert output.shape == (2, 10)


def test_regulate_length_single():
    """T-053: regulate_length(encoder_output, durations) 输入 (1, 3, 192) + durations [2, 3, 1]，输出 shape (1, 6, 192)"""
    encoder_output = torch.randn(1, 3, 192)
    durations = torch.tensor([[2, 3, 1]])

    output = regulate_length(encoder_output, durations)

    assert output.shape == (1, 6, 192)
    # 检查展开是否正确
    assert torch.allclose(output[0, 0, :], encoder_output[0, 0, :])  # 第一个音素重复2次
    assert torch.allclose(output[0, 2, :], encoder_output[0, 1, :])  # 第二个音素重复3次


def test_regulate_length_batch():
    """T-055: batch 内不同 duration 总和时，输出正确 padding 到最大长度"""
    encoder_output = torch.randn(2, 4, 192)
    # 第一个样本 durations: [1, 2, 1, 2] -> 总和 6
    # 第二个样本 durations: [3, 1, 1, 1] -> 总和 6
    durations = torch.tensor([[1, 2, 1, 2], [3, 1, 1, 1]])

    output = regulate_length(encoder_output, durations)

    # 输出 shape 应该是 (2, 6, 192)，padding 到 batch 内最大长度
    assert output.shape == (2, 6, 192)
