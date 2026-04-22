"""T-042~T-048: 文本编码器测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from model.text_encoder import TextEncoder
from config import TextEncoderConfig


def test_text_encoder_instantiation():
    """T-042: TextEncoder(config) 能实例化，参数量在合理范围（< 5M）"""
    config = TextEncoderConfig()
    encoder = TextEncoder(config)

    # 计算参数量
    params = sum(p.numel() for p in encoder.parameters())
    params_m = params / 1e6

    assert params_m < 5, f"TextEncoder 参数过多: {params_m:.2f}M"


def test_text_encoder_forward_output_shape():
    """T-044: forward(x, x_lengths) 输入 x=(2, 10) int, x_lengths=(2,)，输出 shape 为 (2, 10, 192) 和 mask (2, 1, 10)"""
    config = TextEncoderConfig()
    encoder = TextEncoder(config)

    x = torch.randint(0, config.vocab_size, (2, 10))  # (batch, time)
    x_lengths = torch.tensor([10, 8])  # 两个样本的实际长度

    output, mask = encoder(x, x_lengths)

    assert output.shape == (2, 10, config.hidden_dim)
    assert mask.shape == (2, 1, 10)


def test_text_encoder_masking():
    """T-046: 不同长度输入的 mask 正确（短序列被 mask 的位置输出为 0）"""
    config = TextEncoderConfig()
    encoder = TextEncoder(config)

    # 样本1长度10，样本2长度8
    x = torch.randint(0, config.vocab_size, (2, 10))
    x_lengths = torch.tensor([10, 8])

    output, mask = encoder(x, x_lengths)

    # 样本2的有效位置（mask=1）和填充位置（mask=0）
    # mask 应该是 (batch, 1, time)
    # 前8个位置 mask=1，后2个位置 mask=0
    assert mask.shape == (2, 1, 10)
    # 检查第二个样本的 mask：前8个为1，后2个为0
    assert torch.all(mask[1, 0, :8] == 1)
    assert torch.all(mask[1, 0, 8:] == 0)
