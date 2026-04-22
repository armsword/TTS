"""T-073~T-081: HiFi-GAN 声码器测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from model.hifigan import HiFiGAN, ResBlock
from config import HiFiGANConfig


def test_hifigan_instantiation():
    """T-073: HiFi-GAN 能实例化，参数量在 1-2M 范围"""
    config = HiFiGANConfig()
    model = HiFiGAN(config)

    params = sum(p.numel() for p in model.parameters())
    params_m = params / 1e6

    print(f"HiFi-GAN 参数总量: {params_m:.2f}M")
    assert params_m < 2, f"HiFi-GAN 参数过多: {params_m:.2f}M"


def test_resblock_input_output_shape():
    """T-075: ResBlock 子模块输入输出 shape 一致"""
    config = HiFiGANConfig()
    model = HiFiGAN(config)

    # 第一个残差块的通道数是 128（经过第一次上采样后）
    x = torch.randn(2, 128, 100)  # (batch, channels, time)

    # 获取第一个残差块
    resblock = model.resblocks[0]
    out = resblock(x)

    assert out.shape == x.shape


def test_hifigan_forward():
    """T-077: forward(mel) 输入 (2, 80, 50)，输出 shape (2, 1, 50*256) = (2, 1, 12800)"""
    config = HiFiGANConfig()
    model = HiFiGAN(config)

    mel = torch.randn(2, 80, 50)  # (batch, n_mels, time)
    waveform = model(mel)

    expected_length = 50 * 8 * 8 * 2 * 2  # 50 * 256 = 12800
    assert waveform.shape == (2, 1, expected_length)


def test_hifigan_output_range():
    """T-079: 输出波形值范围在 [-1, 1]"""
    config = HiFiGANConfig()
    model = HiFiGAN(config)

    mel = torch.randn(2, 80, 50)
    waveform = model(mel)

    assert waveform.min() >= -1.0
    assert waveform.max() <= 1.0
