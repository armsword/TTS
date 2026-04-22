"""T-058~T-064: VAE 解码器测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from model.decoder import Decoder
from config import DecoderConfig


def test_decoder_instantiation():
    """T-058: Decoder(config) 能实例化"""
    config = DecoderConfig()
    decoder = Decoder(config)
    assert decoder is not None


def test_decoder_forward():
    """T-060: forward(z) 输入 (2, 50, 96)，输出 shape (2, 80, 50)"""
    config = DecoderConfig()
    decoder = Decoder(config)

    z = torch.randn(2, 50, config.in_channels)  # (batch, time, channels)
    output = decoder(z)

    assert output.shape == (2, config.out_channels, 50)


def test_reparameterize():
    """T-062: VAE 采样逻辑——给定 mu 和 log_var，reparameterize(mu, log_var) 输出 shape 与 mu 一致"""
    config = DecoderConfig()
    decoder = Decoder(config)

    # 使用与 decoder 输入一致的维度
    latent_dim = config.in_channels
    mu = torch.randn(2, 50, latent_dim)
    log_var = torch.randn(2, 50, latent_dim)

    z = decoder.reparameterize(mu, log_var)

    assert z.shape == mu.shape
    # 验证采样值在合理范围内
    assert not torch.allclose(z, mu)  # 不应该直接返回 mu


def test_decoder_with_vae_sampling():
    """测试 VAE 重参数化采样在解码器中的工作"""
    config = DecoderConfig()
    decoder = Decoder(config)

    # 使用与 decoder 输入一致的维度
    latent_dim = config.in_channels
    mu = torch.randn(2, 50, latent_dim)
    log_var = torch.randn(2, 50, latent_dim)

    # 重参数化采样
    z = decoder.reparameterize(mu, log_var)

    # 解码
    output = decoder.decode(z)

    assert output.shape == (2, config.out_channels, 50)
