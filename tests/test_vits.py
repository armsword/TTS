"""T-065~T-072: VITS 主模型测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from model.vits import VITS
from config import (
    TextEncoderConfig, DecoderConfig, DurationPredictorConfig,
    HiFiGANConfig, TrainConfig, load_config
)


def test_vits_instantiation():
    """T-065: VITS(config) 能实例化，包含 text_encoder / duration_predictor / decoder 子模块"""
    configs = {
        'text_encoder': TextEncoderConfig(),
        'decoder': DecoderConfig(),
        'duration_predictor': DurationPredictorConfig(),
        'hifigan': HiFiGANConfig(),
        'train': TrainConfig(),
    }
    model = VITS(configs)

    assert hasattr(model, 'text_encoder')
    assert hasattr(model, 'duration_predictor')
    assert hasattr(model, 'decoder')


def test_vits_forward_training():
    """T-067: forward() 训练模式，输入 phoneme_ids (2, 10) + mel_targets (2, 80, 50) + 对应 lengths"""
    configs = {
        'text_encoder': TextEncoderConfig(),
        'decoder': DecoderConfig(),
        'duration_predictor': DurationPredictorConfig(),
        'hifigan': HiFiGANConfig(),
        'train': TrainConfig(),
    }
    model = VITS(configs)
    model.train()

    # 准备输入
    phoneme_ids = torch.randint(1, 256, (2, 10))  # (batch, time)
    phoneme_lengths = torch.tensor([10, 8])
    mel_targets = torch.randn(2, 80, 50)  # (batch, n_mels, time)

    # 前向传播
    output = model(phoneme_ids, phoneme_lengths, mel_targets)

    # 验证输出包含必要的字段
    assert 'mel_output' in output
    assert 'duration_pred' in output
    assert 'mu' in output
    assert 'log_var' in output
    assert 'z' in output


def test_vits_infer():
    """T-069: infer() 推理模式，输入 phoneme_ids (1, 10)，返回 mel shape (1, 80, M) 且 M > 0"""
    configs = {
        'text_encoder': TextEncoderConfig(),
        'decoder': DecoderConfig(),
        'duration_predictor': DurationPredictorConfig(),
        'hifigan': HiFiGANConfig(),
        'train': TrainConfig(),
    }
    model = VITS(configs)
    model.eval()

    phoneme_ids = torch.randint(1, 256, (1, 10))

    with torch.no_grad():
        mel_output = model.infer(phoneme_ids)

    assert mel_output.shape[0] == 1
    assert mel_output.shape[1] == 80
    assert mel_output.shape[2] > 0


def test_vits_parameter_count():
    """T-071: 模型总参数量在合理范围内"""
    configs = {
        'text_encoder': TextEncoderConfig(),
        'decoder': DecoderConfig(),
        'duration_predictor': DurationPredictorConfig(),
        'hifigan': HiFiGANConfig(),
        'train': TrainConfig(),
    }
    model = VITS(configs)

    params = sum(p.numel() for p in model.parameters())
    params_m = params / 1e6

    print(f"VITS 参数总量: {params_m:.2f}M")
    # VITS 约 1.26M + HiFi-GAN 约 1.5M = 约 2.7M
    assert params_m <= 10, f"参数过多: {params_m:.2f}M"
