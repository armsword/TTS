"""T-116~T-123: 推理引擎测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np


def test_tts_inferencer_instantiation():
    """T-116: TTSInferencer 能实例化（不加载权重）"""
    from infer import TTSInferencer

    # 由于没有预训练权重，测试类能实例化
    inferencer = TTSInferencer.__new__(TTSInferencer)
    inferencer.model = None
    inferencer.hifigan = None

    assert inferencer is not None


def test_synthesize_basic():
    """T-118: synthesize("hello world", "en") 返回 1D numpy 数组，值在 [-1, 1]，长度 > 0"""
    from infer import TTSInferencer
    from model.vits import VITS
    from model.hifigan import HiFiGAN
    from config import (
        TextEncoderConfig, DecoderConfig, DurationPredictorConfig,
        HiFiGANConfig, TrainConfig
    )

    # 创建 mock 模型
    configs = {
        'text_encoder': TextEncoderConfig(),
        'decoder': DecoderConfig(),
        'duration_predictor': DurationPredictorConfig(),
        'hifigan': HiFiGANConfig(),
        'train': TrainConfig(),
    }
    model = VITS(configs)
    hifigan = HiFiGAN(HiFiGANConfig())

    inferencer = TTSInferencer.__new__(TTSInferencer)
    inferencer.model = model
    inferencer.hifigan = hifigan
    inferencer.device = "cpu"

    # 测试合成
    waveform = inferencer.synthesize("hello world", "en")

    assert isinstance(waveform, np.ndarray)
    assert waveform.ndim == 1
    assert len(waveform) > 0
    assert waveform.min() >= -1.0
    assert waveform.max() <= 1.0


def test_synthesize_to_file(tmp_path):
    """T-120: synthesize_to_file("hello", "output.wav") 生成有效 wav 文件"""
    from infer import TTSInferencer
    from model.vits import VITS
    from model.hifigan import HiFiGAN
    from config import (
        TextEncoderConfig, DecoderConfig, DurationPredictorConfig,
        HiFiGANConfig, TrainConfig
    )

    configs = {
        'text_encoder': TextEncoderConfig(),
        'decoder': DecoderConfig(),
        'duration_predictor': DurationPredictorConfig(),
        'hifigan': HiFiGANConfig(),
        'train': TrainConfig(),
    }
    model = VITS(configs)
    hifigan = HiFiGAN(HiFiGANConfig())

    inferencer = TTSInferencer.__new__(TTSInferencer)
    inferencer.model = model
    inferencer.hifigan = hifigan
    inferencer.device = "cpu"

    output_path = tmp_path / "output.wav"
    inferencer.synthesize_to_file("hello", str(output_path), "en")

    assert output_path.exists()

    # 验证 wav 文件可读
    import scipy.io.wavfile as wavfile
    sample_rate, data = wavfile.read(output_path)
    assert sample_rate == 22050
    assert len(data) > 0
