"""配置模块 - TTS 模型的超参数配置"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AudioConfig:
    """音频配置"""
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80


@dataclass
class TextEncoderConfig:
    """文本编码器配置"""
    vocab_size: int = 256
    hidden_dim: int = 192
    n_layers: int = 2
    n_heads: int = 4


@dataclass
class DecoderConfig:
    """VAE 解码器配置"""
    in_channels: int = 96
    out_channels: int = 80
    n_layers: int = 6


@dataclass
class DurationPredictorConfig:
    """时长预测器配置"""
    in_channels: int = 192
    n_layers: int = 2


@dataclass
class HiFiGANConfig:
    """HiFi-GAN 声码器配置"""
    upsample_rates: Tuple[int, ...] = (8, 8, 2, 2)


@dataclass
class TrainConfig:
    """训练配置"""
    batch_size: int = 16
    lr: float = 1e-4
    epochs: int = 200


def load_config(path: str):
    """从 JSON 文件加载配置"""
    import json

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return {
        'audio': AudioConfig(**data.get('audio', {})),
        'text_encoder': TextEncoderConfig(**data.get('text_encoder', {})),
        'decoder': DecoderConfig(**data.get('decoder', {})),
        'duration_predictor': DurationPredictorConfig(**data.get('duration_predictor', {})),
        'hifigan': HiFiGANConfig(**data.get('hifigan', {})),
        'train': TrainConfig(**data.get('train', {})),
    }
