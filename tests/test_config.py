import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import (
    AudioConfig, TextEncoderConfig, DecoderConfig,
    DurationPredictorConfig, HiFiGANConfig, TrainConfig, load_config
)


def test_audio_config_defaults():
    """T-004: AudioConfig 能以默认值实例化，sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80"""
    config = AudioConfig()
    assert config.sample_rate == 22050
    assert config.n_fft == 1024
    assert config.hop_length == 256
    assert config.n_mels == 80


def test_text_encoder_config_defaults():
    """T-006: TextEncoderConfig 默认值 vocab_size=256, hidden_dim=192, n_layers=2, n_heads=4"""
    config = TextEncoderConfig()
    assert config.vocab_size == 256
    assert config.hidden_dim == 192
    assert config.n_layers == 2
    assert config.n_heads == 4


def test_decoder_config_defaults():
    """T-008: DecoderConfig 默认值 in_channels=96, out_channels=80, n_layers=6"""
    config = DecoderConfig()
    assert config.in_channels == 96
    assert config.out_channels == 80
    assert config.n_layers == 6


def test_duration_predictor_config_defaults():
    """T-010: DurationPredictorConfig 默认值 in_channels=192, n_layers=2"""
    config = DurationPredictorConfig()
    assert config.in_channels == 192
    assert config.n_layers == 2


def test_hifigan_config_defaults():
    """T-012: HiFiGANConfig 默认值 upsample_rates=(8,8,2,2), 乘积等于 256"""
    config = HiFiGANConfig()
    assert config.upsample_rates == (8, 8, 2, 2)
    result = 1
    for r in config.upsample_rates:
        result *= r
    assert result == 256


def test_train_config_defaults():
    """T-014: TrainConfig 默认值 batch_size=16, lr=1e-4, epochs=200"""
    config = TrainConfig()
    assert config.batch_size == 16
    assert config.lr == 1e-4
    assert config.epochs == 200


def test_load_config(tmp_path):
    """T-016: load_config(path) 能从 JSON 文件加载并返回各 Config 对象"""
    import json

    config_data = {
        "audio": {"sample_rate": 22050, "n_fft": 1024, "hop_length": 256, "n_mels": 80},
        "text_encoder": {"vocab_size": 256, "hidden_dim": 192, "n_layers": 2, "n_heads": 4},
        "decoder": {"in_channels": 96, "out_channels": 80, "n_layers": 6},
        "duration_predictor": {"in_channels": 192, "n_layers": 2},
        "hifigan": {"upsample_rates": [8, 8, 2, 2]},
        "train": {"batch_size": 16, "lr": 1e-4, "epochs": 200}
    }

    config_file = tmp_path / "test_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f)

    configs = load_config(str(config_file))

    assert isinstance(configs['audio'], AudioConfig)
    assert configs['audio'].sample_rate == 22050
    assert isinstance(configs['text_encoder'], TextEncoderConfig)
    assert isinstance(configs['decoder'], DecoderConfig)
    assert isinstance(configs['duration_predictor'], DurationPredictorConfig)
    assert isinstance(configs['hifigan'], HiFiGANConfig)
    assert isinstance(configs['train'], TrainConfig)
