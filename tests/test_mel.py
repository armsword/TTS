"""T-036~T-041: 梅尔频谱提取测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from audio.mel import MelSpectrogramExtractor, wav_to_mel
from config import AudioConfig


def test_mel_spectrogram_extractor_instantiation():
    """T-036: MelSpectrogramExtractor(config) 能实例化"""
    config = AudioConfig()
    extractor = MelSpectrogramExtractor(config)
    assert extractor is not None
    assert extractor.config.sample_rate == 22050


def test_wav_to_mel_output_shape():
    """T-038: wav_to_mel(waveform) 输入 (1, 22050) 随机波形，输出 shape 为 (80, T) 且 T > 0"""
    config = AudioConfig()
    extractor = MelSpectrogramExtractor(config)

    # 创建 1 秒随机波形
    waveform = torch.randn(1, 22050)
    mel = wav_to_mel(waveform, extractor)

    assert mel.shape[0] == 80  # n_mels
    assert mel.shape[1] > 0   # 时间步 T > 0


def test_extract_from_wav_file(tmp_path):
    """T-040: extract(wav_path) 能从实际 wav 文件提取梅尔频谱，shape 为 (80, T)"""
    import scipy.io.wavfile as wavfile

    config = AudioConfig()

    # 创建测试 wav 文件
    sample_rate = 22050
    duration = 1.0  # 1 秒
    t = np.linspace(0, duration, int(sample_rate * duration))
    # 440 Hz 正弦波
    waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    wav_path = tmp_path / "test.wav"
    wavfile.write(str(wav_path), sample_rate, waveform)

    extractor = MelSpectrogramExtractor(config)
    mel = extractor.extract(str(wav_path))

    assert mel.shape[0] == 80
    assert mel.shape[1] > 0
