"""梅尔频谱提取模块 - 将音频波形转换为梅尔频谱表示"""
import torch
import numpy as np
from typing import Optional
from config import AudioConfig


class MelSpectrogramExtractor:
    """梅尔频谱提取器

    使用 torchaudio 的 MelSpectrogram 变换将音频波形转换为梅尔频谱
    """

    def __init__(self, config: AudioConfig):
        """初始化梅尔频谱提取器

        Args:
            config: AudioConfig 配置对象
        """
        self.config = config

        # 动态导入 torchaudio（避免在缺少 torchcodec 时报错）
        try:
            import torchaudio
            self.torchaudio = torchaudio
        except ImportError:
            self.torchaudio = None

        # 创建 MelSpectrogram 变换
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=0,
            f_max=config.sample_rate / 2,
        )

        # 功率到分贝的变换
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=80,
        )

    def extract(self, wav_path: str) -> torch.Tensor:
        """从 wav 文件提取梅尔频谱

        Args:
            wav_path: wav 文件路径

        Returns:
            梅尔频谱 tensor，shape (n_mels, time)
        """
        # 使用 scipy 加载 wav 文件（避免 torchcodec 依赖）
        import scipy.io.wavfile as wavfile
        sample_rate, waveform = wavfile.read(wav_path)
        waveform = waveform.astype(np.float32) / 32768.0  # 归一化到 [-1, 1]

        # 转换为 torch tensor
        waveform = torch.from_numpy(waveform).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (time,) -> (1, time)

        # 重采样到目标采样率（如果需要）
        if sample_rate != self.config.sample_rate:
            resampler = self.torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.config.sample_rate,
            )
            waveform = resampler(waveform)

        # 转换为梅尔频谱
        return wav_to_mel(waveform, self)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """将波形转换为梅尔频谱

        Args:
            waveform: 输入波形 tensor，shape (batch, time) 或 (time,)

        Returns:
            梅尔频谱 tensor，shape (n_mels, time)
        """
        return wav_to_mel(waveform, self)


def wav_to_mel(waveform: torch.Tensor, extractor: MelSpectrogramExtractor) -> torch.Tensor:
    """将波形转换为梅尔频谱

    Args:
        waveform: 输入波形 tensor
        extractor: MelSpectrogramExtractor 实例

    Returns:
        梅尔频谱 tensor，shape (n_mels, time)
    """
    # 确保是 float 类型
    if waveform.dtype != torch.float32:
        waveform = waveform.to(torch.float32)

    # 添加 batch 维度（如果需要）
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # 计算梅尔频谱
    mel = extractor.mel_transform(waveform)

    # 转换为分贝刻度
    mel_db = extractor.amplitude_to_db(mel)

    # 移除 batch 维度
    mel_db = mel_db.squeeze(0)

    return mel_db


def wav_to_mel_numpy(waveform: np.ndarray, config: AudioConfig) -> np.ndarray:
    """将 numpy 波形转换为梅尔频谱

    Args:
        waveform: 输入波形 numpy array
        config: AudioConfig 配置对象

    Returns:
        梅尔频谱 numpy array，shape (n_mels, time)
    """
    waveform_tensor = torch.from_numpy(waveform).float()
    extractor = MelSpectrogramExtractor(config)
    mel = wav_to_mel(waveform_tensor, extractor)
    return mel.numpy()
