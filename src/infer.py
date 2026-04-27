"""TTS 推理引擎"""
import os
import sys
import torch
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from model.vits import VITS
from model.hifigan import HiFiGAN
from text.symbols import text_to_sequence
from config import (
    TextEncoderConfig, DecoderConfig, DurationPredictorConfig,
    HiFiGANConfig, TrainConfig
)

# 尝试导入 librosa用于 Griffin-Lim
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class TTSInferencer:
    """TTS 推理引擎

    加载 VITS 和 HiFi-GAN 模型，执行文本到语音合成
    """

    def __init__(
        self,
        vits_path: Optional[str] = None,
        hifigan_path: Optional[str] = None,
        device: str = "auto",
        checkpoint_path: Optional[str] = None,
    ):
        """初始化推理引擎

        Args:
            vits_path: VITS 模型路径（可选，已废弃，推荐用 checkpoint_path）
            hifigan_path: HiFi-GAN 模型路径（可选）
            device: 设备，"auto" 自动选择
            checkpoint_path: checkpoint 路径，会自动从中提取 model_state_dict
        """
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                # MPS 不支持 transformer 算子，强制使用 CPU
                print("[TTSInferencer] MPS detected but using CPU due to PyTorch limitation")
                self.device = "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # 创建模型配置
        self.configs = {
            'text_encoder': TextEncoderConfig(),
            'decoder': DecoderConfig(),
            'duration_predictor': DurationPredictorConfig(),
            'hifigan': HiFiGANConfig(),
            'train': TrainConfig(),
        }

        # 创建模型
        self.model = VITS(self.configs).to(self.device)
        self.hifigan = HiFiGAN(HiFiGANConfig()).to(self.device)

        # 加载权重优先级：checkpoint_path > vits_path
        loaded = False
        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in state_dict:
                self.model.load_state_dict(state_dict["model_state_dict"])
                epoch = state_dict.get("epoch", -1) + 1
                loss = state_dict.get("train_loss", -1)
                print(f"[TTSInferencer] Loaded checkpoint: epoch={epoch}, loss={loss:.4f}")
            else:
                self.model.load_state_dict(state_dict)
            loaded = True
        elif vits_path and os.path.exists(vits_path):
            state_dict = torch.load(vits_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in state_dict:
                self.model.load_state_dict(state_dict["model_state_dict"])
            else:
                self.model.load_state_dict(state_dict)
            loaded = True

        if not loaded:
            print("[TTSInferencer] Warning: No checkpoint loaded, using random weights")

        # 记录当前 checkpoint 路径
        self._checkpoint_path = checkpoint_path if checkpoint_path else vits_path

        if hifigan_path and os.path.exists(hifigan_path):
            state_dict = torch.load(hifigan_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in state_dict:
                self.hifigan.load_state_dict(state_dict["model_state_dict"])
            else:
                self.hifigan.load_state_dict(state_dict)
            self.use_hifigan = True
        else:
            self.use_hifigan = False
            if not HAS_LIBROSA:
                print("[TTSInferencer] Warning: librosa not installed and HiFi-GAN not available. Using Griffin-Lim.")

        # 设置为评估模式
        self.model.eval()
        self.hifigan.eval()

    def synthesize(self, text: str, language: str = "en") -> np.ndarray:
        """合成语音

        Args:
            text: 输入文本
            language: 语言代码 ("en" 或 "zh")

        Returns:
            波形数组 (1D, dtype=float32), 值范围 [-1, 1]
        """
        # 文本转音素 ID
        phoneme_ids = text_to_sequence(text, language)
        phoneme_ids_tensor = torch.tensor([phoneme_ids], dtype=torch.long, device=self.device)

        # VITS 推理
        with torch.no_grad():
            mel_output = self.model.infer(phoneme_ids_tensor)

            # 使用 HiFi-GAN 或 Griffin-Lim 转换为波形
            if self.use_hifigan:
                waveform = self.hifigan(mel_output)
                waveform = waveform.squeeze().cpu().numpy()
                waveform = np.clip(waveform, -1.0, 1.0)
            else:
                # Griffin-Lim 替代方案
                mel_db = mel_output.squeeze().cpu().numpy()  # dB 格式
                if HAS_LIBROSA:
                    # 将 dB 转换回线性功率谱
                    # dB = 10 * log10(power) => power = 10^(dB/10)
                    mel_power = np.power(10.0, mel_db / 10.0)

                    waveform = librosa.feature.inverse.mel_to_audio(
                        mel_power,
                        sr=22050,
                        n_fft=1024,
                        hop_length=256,
                        win_length=1024
                    )
                else:
                    # 真正的 Griffin-Lim 实现
                    waveform = self._griffin_lim(mel_db)

        return waveform.astype(np.float32)

    def _griffin_lim(self, mel_db: np.ndarray, n_iter: int = 60) -> np.ndarray:
        """Griffin-Lim 算法：从梅尔频谱重建音频波形

        Args:
            mel_db: 梅尔频谱 (n_mels, time)，dB 格式
            n_iter: Griffin-Lim 迭代次数

        Returns:
            waveform: 音频波形 (samples,)
        """
        import scipy.signal

        n_fft = 1024
        hop_length = 256
        win_length = 1024
        n_mels = mel_db.shape[0]
        sr = 22050

        # 1. dB -> 线性功率
        mel_power = np.power(10.0, mel_db / 10.0)
        mel_power = np.maximum(mel_power, 1e-10)

        # 2. 构造梅尔滤波器组并伪逆，将 mel 频谱转回线性频谱
        n_freqs = n_fft // 2 + 1
        mel_basis = self._mel_filter_bank(sr, n_fft, n_mels)  # (n_mels, n_freqs)
        mel_basis_pinv = np.linalg.pinv(mel_basis)  # (n_freqs, n_mels)

        # mel -> 线性幅度谱
        magnitude = np.dot(mel_basis_pinv, mel_power)  # (n_freqs, time)
        magnitude = np.maximum(magnitude, 0.0)
        magnitude = np.sqrt(magnitude)  # power -> amplitude

        # 3. Griffin-Lim 迭代重建相位
        window = scipy.signal.get_window('hann', win_length)
        n_frames = magnitude.shape[1]
        expected_len = (n_frames - 1) * hop_length + win_length

        # 随机初始化相位
        phase = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
        stft = magnitude * phase

        for _ in range(n_iter):
            # ISTFT
            waveform = np.zeros(expected_len)
            window_sum = np.zeros(expected_len)
            for t in range(n_frames):
                start = t * hop_length
                frame = np.fft.irfft(stft[:, t], n=n_fft)[:win_length]
                waveform[start:start + win_length] += frame * window
                window_sum[start:start + win_length] += window ** 2

            # 归一化
            nonzero = window_sum > 1e-8
            waveform[nonzero] /= window_sum[nonzero]

            # STFT 重新估计相位
            for t in range(n_frames):
                start = t * hop_length
                frame = waveform[start:start + win_length] * window
                spectrum = np.fft.rfft(frame, n=n_fft)
                phase = np.exp(1j * np.angle(spectrum))
                stft[:, t] = magnitude[:, t] * phase

        # 最终 ISTFT
        waveform = np.zeros(expected_len)
        window_sum = np.zeros(expected_len)
        for t in range(n_frames):
            start = t * hop_length
            frame = np.fft.irfft(stft[:, t], n=n_fft)[:win_length]
            waveform[start:start + win_length] += frame * window
            window_sum[start:start + win_length] += window ** 2

        nonzero = window_sum > 1e-8
        waveform[nonzero] /= window_sum[nonzero]

        # 归一化到 [-1, 1]
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val * 0.95

        return waveform.astype(np.float32)

    @staticmethod
    def _mel_filter_bank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
        """构造梅尔滤波器组

        Args:
            sr: 采样率
            n_fft: FFT 窗口大小
            n_mels: 梅尔频带数

        Returns:
            mel_basis: 梅尔滤波器组 (n_mels, n_fft//2+1)
        """
        n_freqs = n_fft // 2 + 1

        # Hz -> Mel
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_min = hz_to_mel(0)
        mel_max = hz_to_mel(sr / 2)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        freqs = np.linspace(0, sr / 2, n_freqs)
        mel_basis = np.zeros((n_mels, n_freqs))

        for i in range(n_mels):
            lower = hz_points[i]
            center = hz_points[i + 1]
            upper = hz_points[i + 2]

            for j, freq in enumerate(freqs):
                if lower <= freq <= center and center > lower:
                    mel_basis[i, j] = (freq - lower) / (center - lower)
                elif center < freq <= upper and upper > center:
                    mel_basis[i, j] = (upper - freq) / (upper - center)

        return mel_basis

    @staticmethod
    def to_int16_wav(waveform: np.ndarray) -> np.ndarray:
        """将 float32 波形转换为 int16 格式（浏览器兼容）

        Args:
            waveform: float32 波形

        Returns:
            int16 波形
        """
        # 归一化到 [-1, 1]
        max_val = np.abs(waveform).max()
        if max_val > 1e-6:
            waveform = waveform / max_val * 0.95
        # 转 int16
        return (waveform * 32767).astype(np.int16)

    def synthesize_to_file(self, text: str, output_path: str, language: str = "en"):
        """合成语音并保存到文件

        Args:
            text: 输入文本
            output_path: 输出 wav 文件路径
            language: 语言代码
        """
        import scipy.io.wavfile as wavfile

        waveform = self.synthesize(text, language)

        # 保存为 int16 wav 文件（浏览器兼容）
        wavfile.write(output_path, 22050, self.to_int16_wav(waveform))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TTS Inference")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, required=True, help="Output wav file path")
    parser.add_argument("--language", type=str, default="en", help="Language (en or zh)")
    parser.add_argument("--vits_path", type=str, default=None, help="VITS model path")
    parser.add_argument("--hifigan_path", type=str, default=None, help="HiFi-GAN model path")
    args = parser.parse_args()

    inferencer = TTSInferencer(args.vits_path, args.hifigan_path)
    inferencer.synthesize_to_file(args.text, args.output, args.language)
    print(f"Saved to {args.output}")
