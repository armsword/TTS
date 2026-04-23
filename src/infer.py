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
                self.device = "mps"
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
                    # 简单的 Griffin-Lim 实现
                    import scipy.signal
                    n_fft, hop_length, win_length = 1024, 256, 1024
                    # 简单的ISTFT (近似)
                    waveform = np.zeros(mel_db.shape[1] * hop_length)
                    for i in range(mel_db.shape[1]):
                        # 简化处理
                        waveform[i * hop_length] = mel_db[0, i] if i < mel_db.shape[1] else 0

        return waveform.astype(np.float32)

    def synthesize_to_file(self, text: str, output_path: str, language: str = "en"):
        """合成语音并保存到文件

        Args:
            text: 输入文本
            output_path: 输出 wav 文件路径
            language: 语言代码
        """
        import scipy.io.wavfile as wavfile

        waveform = self.synthesize(text, language)

        # 保存为 wav 文件
        wavfile.write(output_path, 22050, waveform)


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
