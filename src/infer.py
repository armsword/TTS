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


class TTSInferencer:
    """TTS 推理引擎

    加载 VITS 和 HiFi-GAN 模型，执行文本到语音合成
    """

    def __init__(
        self,
        vits_path: Optional[str] = None,
        hifigan_path: Optional[str] = None,
        device: str = "auto"
    ):
        """初始化推理引擎

        Args:
            vits_path: VITS 模型路径（可选）
            hifigan_path: HiFi-GAN 模型路径（可选）
            device: 设备，"auto" 自动选择
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

        # 加载权重（如果提供）
        if vits_path and os.path.exists(vits_path):
            state_dict = torch.load(vits_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in state_dict:
                self.model.load_state_dict(state_dict["model_state_dict"])
            else:
                self.model.load_state_dict(state_dict)

        if hifigan_path and os.path.exists(hifigan_path):
            state_dict = torch.load(hifigan_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in state_dict:
                self.hifigan.load_state_dict(state_dict["model_state_dict"])
            else:
                self.hifigan.load_state_dict(state_dict)

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

            # HiFi-GAN 将梅尔频谱转换为波形
            waveform = self.hifigan(mel_output)

        # 转换为 numpy 数组
        waveform = waveform.squeeze().cpu().numpy()

        # 归一化到 [-1, 1]（HiFi-GAN 输出已经在范围内）
        waveform = np.clip(waveform, -1.0, 1.0)

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
