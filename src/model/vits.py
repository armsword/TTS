"""VITS 主模型 - 条件变分自编码器用于文本到语音合成"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    TextEncoderConfig, DecoderConfig, DurationPredictorConfig,
    HiFiGANConfig, TrainConfig
)
from model.text_encoder import TextEncoder
from model.duration_predictor import DurationPredictor, regulate_length
from model.decoder import Decoder


class VITS(nn.Module):
    """VITS 主模型

    包含文本编码器、时长预测器、VAE 解码器
    """

    def __init__(self, configs: dict):
        super().__init__()

        self.text_encoder = TextEncoder(configs['text_encoder'])
        self.duration_predictor = DurationPredictor(configs['duration_predictor'])
        self.decoder = Decoder(configs['decoder'])

        # VAE 的先验分布参数
        # 将编码器输出映射到 mu 和 log_var
        self.proj_mu = nn.Linear(configs['text_encoder'].hidden_dim, configs['decoder'].in_channels)
        self.proj_log_var = nn.Linear(configs['text_encoder'].hidden_dim, configs['decoder'].in_channels)

        # 临时保存配置用于推理
        self.configs = configs

    def forward(self, phoneme_ids: torch.Tensor, phoneme_lengths: torch.Tensor,
                mel_targets: torch.Tensor = None, durations: torch.Tensor = None) -> dict:
        """训练模式前向传播

        Args:
            phoneme_ids: 音素 IDs (batch, time)
            phoneme_lengths: 每个样本的实际长度 (batch,)
            mel_targets: 目标梅尔频谱 (batch, n_mels, target_time) - 可选
            durations: 目标时长 (batch, time) - 可选，用于展长

        Returns:
            包含预测结果的字典
        """
        # 文本编码
        encoder_output, mask = self.text_encoder(phoneme_ids, phoneme_lengths)

        # 时长预测
        duration_pred = self.duration_predictor(encoder_output, mask)

        # 展长：优先使用目标时长（teacher forcing），否则使用预测时长
        if durations is not None:
            # 使用 ground truth 时长（每个音素的帧数）
            expanded_output = regulate_length(encoder_output, durations, phoneme_lengths)
        else:
            # 使用预测时长（推理时）
            expanded_output = regulate_length(encoder_output, duration_pred, phoneme_lengths)

        # VAE: 计算 mu 和 log_var
        mu = self.proj_mu(expanded_output)
        log_var = self.proj_log_var(expanded_output)

        # 重参数化采样
        z = self.decoder.reparameterize(mu, log_var)

        # 解码到梅尔频谱
        mel_output = self.decoder.decode(z)

        # 如果有目标梅尔频谱，调整输出长度匹配目标
        if mel_targets is not None:
            target_len = mel_targets.shape[2]
            mel_output_len = mel_output.shape[2]
            if mel_output_len != target_len:
                if mel_output_len > target_len:
                    mel_output = mel_output[:, :, :target_len]
                else:
                    mel_output = F.pad(mel_output, (0, target_len - mel_output_len))

        return {
            'mel_output': mel_output,
            'duration_pred': duration_pred,
            'mu': mu,
            'log_var': log_var,
            'z': z,
        }

    def infer(self, phoneme_ids: torch.Tensor) -> torch.Tensor:
        """推理模式

        Args:
            phoneme_ids: 音素 IDs (batch, time)

        Returns:
            mel_output: 预测的梅尔频谱 (batch, n_mels, time)
        """
        batch_size, time_steps = phoneme_ids.shape
        phoneme_lengths = torch.full((batch_size,), time_steps, device=phoneme_ids.device)

        # 文本编码
        encoder_output, mask = self.text_encoder(phoneme_ids, phoneme_lengths)

        # 时长预测
        duration_pred = self.duration_predictor(encoder_output, mask)

        # 展长（推理时用预测时长）
        expanded_output = regulate_length(encoder_output, duration_pred, phoneme_lengths)

        # VAE: 计算 mu 和 log_var（使用确定性输出，log_var=0）
        mu = self.proj_mu(expanded_output)
        log_var = torch.zeros_like(mu)

        # 重参数化采样
        z = self.decoder.reparameterize(mu, log_var)

        # 解码到梅尔频谱
        mel_output = self.decoder.decode(z)

        return mel_output
