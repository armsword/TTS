"""HiFi-GAN 声码器 - 将梅尔频谱转换为高质量音频波形"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import HiFiGANConfig


class ResBlock(nn.Module):
    """残差块 (ResBlock)

    包含多个膨胀卷积层和残差连接
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilations: tuple = (1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList()
        for dilation in dilations:
            self.convs.append(
                nn.Conv1d(
                    channels, channels,
                    kernel_size=kernel_size,
                    padding=dilation * (kernel_size - 1) // 2,
                    dilation=dilation
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入 (batch, channels, time)

        Returns:
            输出 (batch, channels, time)
        """
        for conv in self.convs:
            h = F.relu(conv(x))
            x = x + h  # 残差连接
        return x


class HiFiGAN(nn.Module):
    """HiFi-GAN 声码器

    将梅尔频谱转换为高质量音频波形
    """

    def __init__(self, config: HiFiGANConfig):
        super().__init__()
        self.config = config

        # 初始卷积：将梅尔频谱维度映射到隐藏通道
        self.input_conv = nn.Conv1d(
            80, 256,  # n_mels -> hidden channels
            kernel_size=7,
            padding=3
        )

        # 上采样层和残差块
        self.upsamples = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        channels = 256
        for i, rate in enumerate(config.upsample_rates):
            # 每个上采样后接一个残差块
            self.upsamples.append(
                nn.ConvTranspose1d(
                    channels, channels // 2,
                    kernel_size=rate * 2,
                    stride=rate,
                    padding=rate // 2
                )
            )
            channels = channels // 2
            self.resblocks.append(
                ResBlock(channels, kernel_size=3, dilations=(1, 3, 5))
            )

        # 输出卷积
        self.output_conv = nn.Conv1d(
            channels, 1,  # -> mono waveform
            kernel_size=7,
            padding=3
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            mel: 梅尔频谱 (batch, n_mels, time)

        Returns:
            waveform: 波形 (batch, 1, time')
        """
        # 初始卷积
        x = self.input_conv(mel)

        # 上采样 + 残差块
        for upsample, resblock in zip(self.upsamples, self.resblocks):
            x = F.relu(x)
            x = upsample(x)
            x = resblock(x)

        # 输出卷积 + tanh 激活
        x = self.output_conv(x)
        x = torch.tanh(x)

        return x
