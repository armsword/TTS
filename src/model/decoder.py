"""VAE 解码器 - 将潜变量解码为梅尔频谱

使用 Conv1d + ResBlock 架构替代纯 MLP，能够建模时间依赖关系，
生成连贯的梅尔频谱而非噪声。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DecoderConfig


class ResBlock1D(nn.Module):
    """1D 残差块，包含膨胀卷积"""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, time)"""
        residual = x
        # Conv1d expects (B, C, T), LayerNorm expects (B, T, C)
        out = self.conv1(x)
        out = out.transpose(1, 2)
        out = self.norm1(out)
        out = out.transpose(1, 2)
        out = F.gelu(out)

        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = F.gelu(out)

        return out + residual


class Decoder(nn.Module):
    """VAE 解码器

    使用 Conv1d + 残差块将潜变量 z 解码为梅尔频谱
    """

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config

        # 输入投影：从潜变量维度映射到隐藏通道
        hidden_dim = config.in_channels * 4  # 扩展通道数增强表达能力
        self.input_proj = nn.Linear(config.in_channels, hidden_dim)

        # 多层 Conv1d 残差块（不同膨胀率捕获不同时间尺度）
        self.res_blocks = nn.ModuleList()
        dilations = [1, 2, 4, 1, 2, 4]  # 6 层，两组膨胀卷积
        for i in range(config.n_layers):
            dilation = dilations[i % len(dilations)]
            self.res_blocks.append(ResBlock1D(hidden_dim, kernel_size=5, dilation=dilation))

        # 输出投影：hidden_dim -> out_channels (mel)
        self.output_proj = nn.Conv1d(hidden_dim, config.out_channels, kernel_size=1)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """VAE 重参数化采样

        z = mu + exp(0.5 * log_var) * eps

        Args:
            mu: 均值 (batch, time, latent_dim)
            log_var: 对数方差 (batch, time, latent_dim)

        Returns:
            z: 采样结果 (batch, time, latent_dim)
        """
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * log_var) * eps
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码

        Args:
            z: 潜变量 (batch, time, in_channels)

        Returns:
            mel: 梅尔频谱 (batch, out_channels, time)
        """
        # 投影到隐藏维度 (batch, time, hidden_dim)
        x = self.input_proj(z)
        x = F.gelu(x)

        # 转为 (batch, hidden_dim, time) 以便 Conv1d 处理
        x = x.transpose(1, 2)

        # 通过残差卷积块
        for block in self.res_blocks:
            x = block(x)

        # 投影到梅尔频谱维度
        x = self.output_proj(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入 (batch, time, in_channels)

        Returns:
            mel: 梅尔频谱 (batch, out_channels, time)
        """
        return self.decode(x)
