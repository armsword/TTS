"""VAE 解码器 - 将潜变量解码为梅尔频谱"""
import torch
import torch.nn as nn
from config import DecoderConfig


class Decoder(nn.Module):
    """VAE 解码器

    将潜变量 z 解码为梅尔频谱参数
    """

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config

        # 6 层前馈网络（Linear + ReLU），保持维度不变
        layers = []
        for _ in range(config.n_layers):
            layers.extend([
                nn.Linear(config.in_channels, config.in_channels),
                nn.ReLU(),
            ])
        self.ff_layers = nn.Sequential(*layers)

        # 输出投影：hidden_dim -> out_channels (mel)
        self.output_proj = nn.Linear(config.in_channels, config.out_channels)

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
        # 通过前馈层
        x = self.ff_layers(z)

        # 投影到输出
        x = self.output_proj(x)

        # 转置使得 (batch, channels, time)
        x = x.transpose(1, 2)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入 (batch, time, in_channels)

        Returns:
            mel: 梅尔频谱 (batch, out_channels, time)
        """
        return self.decode(x)
