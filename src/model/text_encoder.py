"""文本编码器 - 将音素序列编码为隐藏表示"""
import torch
import torch.nn as nn
import math
from config import TextEncoderConfig


class PositionalEncoding(nn.Module):
    """位置编码 - 为序列中的每个位置添加位置信息"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """添加位置编码

        Args:
            x: 输入 tensor (batch, time, d_model)

        Returns:
            添加位置编码后的 tensor
        """
        return x + self.pe[:, :x.size(1), :]


class TextEncoder(nn.Module):
    """文本编码器

    将音素序列通过嵌入层、位置编码和 Transformer 编码器转换为隐藏表示
    """

    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.config = config

        # 音素嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_dim,
            padding_idx=0  # 使用 0 作为 padding
        )

        # 位置编码
        self.positional_encoding = PositionalEncoding(config.hidden_dim)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor) -> tuple:
        """前向传播

        Args:
            x: 输入音素 IDs (batch, time)
            x_lengths: 每个样本的实际长度 (batch,)

        Returns:
            output: 编码后的隐藏表示 (batch, time, hidden_dim)
            mask: padding mask (batch, 1, time) - 有效位置为 1，填充位置为 0
        """
        batch_size, max_len = x.size()

        # 嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.config.hidden_dim)
        x = self.positional_encoding(x)

        # 创建 padding mask
        # mask[i, j] = True 表示位置 j 需要被 mask 掉（不参与注意力计算）
        key_padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=x.device)
        for i, length in enumerate(x_lengths):
            key_padding_mask[i, length:] = True

        # Transformer 编码
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # 创建输出 mask（有效位置为 1，填充位置为 0）
        # 用于后续模块区分有效和填充位置
        mask = torch.ones(batch_size, 1, max_len, device=x.device, dtype=torch.float)
        for i, length in enumerate(x_lengths):
            mask[i, :, length:] = 0

        return x, mask
