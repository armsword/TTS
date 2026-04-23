"""时长预测器 - 预测每个音素的持续时间步数"""
import torch
import torch.nn as nn
from config import DurationPredictorConfig


class DurationPredictor(nn.Module):
    """时长预测器

    预测每个音素应该重复多少个时间步，用于将短序列展开为长序列
    """

    def __init__(self, config: DurationPredictorConfig):
        super().__init__()
        self.config = config

        # 卷积层 + ReLU
        layers = []
        in_channels = config.in_channels

        for _ in range(config.n_layers):
            layers.extend([
                nn.Conv1d(in_channels, config.in_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            ])

        self.conv_layers = nn.Sequential(*layers)

        # LayerNorm 在时间维度上
        self.layer_norm = nn.LayerNorm(config.in_channels)

        # 输出层：预测每个位置的 log duration
        self.output_linear = nn.Linear(config.in_channels, 1)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 编码器输出 (batch, time, channels)
            x_mask: mask (batch, 1, time)

        Returns:
            duration: 预测的时长 (batch, time)
        """
        # 转置以适应 Conv1d (batch, channels, time)
        x = x.transpose(1, 2)

        # 通过卷积层
        x = self.conv_layers(x)

        # 转置回来 (batch, time, channels)
        x = x.transpose(1, 2)

        # LayerNorm
        x = self.layer_norm(x)

        # 应用 mask（将填充位置置为0）
        x = x * x_mask.transpose(1, 2)  # (batch, time, 1) 乘 (batch, time, channels)

        # 预测时长
        duration = self.output_linear(x).squeeze(-1)

        # 确保 duration 非负
        duration = torch.clamp(duration.exp(), min=1.0)

        return duration


def regulate_length(encoder_output: torch.Tensor, durations: torch.Tensor,
                    phoneme_lengths: torch.Tensor = None) -> torch.Tensor:
    """根据每个音素的时长展开编码器输出

    Args:
        encoder_output: 编码器输出 (batch, time, channels)
        durations: 每个音素的时长 (batch, time)，和应为该样本的 mel 帧数
        phoneme_lengths: 每个样本的实际音素数 (batch,)

    Returns:
        展开后的输出 (batch, max_expanded_len, channels)
    """
    batch_size, time_steps, channels = encoder_output.shape

    if phoneme_lengths is not None:
        # 只使用有效音素位置的时长
        # 找到每个样本的总时长
        max_len = 0
        total_durations = []
        for b in range(batch_size):
            valid_dur = durations[b, :phoneme_lengths[b]]
            total = valid_dur.sum().int().item()
            total_durations.append(total)
            max_len = max(max_len, total)
    else:
        max_len = int(durations.sum(dim=1).max().int().item())
        total_durations = [int(durations[b, :].sum().item()) for b in range(batch_size)]
        phoneme_lengths = torch.full((batch_size,), time_steps, device=encoder_output.device)

    # 初始化输出
    output = torch.zeros(
        batch_size, max_len, channels,
        device=encoder_output.device,
        dtype=encoder_output.dtype
    )

    # 对每个样本进行展开
    for b in range(batch_size):
        current_pos = 0
        for t in range(phoneme_lengths[b]):
            dur = int(durations[b, t].item())
            if dur > 0 and current_pos + dur <= max_len:
                output[b, current_pos:current_pos + dur] = encoder_output[b, t]
                current_pos += dur

    return output
