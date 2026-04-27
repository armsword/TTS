# Griffin-Lim 相位重建算法原理文档

## 1. 概述

Griffin-Lim 是一种从幅度谱重建时域信号的经典算法。在 TTS 中，模型输出的梅尔频谱只有幅度信息（80 维），丢失了相位信息（513 维复数），Griffin-Lim 通过迭代方式近似恢复相位。

## 2. 核心问题

**短时傅里叶变换 (STFT)** 将信号分解为 复数频谱 = 幅度 × e^(jφ)：

```
X(t, f) = |X(t, f)| × e^(j·φ(t,f))
         ─────────   ──────────────
          幅度 (已知)   相位 (丢失)
```

梅尔频谱进一步将 513 个频率 bin 压缩到 80 个 mel bin。要重建音频需要：
1. **mel → 线性频谱**：通过梅尔滤波器组的伪逆矩阵
2. **补全相位**：这就是 Griffin-Lim 要做的事

## 3. 算法流程

### 3.1 初始化

```python
# 1. dB → 线性功率谱
mel_power = 10^(mel_dB / 10)

# 2. mel → 线性频谱 (通过伪逆)
mel_basis = build_mel_filterbank()           # (80, 513)
mel_basis_pinv = pseudo_inverse(mel_basis)   # (513, 80)
magnitude = sqrt(mel_basis_pinv @ mel_power) # (513, T)

# 3. 随机初始化相位
phase = e^(j × 2π × random)
stft = magnitude × phase
```

### 3.2 迭代（64 次）

```
for each iteration:
    ┌──────────────────────────────────────────┐
    │  ISTFT: stft → waveform                  │  用当前估计还原波形
    │  STFT:  waveform → new_stft              │  重新分析波形
    │  更新:  phase = angle(new_stft)           │  取新相位
    │         stft = magnitude × e^(j·phase)   │  保持原始幅度
    └──────────────────────────────────────────┘
```

每次迭代都在做投影：
- ISTFT → STFT：投影到"合法时域信号"的集合
- 替换幅度：投影到"与给定幅度一致"的集合

这是**交替投影 (Alternating Projection)** 算法，保证收敛到局部最优。

### 3.3 最终输出

```python
# 最后一次 ISTFT
waveform = ISTFT(magnitude × e^(j·phase))
# 归一化
waveform = waveform / max(|waveform|) × 0.95
```

## 4. 关键实现细节

### 4.1 Overlap-Add ISTFT

```python
for each frame t:
    frame = IFFT(stft[:, t])[:win_length]  # 逆 FFT
    waveform[t*hop : t*hop+win] += frame × window
    window_sum[t*hop : t*hop+win] += window²

waveform /= window_sum  # 归一化 overlap 部分
```

### 4.2 梅尔滤波器组构造

80 个三角滤波器在梅尔刻度等间距排列：

```python
mel_points = linspace(hz_to_mel(0), hz_to_mel(sr/2), n_mels+2)
hz_points = mel_to_hz(mel_points)

for i in range(n_mels):
    # 三角滤波器: lower → center → upper
    filter[i] = triangle(hz_points[i], hz_points[i+1], hz_points[i+2])
```

### 4.3 伪逆矩阵

```
mel = mel_basis @ linear_spectrum   # (80, 513) × (513, T) → (80, T)
linear_spectrum ≈ pinv(mel_basis) @ mel  # (513, 80) × (80, T) → (513, T)
```

伪逆是最小二乘意义上的最优逆映射。由于 mel 滤波器有重叠，信息损失有限。

## 5. 迭代次数的影响

| 迭代次数 | 质量 | 耗时 |
|---------|------|------|
| 1 | 很差（随机相位） | 快 |
| 16 | 可接受 | 中 |
| 32 | 较好 | 中 |
| **64** | **好（本项目默认）** | **稍慢** |
| 128 | 略有提升 | 慢 |

收益递减：大部分相位在前 30 次迭代内收敛。

## 6. 局限性

1. **无法恢复真实相位**：只能找到"与给定幅度一致"的某个相位，不是原始相位
2. **音质上限有限**：重建音频有"金属味"或"机器人味"
3. **mel 压缩损失**：80 bin → 513 bin 的伪逆是有损的
4. **计算较慢**：64 次 STFT/ISTFT 迭代

## 7. vs 神经声码器

| | Griffin-Lim | HiFi-GAN |
|--|------------|----------|
| 原理 | 信号处理（迭代投影） | 深度学习（GAN） |
| 训练 | 不需要 | 需要对抗训练 |
| 音质 | 中等（机器人味） | 高（接近真人） |
| 速度 | 中等 | 快（GPU 上） |
| 依赖 | 只需 numpy/scipy | 需要训练好的模型权重 |

本项目当前使用 Griffin-Lim 作为过渡方案，后续训练 HiFi-GAN 后可切换。
