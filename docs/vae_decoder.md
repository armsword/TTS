# VAE 解码器原理文档

## 1. 概述

VAE 解码器是 VITS 模型中将潜变量 z 转换为梅尔频谱的核心组件。本项目经历了两版架构迭代：v1 使用纯 MLP（无法建模时间依赖，输出噪声），v2 使用 **Conv1d + 膨胀残差块**，能捕获时间上下文，生成连贯的梅尔频谱。

## 2. 架构演进

### 2.1 v1 架构（已废弃）— 纯 MLP

```
z (batch, time, 96)
  ↓
[Linear(96→96) + ReLU] × 6
  ↓
Linear(96→80)
  ↓
mel (batch, 80, time)
```

**致命缺陷**：每个时间步独立处理，相邻帧之间没有信息交互。语音的梅尔频谱在时间轴上高度连续（共振峰平滑变化、协同发音效应），纯 MLP 无法捕获这种模式，输出的 mel 帧间不连贯，最终合成出噪声。

### 2.2 v2 架构（当前）— Conv1d 残差块

```
z (batch, time, 96)
  ↓
Linear(96 → 384) + GELU          # 通道扩展 4x
  ↓
转置 → (batch, 384, time)
  ↓
[ResBlock1D(384, dilation=1)] ──┐
[ResBlock1D(384, dilation=2)] ──┤  6 层残差卷积
[ResBlock1D(384, dilation=4)] ──┤  膨胀率循环 [1,2,4,1,2,4]
[ResBlock1D(384, dilation=1)] ──┤
[ResBlock1D(384, dilation=2)] ──┤
[ResBlock1D(384, dilation=4)] ──┘
  ↓
Conv1d(384 → 80, kernel=1)       # 投影到 mel 维度
  ↓
mel (batch, 80, time)
```

## 3. 核心组件详解

### 3.1 ResBlock1D — 一维膨胀残差块

每个残差块的内部结构：

```
输入 x ─────────────────────────────────┐
  ↓                                     │ (残差连接)
Conv1d(kernel=5, dilation=d, padding=auto)
  ↓                                     │
LayerNorm → GELU                        │
  ↓                                     │
Conv1d(kernel=5, dilation=1, padding=2)  │
  ↓                                     │
LayerNorm → GELU                        │
  ↓                                     │
输出 = x + residual ←───────────────────┘
```

**为什么用膨胀卷积？**

普通 Conv1d(kernel=5) 的感受野只有 5 帧。通过膨胀率 d，感受野扩大到 `kernel + (kernel-1) * (d-1)` 帧：

| 膨胀率 d | 感受野 | 对应时间 (@256 hop) |
|----------|--------|---------------------|
| 1 | 5 帧 | 58 ms |
| 2 | 9 帧 | 104 ms |
| 4 | 17 帧 | 197 ms |

6 层堆叠后的总感受野覆盖约 **数百毫秒**，足以捕获音节级别的时间依赖。

### 3.2 LayerNorm

对每个时间步的通道维度做归一化：

```
x_normalized = (x - mean) / sqrt(var + eps) * gamma + beta
```

相比 BatchNorm，LayerNorm 不依赖 batch 统计量，在小 batch 和变长序列场景下更稳定。

### 3.3 GELU 激活函数

```
GELU(x) = x * Φ(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

GELU 是 ReLU 的平滑版本，在 x=0 附近不会产生硬拐点，梯度更平滑。在 Transformer 和语音模型中表现优于 ReLU。

## 4. VAE 重参数化技巧

### 4.1 问题

VAE 需要从 $q(z|x) = \mathcal{N}(\mu, \sigma^2)$ 中采样 z，但采样操作不可微，无法反向传播。

### 4.2 解决方案

将随机性转移到外部变量 $\epsilon$：

```
z = μ + exp(0.5 * log_var) * ε,  其中 ε ~ N(0, 1)
```

这样 μ 和 log_var 都可以通过梯度下降学习。

### 4.3 训练 vs 推理

| 阶段 | 行为 | 原因 |
|------|------|------|
| 训练 | z = μ + σ * ε | 需要随机性进行正则化，让模型学到平滑的潜空间 |
| 推理 | z = μ | 使用确定性输出，避免随机噪声干扰合成质量 |

**重要**：v1 代码在推理时仍调用 reparameterize(mu, 0)，实际效果是 z = mu + 1.0 * 随机噪声，完全淹没了有用信号。这是导致"滋啦滋啦"噪声的首要原因。

## 5. 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| in_channels | 96 | 潜变量维度（编码器输出经 proj_mu 映射后的维度） |
| out_channels | 80 | 梅尔频谱维度 |
| n_layers | 6 | 残差卷积块层数 |
| hidden_dim | 384 | 内部通道数（in_channels × 4） |

**参数量**：约 8.9M（占模型总 10.35M 的 86%）

## 6. 输入输出

- **输入**：z (batch, time, 96) — 潜变量
- **输出**：mel (batch, 80, time) — 预测的梅尔频谱（dB 刻度）

## 7. 在 VITS 中的位置

```
TextEncoder → DurationPredictor → regulate_length
                                       ↓
                                  proj_mu(192→96) → z = mu (推理)
                                       ↓
                                  Decoder(Conv1d ResBlocks)
                                       ↓
                                  mel (80, T)
                                       ↓
                              Griffin-Lim / HiFi-GAN
                                       ↓
                                   waveform
```
