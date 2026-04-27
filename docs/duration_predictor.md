# 时长预测器原理文档

## 1. 概述

时长预测器（Duration Predictor）解决的核心问题是：**文本序列和音频序列长度不一致**。一个音素 "AH" 可能对应 3 帧 mel，而 "IY" 可能对应 8 帧。时长预测器为每个音素预测其在梅尔频谱中占据的帧数，然后通过 regulate_length 将编码器输出展开到正确长度。

这是非自回归 TTS（如 FastSpeech、VITS）的关键组件，避免了 Tacotron 式自回归解码的慢速推理。

## 2. 架构

```
编码器输出 (batch, time, 192)
    ↓ 转置
(batch, 192, time)
    ↓
[Conv1d(192, 192, kernel=3, pad=1) + ReLU] × 4 层
    ↓
LayerNorm
    ↓ 转置
(batch, time, 192)
    ↓
Linear(192 → 1)
    ↓
exp() + clamp(min=1.0)
    ↓
时长预测 (batch, time)  每个值 ≥ 1.0
```

### 2.1 为什么用 4 层 Conv1d？

时长预测需要上下文信息——一个音素的发音时长取决于：
- 前后音素（协同发音效应）
- 在词中的位置
- 句子韵律

4 层 Conv1d(kernel=3) 提供 9 个音素的感受野，足以捕获局部上下文。

### 2.2 exp() 输出激活

时长必须为正数。使用 exp() 将网络输出从 (-∞, +∞) 映射到 (0, +∞)，再 clamp(min=1.0) 确保最短时长为 1 帧。

## 3. regulate_length — 时长展开

### 3.1 原理

根据预测的时长，将每个音素的编码向量重复对应次数：

```
输入:  encoder_out = [[v0, v1, v2]]     shape: (1, 3, 192)
       durations   = [2, 3, 1]

输出:  [[v0, v0, v1, v1, v1, v2]]       shape: (1, 6, 192)
```

### 3.2 实现方式

```python
for each phoneme i:
    expanded[current_pos : current_pos + dur[i]] = encoder_out[i]
    current_pos += dur[i]
```

### 3.3 Batch 处理

同一 batch 内不同样本的展开长度可能不同，需要 padding 到最长：

```
样本 1: durations=[2,3,1] → 展开长度 6
样本 2: durations=[4,2]   → 展开长度 6
→ max_len = 6, 样本 2 无需 padding
```

## 4. Duration 的获取方式

### 4.1 训练时：基于音素类型的加权分配

由于 LJSpeech 没有逐音素的对齐标注，我们根据音素类型和 mel 帧总数计算伪 duration：

```python
weights = {
    'vowel': 1.2,      # 元音较长（"AA", "IY" 等）
    'stop': 0.7,       # 爆破音较短（"P", "B", "T" 等）
    'fricative': 0.9,  # 擦音中等（"S", "F" 等）
    'nasal': 0.8,      # 鼻音中等（"M", "N" 等）
    'punct': 0.3,      # 标点很短
}
durations = normalize(weights) * total_frames
```

训练时还添加少量高斯噪声（std=1.5），让模型学会处理不确定性。

### 4.2 推理时：模型预测

使用训练好的时长预测器直接输出每个音素的帧数。

## 5. 损失函数

时长预测使用**对数域 MSE 损失**（对长短音素同等关注）：

```python
loss = MSE(log(pred_duration), log(target_duration))
```

只在有效音素位置计算（通过 mask 排除 padding）。

## 6. 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| in_channels | 192 | 输入通道数（= TextEncoder hidden_dim） |
| n_layers | 4 | Conv1d 层数（从 v1 的 2 层增加到 4 层） |

**参数量**：约 444K

## 7. 与传统方法对比

| 方法 | 时长来源 | 解码方式 | 推理速度 |
|------|----------|----------|----------|
| Tacotron 2 | 无（注意力对齐） | 自回归 | 慢 |
| FastSpeech | 外部对齐工具 | 非自回归 | 快 |
| VITS (本项目) | 隐式学习 | 非自回归 | 快 |
