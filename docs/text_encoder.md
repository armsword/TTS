# 文本编码器原理文档

## 1. 概述

文本编码器（Text Encoder）是 VITS 模型的第一个主要组件，负责将音素序列转换为连续的隐藏表示。这是 VITS 区别于传统 Tacotron 的关键改进之一——使用自注意力机制替代 RNN/LSTM 来处理序列。

## 2. 架构

### 2.1 组件

```
音素 IDs → 嵌入层 → 位置编码 → Transformer 编码器 → 隐藏表示
```

1. **音素嵌入层 (Embedding)**
   - 将离散的音素 ID 转换为密集向量
   - 维度：`vocab_size × hidden_dim`
   - 使用 `padding_idx=0` 使得 PAD 符号的嵌入向量为全零

2. **位置编码 (Positional Encoding)**
   - 使用正弦/余弦函数生成固定位置编码
   - 公式：
     ```
     PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
     PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
     ```
   - 与嵌入向量相加，为模型提供序列位置信息

3. **Transformer 编码器层**
   - 使用 2 层 TransformerEncoderLayer
   - 配置：
     - `nhead=4`：4 头注意力
     - `dim_feedforward=hidden_dim * 4`：前馈网络维度
     - `dropout=0.1`：防止过拟合

### 2.2 自注意力机制

每个注意力头的计算：
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

多头注意力将输入分成多个头，分别计算注意力，然后拼接：
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
```

## 3. Mask 生成

### 3.1 Key Padding Mask

在 Transformer 中，我们需要告诉模型哪些位置是填充的（不应该参与注意力计算）：

```python
key_padding_mask[i, length:] = True  # 填充位置为 True
```

### 3.2 输出 Mask

为后续模块生成 mask，表示有效位置（1）和填充位置（0）：

```python
mask[i, :, length:] = 0  # 填充位置为 0
```

## 4. 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| vocab_size | 256 | 音素词汇表大小 |
| hidden_dim | 192 | 隐藏层维度 |
| n_layers | 2 | Transformer 层数 |
| n_heads | 4 | 注意力头数 |

## 5. 输入输出

- **输入**：
  - `x`: (batch, time) 音素 ID 序列
  - `x_lengths`: (batch,) 每个样本的实际长度

- **输出**：
  - `output`: (batch, time, hidden_dim) 编码后的隐藏表示
  - `mask`: (batch, 1, time) 有效位置为 1，填充为 0

## 6. 在 VITS 中的作用

文本编码器的输出用于：
1. **时长预测器**：预测每个音素的持续时间
2. **VAE 编码器**：与先验分布结合生成潜变量

## 7. 参数量估算

- 嵌入层：256 × 192 ≈ 49K
- Transformer 层（2层）：约 1-2M
- 总计：< 5M 符合设计要求
