# VAE 解码器原理文档

## 1. 概述

VAE 解码器是 VITS 模型中负责将编码后的表示转换为梅尔频谱的组件。在 VITS 中，编码器输出的高维表示通过一个全连接网络解码为梅尔频谱参数。

## 2. 架构

### 2.1 组件结构

```
输入 (batch, time, 96)
    ↓
[Linear(96→96) + ReLU] × 6层
    ↓
Linear(96→80)  # 输出投影到梅尔频谱维度
    ↓
转置 (batch, 80, time)
    ↓
梅尔频谱 (batch, n_mels, time)
```

### 2.2 前馈网络

6 层前馈网络，每层包含：
- Linear：保持维度不变 (96 → 96)
- ReLU：非线性激活

这种结构提供了足够的容量来学习从文本表示到音频表示的复杂映射。

## 3. VAE 重参数化采样

### 3.1 为什么要重参数化？

VAE 的损失函数包含一个 KL 散度项，要求我们从先验分布中采样。在反向传播时，采样操作是不可导的。重参数化技巧将随机性转移到另一个变量上：

```
z = μ + σ * ε,  其中 ε ~ N(0, 1)
σ = exp(0.5 * log_var)
```

这样 μ 和 log_var 可以通过梯度下降学习，而 ε 的随机性可以正常反向传播。

### 3.2 实现

```python
def reparameterize(self, mu, log_var):
    eps = torch.randn_like(mu)
    z = mu + torch.exp(0.5 * log_var) * eps
    return z
```

## 4. 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| in_channels | 96 | 输入通道数 |
| out_channels | 80 | 输出梅尔频谱维度 |
| n_layers | 6 | 前馈层层数 |

## 5. 输入输出

- **输入**：
  - `z`: (batch, time, 96) 编码器输出或 VAE 采样结果

- **输出**：
  - `mel`: (batch, 80, time) 梅尔频谱

## 6. 在 VITS 中的作用

1. **训练阶段**：接收展长后的编码器输出，解码为梅尔频谱预测
2. **推理阶段**：同样解码展长后的表示为梅尔频谱

## 7. 与其他组件的关系

```
TextEncoder ──┬──> DurationPredictor ──> regulate_length
               │
               └──> [mu, log_var] ──> reparameterize ──> Decoder ──> mel
```

VITS 的一个关键创新是在 VAE 框架内联合学习文本编码和音频生成。
