# 修复 TTS 输出"滋啦滋啦"噪声问题 - 原理文档

## 问题现象

合成的音频只有滋啦滋啦的静电噪声，没有任何有效语音内容。

## 根因分析

经过完整的推理链路排查，发现 **5 个关键 bug** 协同导致了噪声输出：

---

### Bug 1（P0）：推理时 VAE 注入随机噪声

**文件**：`src/model/vits.py` (infer 方法)

**问题**：推理时代码设置 `log_var = 0`，然后调用 `reparameterize(mu, log_var)`：

```python
z = mu + exp(0.5 * log_var) * eps    # eps = randn_like(mu)
```

当 `log_var = 0` 时，`exp(0) = 1`，所以 `z = mu + 1.0 * 随机噪声`。这意味着推理时给每个潜变量都加了**单位方差的高斯噪声**，完全淹没了有用信号。

**原理**：VAE 的重参数化技巧（reparameterization trick）是**训练时**用的，目的是让采样操作可微分以支持反向传播。但在**推理时**，我们要的是确定性输出，应该直接用均值 `mu` 作为潜变量，跳过随机采样。

**修复**：推理时 `z = mu`，不调用 reparameterize。

---

### Bug 2（P0）：Griffin-Lim 回退实现完全错误

**文件**：`src/infer.py`（无 librosa 时的回退路径）

**问题**：原来的"简化 Griffin-Lim"实现：

```python
waveform = np.zeros(mel_db.shape[1] * hop_length)
for i in range(mel_db.shape[1]):
    waveform[i * hop_length] = mel_db[0, i]
```

这根本不是 Griffin-Lim！它只是在一个全零数组的每 256 个采样点放一个梅尔频谱值。结果是**稀疏脉冲序列**，听起来就是"滋啦滋啦"的点击噪声。

**原理**：真正的 Griffin-Lim 算法是一种**迭代相位重建算法**，流程如下：

1. **Mel → 线性频谱**：通过梅尔滤波器组的伪逆矩阵，将 80 维梅尔频谱转回 513 维线性幅度谱
2. **随机初始化相位**：由于梅尔频谱丢失了相位信息，需要从随机相位开始
3. **迭代优化**（60 次）：
   - 用当前幅度 + 相位做 ISTFT 得到时域波形
   - 对波形重新做 STFT 得到新的复数频谱
   - 保留新频谱的**相位**，替换**幅度**为原始幅度
   - 重复直到相位收敛
4. **最终 ISTFT**：用收敛后的相位 + 原始幅度做最终逆变换

每次迭代会让重建相位越来越接近"与给定幅度谱一致"的最优相位。

**修复**：实现完整的 Griffin-Lim 算法，包括梅尔滤波器组构造、伪逆变换和迭代相位重建。

---

### Bug 3（P0）：Decoder 架构过于简单

**文件**：`src/model/decoder.py`

**问题**：原来的 Decoder 是 6 层 `Linear(96, 96) + ReLU` 的纯 MLP。MLP 独立处理每个时间步，**完全没有时间维度上的信息交互**。梅尔频谱是时序信号，相邻帧之间有强相关性，纯 MLP 无法捕获这种模式。

**原理**：语音的梅尔频谱在时间轴上有很强的连续性——共振峰（formant）是平滑变化的，音素之间有协同发音（coarticulation）效应。需要用**卷积网络**来建模这种时间依赖：

- **1D 卷积**：每个输出帧依赖于周围多个输入帧（感受野 = kernel_size）
- **膨胀卷积**（dilated convolution）：通过膨胀率 [1, 2, 4]，在不增加参数的情况下指数级扩大感受野
- **残差连接**：避免梯度消失，让深层网络更容易训练
- **LayerNorm**：稳定训练过程
- **GELU 激活**：比 ReLU 更平滑，在 NLP/语音任务中表现更好

**修复**：将 Decoder 改为 `Linear投影 → 6层Conv1d残差块（膨胀率循环 1,2,4） → Conv1d输出投影` 架构。

---

### Bug 4（P1）：音素符号表重复

**文件**：`src/text/symbols.py`

**问题**：`PHONEME_LIST` 中 `"B"` 出现了两次（一次在"停顿符"区、一次在"辅音 Stops"区），`"DD"` 和 `"E"` 作为"停顿符"也是多余的。由于 `SYMBOL_TO_ID` 用字典推导构建，后出现的重复项会覆盖前面的 ID 映射，导致部分音素的 ID 对应错误。

**原理**：文本编码器的 Embedding 层根据音素 ID 查找向量。如果 ID 映射错误，模型就相当于在"说错的音素"上训练和推理，输出自然是乱码。

**修复**：删除重复的"停顿符"区域，保留正确的 ARPAbet 音素集。

---

### Bug 5（背景问题）：训练数据和训练量不足

- **训练数据是 440Hz 正弦波**（`generate_mock_data.py`），不是真实语音
- **只训练了 10 个 epoch**，损失从 10.67 降到 9.58，远未收敛
- **HiFi-GAN 声码器未训练**，无 checkpoint 文件

这些是更深层的问题。即使修复了上述 4 个代码 bug，也需要用**真实语音数据**重新训练足够多的 epoch 才能得到有效语音输出。

---

## 修复后的推理流程

```
Text
  ↓ text_to_phonemes() + text_to_sequence()
Phoneme IDs (正确的 ID 映射，无重复)
  ↓ TextEncoder
Encoder Output (batch, time, 192)
  ↓ DurationPredictor + regulate_length
Expanded Output (batch, expanded_time, 192)
  ↓ proj_mu (Linear 192→96)
mu (batch, expanded_time, 96)
  ↓ z = mu (推理时不加噪声)
z (batch, expanded_time, 96)
  ↓ Decoder (Conv1d 残差块架构)
Mel Spectrogram (batch, 80, expanded_time)
  ↓ Griffin-Lim (迭代相位重建) 或 HiFi-GAN
Waveform
```

## 下一步

1. 准备真实语音数据集（LJSpeech 等）
2. 用新架构重新训练模型（建议 200+ epochs）
3. 训练 HiFi-GAN 声码器（或使用预训练权重）替代 Griffin-Lim
