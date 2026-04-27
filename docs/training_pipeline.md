# 训练流程原理文档

## 1. 概述

本文档描述 VITS 模型的完整训练流程，包括数据预处理、模型训练、损失函数设计和推理。

## 2. 数据预处理

### 2.1 LJSpeech 数据集

| 属性 | 值 |
|------|-----|
| 样本数 | 13,100 句英文 |
| 说话人 | 单人女声 |
| 采样率 | 22,050 Hz |
| 总时长 | ~24 小时 |
| 格式 | `metadata.csv`（filename\|text）+ `wavs/` 目录 |

### 2.2 预处理流程

```
data/raw/LJSpeech-1.1/
├── metadata.csv
└── wavs/*.wav
        ↓  python src/preprocess.py
data/processed/
├── mels/          # 每个样本的 mel.npy (80, T)
├── phonemes/      # 每个样本的 phoneme_ids.npy (N,)
└── filelists/
    ├── train.txt  # 12,445 条 (95%)
    └── val.txt    #    655 条 (5%)
```

每条样本的处理：
1. **音频 → Mel**：`torchaudio.MelSpectrogram(n_fft=1024, hop=256, n_mels=80)` → `AmplitudeToDB(top_db=80)`
2. **文本 → 音素 ID**：`text_to_phonemes()` (gruut/fallback) → `SYMBOL_TO_ID` 查表
3. **保存**：mel 和 phoneme_ids 分别存为 `.npy`

### 2.3 Mel 频谱统计

| 指标 | 值 |
|------|-----|
| 均值 | -14.96 dB |
| 标准差 | 17.23 dB |
| 范围 | [-49.5, 43.4] dB |

## 3. 数据集与加载

### 3.1 TTSDataset

每个样本返回：
- `phoneme_ids`: 音素 ID 序列 (N,)
- `mel`: 梅尔频谱 (80, T)
- `duration`: 基于音素类型加权分配的伪 duration (N,)

### 3.2 collate_fn

将变长样本 padding 到 batch 内最大长度：
- phoneme_ids → (B, max_N)
- mel → (B, 80, max_T)
- duration → (B, max_N)
- phoneme_lengths → (B,) 记录真实长度

### 3.3 Duration 计算

LJSpeech 没有逐音素对齐标注，因此用**音素类型加权**近似：

```python
# 根据音素类型分配权重
weights = {'vowel': 1.2, 'stop': 0.7, 'fricative': 0.9, ...}
# 归一化后按比例分配 mel 帧数
durations = normalize(weights) * total_frames
# 训练时加噪声增加鲁棒性
durations += N(0, 1.5)
```

## 4. 模型前向传播（训练模式）

```
phoneme_ids (B, N)
    ↓ TextEncoder
encoder_output (B, N, 192) + mask (B, 1, N)
    ↓ DurationPredictor
duration_pred (B, N)
    ↓ regulate_length (用 GT duration, 90% teacher forcing)
expanded_output (B, T, 192)
    ↓ proj_mu, proj_log_var
mu (B, T, 96), log_var (B, T, 96)
    ↓ reparameterize: z = mu + exp(0.5*log_var) * eps
z (B, T, 96)
    ↓ Decoder (Conv1d ResBlocks)
mel_output (B, 80, T)
    ↓ pad/trim to match mel_target length
mel_output (B, 80, T_target)
```

## 5. 损失函数

总损失 = mel_loss + 5.0 × duration_loss + 0.1 × kl_loss

### 5.1 Mel 重建损失 (L1)

```python
mel_loss = |mel_output - mel_target|₁
```

直接度量预测 mel 与真实 mel 的逐像素差异。L1 比 L2 更鲁棒，不会过度惩罚大误差。

### 5.2 Duration 损失 (对数域 MSE)

```python
duration_loss = MSE(log(pred), log(target))  # 只在有效音素位置
```

在对数域计算，使长短音素的误差权重更均衡。使用 mask 排除 padding 位置。

### 5.3 KL 散度损失

```python
kl_loss = -0.5 * mean(1 + log_var - mu² - exp(log_var))
```

约束潜变量分布 q(z|x) 接近标准正态 N(0,1)，防止潜空间坍塌。权重 0.1 避免 KL 过强导致后验坍塌（posterior collapse）。

## 6. 训练配置

### 6.1 优化器

| 参数 | 值 | 说明 |
|------|-----|------|
| 优化器 | AdamW | 比 Adam 有更好的权重衰减正则化 |
| 学习率 | 2e-4 | 初始学习率 |
| 权重衰减 | 0.01 | L2 正则化 |
| 梯度裁剪 | max_norm=1.0 | 防止梯度爆炸 |
| 学习率调度 | CosineAnnealing | T_max=epochs, eta_min=1e-5 |

### 6.2 训练策略

- **Teacher forcing**: 90% 的 batch 使用 GT duration，10% 使用预测 duration
- **Checkpoint**: 每 10 个 epoch 保存
- **Best model**: 验证 loss 最低时保存 `best_model.pt`

### 6.3 子集训练结果 (1000 样本, 50 epochs)

| Epoch | Train Loss | Val Loss | Mel Loss | 耗时/epoch |
|-------|-----------|----------|----------|-----------|
| 1 | 11.66 | 10.98 | 9.73 | ~6 min |
| 10 | 9.57 | 10.08 | 9.51 | ~6 min |
| 25 | 9.41 | 10.01 | 9.43 | ~6 min |
| 44 | 9.38 | **9.92** | **9.34** | ~6 min |

总训练时间约 5 小时（CPU），Val Loss 下降 9.6%。

## 7. 推理流程

```
text
  ↓ text_to_sequence()
phoneme_ids (1, N)
  ↓ TextEncoder
encoder_output (1, N, 192)
  ↓ DurationPredictor (无 teacher forcing)
duration_pred → regulate_length
  ↓
expanded (1, T, 192)
  ↓ proj_mu → z = mu (不加噪声)
z (1, T, 96)
  ↓ Decoder
mel (1, 80, T)
  ↓ _normalize_mel (分布匹配到训练数据统计量)
  ↓ dB→power: 10^(mel/10)
  ↓ librosa Griffin-Lim (64 次迭代) 或 HiFi-GAN
  ↓ to_int16_wav (归一化 + float32→int16)
waveform → WAV file
```

### 7.1 Mel 分布匹配

模型欠训练时输出的 mel std (~10) 远低于训练数据 (~17)，导致 Griffin-Lim 输出能量不足。推理时将模型 mel 的统计量线性映射到训练数据分布：

```python
mel_scaled = (mel - model_mean) / model_std * train_std + train_mean
```

### 7.2 Griffin-Lim 相位重建

mel 频谱只有幅度信息，丢失了相位。Griffin-Lim 通过迭代重建近似相位：

1. mel → 线性频谱（mel 滤波器组伪逆）
2. 随机初始化相位
3. 迭代 64 次：ISTFT → STFT → 保留新相位、替换幅度
4. 最终 ISTFT 输出波形

## 8. 运行命令

```bash
# 预处理
python src/preprocess.py --data_dir data/raw/LJSpeech-1.1 --output_dir data/processed

# 训练（子集快速验证）
python run_train.py

# 推理
python -c "
import sys; sys.path.insert(0, 'src')
from infer import TTSInferencer
inf = TTSInferencer(checkpoint_path='data/checkpoints_v2/best_model.pt')
inf.synthesize_to_file('Hello world', 'output.wav', 'en')
"

# Web 服务
python server.py  # 访问 http://localhost:5000
```
