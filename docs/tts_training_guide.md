# TTS 模型训练完整指南

## 一、从零训练一个 TTS 模型

### 第一步：理解 TTS 系统的核心管线

一个端到端 TTS 系统由以下模块组成，每个模块解决一个具体问题：

```
"Hello world"
     │
     ▼
┌─────────────┐   问题：计算机不认识文字，只认识数字
│  文本前端     │   技术：音素化 (Phonemizer)
│  Text→音素ID  │   把 "hello" → ["HH","AH","L","OW"] → [33,3,38,12]
└──────┬──────┘
       ▼
┌─────────────┐   问题：需要理解音素之间的上下文关系
│  文本编码器   │   技术：Transformer 自注意力
│  TextEncoder │   每个音素的向量都"看到"了所有其他音素
└──────┬──────┘
       ▼
┌─────────────┐   问题：4 个音素要变成 200 帧 mel，长度不匹配
│  时长预测器   │   技术：Duration Predictor + regulate_length
│  Duration    │   预测每个音素占多少帧，然后重复展开
└──────┬──────┘
       ▼
┌─────────────┐   问题：从文本表示生成声学特征（mel 频谱）
│  解码器      │   技术：Conv1d 残差块（捕获时间依赖）
│  Decoder     │   + VAE 框架（学习平滑的潜空间）
└──────┬──────┘
       ▼
┌─────────────┐   问题：mel 频谱不是音频，需要转成波形
│  声码器      │   技术：Griffin-Lim（信号处理）或 HiFi-GAN（神经网络）
│  Vocoder     │   从 80 维 mel → 22050 个采样点/秒的波形
└──────┬──────┘
       ▼
   🔊 音频
```

### 第二步：需要掌握的技术点

#### A. 信号处理基础（必须掌握）

| 概念 | 为什么需要 | 关键公式/参数 |
|------|-----------|--------------|
| **采样率** | 音频的时间分辨率 | 22050 Hz = 每秒 22050 个数字 |
| **STFT** | 把时域信号变成时频表示 | 窗口 n_fft=1024, 帧移 hop=256 |
| **梅尔刻度** | 模拟人耳对频率的非线性感知 | mel = 2595 × log₁₀(1 + f/700) |
| **梅尔频谱** | TTS 模型的目标输出 | (80 mel bins, T 帧) |
| **dB 变换** | 压缩动态范围便于网络学习 | dB = 10 × log₁₀(power) |
| **Griffin-Lim** | 从幅度谱恢复相位重建音频 | 迭代交替投影算法 |

理解路径：wav 波形 ←→ STFT 频谱 ←→ mel 频谱。训练时 wav→mel 提取特征，推理时 mel→wav 重建音频。

#### B. 深度学习基础（必须掌握）

| 概念 | 在 TTS 中的用途 |
|------|----------------|
| **Embedding** | 把离散音素 ID 映射为连续向量 |
| **Transformer / 自注意力** | 文本编码器，捕获全局上下文 |
| **1D 卷积 (Conv1d)** | 解码器和时长预测器，捕获局部时间模式 |
| **膨胀卷积 (Dilated Conv)** | 用小 kernel 获得大感受野 |
| **残差连接** | 防止梯度消失，让深层网络可训练 |
| **LayerNorm** | 稳定训练，不依赖 batch 统计量 |
| **VAE 重参数化** | 让采样操作可微分，训练时加噪推理时不加 |

#### C. 训练技巧（进阶）

| 技巧 | 原因 |
|------|------|
| **AdamW 优化器** | 比 Adam 有更正确的权重衰减 |
| **Cosine Annealing 学习率** | 前期快速收敛，后期精细调优 |
| **梯度裁剪 (max_norm=1.0)** | 防止梯度爆炸导致训练崩溃 |
| **Teacher Forcing 比例** | 90% 用真实 duration，10% 用预测值，平衡质量和泛化 |
| **多损失加权** | mel_loss + 5×dur_loss + 0.1×kl_loss，不同项量级差异大需要手动平衡 |

#### D. 音素化（文本前端）

| 语言 | 方案 | 说明 |
|------|------|------|
| 英文 | gruut / espeak-ng | 文本→ARPAbet 音素（"hello"→"HH AH L OW"） |
| 中文 | pypinyin | 文本→拼音（"你好"→"ni3 hao3"） |

需要建立**音素符号表**（PHONEME_LIST）和 ID 映射（SYMBOL_TO_ID），注意不能有重复符号。

### 第三步：准备数据

#### 数据集选择

| 数据集 | 语言 | 时长 | 说话人 | 特点 |
|--------|------|------|--------|------|
| **LJSpeech** | 英文 | 24h | 单人女声 | 入门首选，干净 |
| LibriTTS | 英文 | 585h | 多人 | 大规模 |
| AISHELL-3 | 中文 | 85h | 多人 | 中文首选 |
| VCTK | 英文 | 44h | 109人 | 多说话人 |

#### 数据预处理

```
原始数据:  wav (22050Hz) + 文本
    ↓
预处理:
  1. wav → mel 频谱 (.npy)    torchaudio MelSpectrogram + AmplitudeToDB
  2. text → 音素 ID (.npy)     phonemizer + symbol_to_id 查表
  3. 生成 filelist             train.txt (95%) + val.txt (5%)
    ↓
训练数据:  mels/*.npy + phonemes/*.npy + filelists/
```

关键参数要统一：sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80。

### 第四步：搭建模型

从简单到复杂的推荐路径：

```
Level 1: Tacotron 2 (自回归，简单但推理慢)
Level 2: FastSpeech 2 (非自回归，需要外部对齐工具)
Level 3: VITS (本项目，端到端 VAE + 非自回归)
Level 4: 原始 VITS (带 Normalizing Flow + GAN 判别器)
```

每个模块的关键设计决策：

| 模块 | 决策 | 本项目选择 | 原因 |
|------|------|-----------|------|
| 编码器 | RNN vs Transformer | Transformer (2层) | 并行计算，全局注意力 |
| 解码器 | MLP vs Conv1d vs WaveNet | Conv1d 残差块 (6层) | 时间建模能力 + 参数效率 |
| 时长 | 外部对齐 vs 隐式学习 | 隐式学习（加权分配） | 不需要 MFA 等外部工具 |
| 声码器 | Griffin-Lim vs HiFi-GAN | Griffin-Lim (过渡) | 无需单独训练 |

### 第五步：训练

#### 损失函数设计

```
总损失 = mel_loss + α × duration_loss + β × kl_loss
         ────────   ──────────────────   ────────────
         L1 重建     对数域 MSE            VAE 正则化
         (主要)      (α=5.0, 对齐质量)     (β=0.1, 防止坍塌)
```

三个损失的作用：
- **mel_loss**: 让模型输出的 mel 接近真实 mel（核心目标）
- **duration_loss**: 让时长预测准确（对齐质量）
- **kl_loss**: 让潜空间平滑（泛化能力，但权重不能太大否则后验坍塌）

#### 训练规模参考

| 配置 | 数据量 | Epochs | 设备 | 时间 | 效果 |
|------|--------|--------|------|------|------|
| 快速验证 | 1000 条 | 50 | CPU | ~5h | 能出声但不像语音 |
| 基础训练 | 13000 条 | 200 | GPU (V100) | ~10h | 可懂但音质一般 |
| 充分训练 | 13000 条 | 500+ | GPU (V100) | ~25h | 音质较好 |
| 生产级 | 50000+ 条 | 1000+ | 多 GPU | 数天 | 接近自然语音 |

#### 训练过程监控

重点观察的指标：
- **mel_loss 下降曲线**：应持续下降。若平台化，考虑降低学习率
- **val_loss vs train_loss 差距**：差距大说明过拟合，增加数据或加正则
- **duration_pred vs target**：均值应接近，否则对齐有问题
- **定期合成样本**：每 10 个 epoch 合成一段听听，比看数字更直观

### 第六步：声码器

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Griffin-Lim** | 不需要训练，即插即用 | 音质有限，机器人味 | 快速验证、开发调试 |
| **预训练 HiFi-GAN** | 音质好，不需要自己训 | 可能与你的 mel 不匹配 | 使用标准参数时 |
| **自训练 HiFi-GAN** | 音质最好，完全匹配 | 需要额外训练时间 | 生产部署 |

### 第七步：推理优化

```python
# 必须做的
model.eval()                     # 关闭 dropout
torch.no_grad()                  # 不计算梯度
z = mu                           # 推理时不加 VAE 噪声

# 可选优化
torch.compile(model)             # PyTorch 2.0+ 编译加速
model.half()                     # FP16 半精度推理
mel 分布匹配                      # 模型欠训练时的后处理补救
int16 WAV 输出                   # 浏览器兼容
```

---

## 二、后训练（Fine-tuning）

后训练是在**已有模型权重基础上**继续训练，而不是从零开始。

### 什么时候需要后训练？

| 场景 | 说明 | 数据需求 |
|------|------|----------|
| **换说话人** | 让模型学新的音色 | 新说话人 1-5 小时录音 |
| **换语言** | 英文模型适配中文 | 目标语言数据 + 新音素表 |
| **提升音质** | 在更多数据上继续训 | 更多同分布数据 |
| **领域适配** | 适配特定场景（新闻播报、有声书） | 领域内数据 |

### 后训练 vs 从零训练的区别

| | 从零训练 | 后训练 |
|--|---------|--------|
| 权重初始化 | 随机 | 加载预训练 checkpoint |
| 学习率 | 较大 (2e-4) | **较小 (1e-5 ~ 5e-5)** |
| Epochs | 200-1000 | **20-100 通常够** |
| 数据量 | 越多越好 (10h+) | **可以很少 (30min-5h)** |
| 冻结层 | 无 | **可选冻结编码器** |
| 风险 | 训练时间长 | **灾难性遗忘** |

### 后训练具体步骤

#### 1. 准备新数据

和从零训练一样的预处理流程，但要注意：

```
关键：音频参数必须与预训练模型一致！
  - sample_rate = 22050  (不能用 16000 或 44100)
  - n_fft = 1024
  - hop_length = 256
  - n_mels = 80
```

如果参数不一致，mel 频谱的分布会完全不同，模型会"听不懂"新数据。

#### 2. 加载预训练权重

```python
# 加载 checkpoint
checkpoint = torch.load('pretrained_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# 可选：冻结部分层
for param in model.text_encoder.parameters():
    param.requires_grad = False  # 冻结编码器，只训练解码器
```

#### 3. 调整训练参数

```python
# 后训练用更小的学习率，避免破坏已学到的知识
optimizer = AdamW(model.parameters(), lr=1e-5)    # 比从零训练小 20 倍

# 更少的 epochs
epochs = 50  # 而不是 200+

# 可选：更大的 KL 权重防止潜空间偏移
kl_weight = 0.5  # 而不是 0.1
```

#### 4. 处理灾难性遗忘

后训练的最大风险是**灾难性遗忘**——模型学了新东西但忘了旧能力。

对策：

| 方法 | 做法 | 效果 |
|------|------|------|
| **混合数据** | 新数据 + 10-20% 旧数据混合训练 | 最有效 |
| **低学习率** | 1e-5 而非 2e-4 | 简单有效 |
| **冻结底层** | 只训练 Decoder，冻结 Encoder | 保留语言理解能力 |
| **EWC 正则化** | 惩罚重要参数的变化 | 理论优雅但实现复杂 |
| **LoRA** | 冻结原始权重，只训练低秩增量 | 参数高效，推荐 |

#### 5. 换说话人的特殊处理

如果是多说话人模型，需要：
- 添加 **Speaker Embedding** 层
- 每个说话人一个 ID，embedding 注入到 Decoder
- 新说话人只需训练新的 embedding + 微调 Decoder

如果是单说话人模型换人：
- 直接用新数据后训练整个模型
- 学习率要小，epochs 不用太多 (30-50)

#### 6. 换语言的特殊处理

```
需要改动的：
  ✅ 音素符号表 (symbols.py) — 添加新语言的音素
  ✅ 音素化器 (phonemizer.py) — 添加新语言的文本→音素转换
  ✅ Embedding 层 — vocab_size 可能需要扩大
  ✅ 训练数据 — 目标语言的 (文本, 音频) 对

不需要改的：
  ❌ Decoder 架构
  ❌ 声码器
  ❌ 音频参数 (mel 配置)
```

扩展 Embedding 层的做法：
```python
# 旧 embedding: vocab_size=256
# 新 embedding: vocab_size=512 (加入中文音素)
new_embedding = nn.Embedding(512, hidden_dim)
new_embedding.weight.data[:256] = old_embedding.weight.data  # 复用旧权重
model.text_encoder.embedding = new_embedding
```

---

## 三、常见坑和排查方法

| 现象 | 可能原因 | 排查方法 |
|------|---------|---------|
| 输出滋啦滋啦 | 推理时 VAE 加了随机噪声 | 检查 infer() 是否用 z=mu |
| 输出嗡嗡嗡 | mel→wav 转换错误 或 WAV 格式不对 | 用真实 mel 跑 Griffin-Lim 验证 vocoder |
| 有声音但听不清 | 模型欠训练 | 增加数据量和 epochs |
| loss 不下降 | 学习率太大/太小，或数据有问题 | 先用小数据跑几步确认管线正确 |
| loss 下降但合成是噪声 | Decoder 架构太弱（如纯 MLP） | 用 Conv1d 残差块替代 |
| 浏览器无法播放 | WAV 是 float32 格式 | 转成 int16 PCM |
| 音量很低 | 模型输出 mel 值域偏窄 | 做分布匹配后处理 |
| 对齐错乱 | Duration 计算有问题 | 检查 duration 总和是否等于 mel 帧数 |
| 音素 ID 错乱 | 符号表有重复 | 检查 PHONEME_LIST 无重复项 |

---

## 四、技术栈总结

```
必须掌握                              建议了解
──────────                           ──────────
PyTorch 基础                          Normalizing Flow
STFT / Mel 频谱                       GAN 对抗训练
Transformer 注意力                    WaveNet / WaveRNN
Conv1d / 膨胀卷积                      MFA (Montreal Forced Aligner)
VAE 重参数化                           混合精度训练 (AMP)
损失函数设计                            分布式训练 (DDP)
Griffin-Lim 算法                      ONNX 导出与推理优化
音素化 (g2p)                          流式推理 (streaming)
```

## 五、推荐学习路线

```
Week 1:  信号处理基础 → 理解 wav/STFT/mel 之间的转换
Week 2:  跑通本项目 → 理解完整 TTS 管线
Week 3:  读 VITS 原始论文 → 理解 Normalizing Flow + GAN
Week 4:  用 LJSpeech 全量训练 → 调参实战
Week 5:  训练 HiFi-GAN → 体验声码器对音质的影响
Week 6:  尝试后训练 → 换说话人/换语言
```
