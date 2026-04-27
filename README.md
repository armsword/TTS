# VITS TTS - 端到端语音合成系统

基于 VITS (Conditional Variational Autoencoder with Adversarial Learning) 架构的文本到语音合成系统，支持英文语音合成，提供 Web UI 和 REST API。

## 系统架构

```
Text ──→ Phonemizer ──→ TextEncoder ──→ DurationPredictor ──→ regulate_length
                         (Transformer)   (4-layer Conv1d)      (展开到mel长度)
                                                                     │
                                                              proj_mu (192→96)
                                                                     │
                                                              Decoder (Conv1d ResBlocks)
                                                                     │
                                                              Mel Spectrogram (80, T)
                                                                     │
                                                         ┌───────────┴───────────┐
                                                    Griffin-Lim            HiFi-GAN
                                                    (信号处理)             (神经声码器)
                                                         │                     │
                                                    Waveform (22050 Hz, int16 PCM)
```

## 模型参数

| 模块 | 参数量 | 说明 |
|------|--------|------|
| TextEncoder | 939K | Embedding + 2 层 Transformer |
| DurationPredictor | 444K | 4 层 Conv1d |
| Decoder | 8.9M | 6 层膨胀 Conv1d 残差块 |
| proj_mu / proj_log_var | 37K | 线性投影 |
| **总计** | **10.35M** | |

## 快速开始

### 环境安装

```bash
pip install torch torchaudio numpy scipy librosa flask gruut pypinyin
```

### 推理（命令行）

```python
import sys; sys.path.insert(0, 'src')
from infer import TTSInferencer

inferencer = TTSInferencer(checkpoint_path='data/checkpoints_v2/best_model.pt')
inferencer.synthesize_to_file('Hello world', 'output.wav', 'en')
```

### Web 服务

```bash
python server.py
# 浏览器访问 http://localhost:5000
```

### REST API

```bash
# 健康检查
curl http://localhost:5000/api/health

# 合成语音
curl -X POST http://localhost:5000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "language": "en"}' \
  -o output.wav
```

## 训练

### 1. 数据预处理

```bash
python src/preprocess.py \
  --data_dir data/raw/LJSpeech-1.1 \
  --output_dir data/processed
```

### 2. 训练模型

```bash
# 子集快速验证 (1000 条, 50 epochs, ~5h CPU)
python run_train.py

# 全量训练 (需要 GPU)
# 修改 run_train.py 中 filelist 路径为 train.txt/val.txt
```

### 3. 训练指标 (子集 1000 条, 50 epochs)

| Epoch | Train Loss | Val Loss | Mel Loss |
|-------|-----------|----------|----------|
| 1 | 11.66 | 10.98 | 9.73 |
| 25 | 9.41 | 10.01 | 9.43 |
| 50 | 9.31 | 9.92 | 9.34 |

## 项目结构

```
TTS/
├── src/
│   ├── model/
│   │   ├── vits.py              # VITS 主模型
│   │   ├── text_encoder.py      # Transformer 文本编码器
│   │   ├── duration_predictor.py # 时长预测器 + regulate_length
│   │   ├── decoder.py           # Conv1d 残差块 VAE 解码器
│   │   └── hifigan.py           # HiFi-GAN 声码器
│   ├── text/
│   │   ├── symbols.py           # ARPAbet 音素表 + ID 映射
│   │   ├── phonemizer.py        # 文本→音素转换
│   │   └── cleaners.py          # 文本清洗
│   ├── audio/
│   │   └── mel.py               # 梅尔频谱提取
│   ├── config.py                # 超参数配置
│   ├── dataset.py               # 数据集 + DataLoader
│   ├── train.py                 # 损失函数 + 训练循环
│   ├── infer.py                 # 推理引擎 + Griffin-Lim
│   └── preprocess.py            # LJSpeech 数据预处理
├── server.py                    # Flask Web API
├── run_train.py                 # 训练启动脚本
├── webui/
│   └── index.html               # Web 前端界面
├── data/
│   ├── raw/LJSpeech-1.1/        # 原始数据集
│   ├── processed/               # 预处理后的 mel + phonemes
│   └── checkpoints_v2/          # 训练 checkpoint
├── models/                      # 最终模型
├── docs/                        # 原理文档
│   ├── vae_decoder.md           # VAE 解码器原理
│   ├── text_encoder.md          # 文本编码器原理
│   ├── duration_predictor.md    # 时长预测器原理
│   ├── mel_spectrogram.md       # 梅尔频谱原理
│   ├── griffin_lim.md           # Griffin-Lim 算法原理
│   ├── hifigan.md               # HiFi-GAN 声码器原理
│   ├── training_pipeline.md     # 训练流程原理
│   └── fix_noise_issue.md       # 噪声问题修复记录
└── tests/                       # 单元测试
```

## 音频配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 采样率 | 22050 Hz | |
| FFT 窗口 | 1024 | 46ms |
| 帧移 | 256 | 11.6ms |
| Mel bins | 80 | |
| 输出格式 | int16 PCM WAV | 浏览器兼容 |

## 技术文档

详细原理文档在 [docs/](docs/) 目录：

- [VAE 解码器](docs/vae_decoder.md) — Conv1d 残差块架构、重参数化技巧
- [文本编码器](docs/text_encoder.md) — Transformer 自注意力编码
- [时长预测器](docs/duration_predictor.md) — 非自回归时长预测 + regulate_length
- [梅尔频谱](docs/mel_spectrogram.md) — STFT、梅尔滤波器组、dB 变换
- [Griffin-Lim](docs/griffin_lim.md) — 迭代相位重建算法
- [HiFi-GAN](docs/hifigan.md) — 神经声码器架构
- [训练流程](docs/training_pipeline.md) — 数据预处理、损失函数、训练策略
- [噪声修复](docs/fix_noise_issue.md) — 修复滋啦噪声的 4 个关键 bug

## 当前状态与后续计划

### 已完成

- VITS 完整推理管线（文本→音素→编码→时长→解码→mel→音频）
- Conv1d 残差块 Decoder 架构（替换原始 MLP）
- Griffin-Lim 声码器（librosa + 自实现两种路径）
- Flask Web API + 现代深色 WebUI
- LJSpeech 子集训练验证（1000 条, 50 epochs）
- Mel 分布匹配后处理

### 待改进

- [ ] 全量 LJSpeech 数据训练（12445 条, 200+ epochs, 需 GPU）
- [ ] 训练 HiFi-GAN 声码器替代 Griffin-Lim
- [ ] 中文语音合成支持
- [ ] 多说话人支持
