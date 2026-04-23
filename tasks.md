# TTS 原子任务列表 (tasks.md)

> 每个任务遵循 TDD：先写失败测试 → 再写最小实现 → 重构
> 状态标记：`[ ]` 待做 | `[x]` 完成 | `[~]` 进行中

---

## 阶段一：基础设施

### 1.1 项目初始化

- [ ] **T-001** 创建 `requirements.txt`，包含 torch/torchaudio/numpy/scipy/gruut/pypinyin/matplotlib/flask/pytest
- [ ] **T-002** 创建目录骨架：`src/`、`src/text/`、`src/audio/`、`src/model/`、`tests/`、`configs/`、`data/`、`models/`、`docs/`、`webui/`，各目录添加 `__init__.py`
- [ ] **T-003** 创建 `configs/default.json`，包含所有默认超参数（audio/text_encoder/decoder/duration_predictor/hifigan/train）

### 1.2 配置模块 `src/config.py`

- [ ] **T-004** 测试：`AudioConfig` 能以默认值实例化，`sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80`
- [ ] **T-005** 实现：`AudioConfig` dataclass
- [ ] **T-006** 测试：`TextEncoderConfig` 默认值 `vocab_size=256, hidden_dim=192, n_layers=2, n_heads=4`
- [ ] **T-007** 实现：`TextEncoderConfig` dataclass
- [ ] **T-008** 测试：`DecoderConfig` 默认值 `in_channels=96, out_channels=80, n_layers=6`
- [ ] **T-009** 实现：`DecoderConfig` dataclass
- [ ] **T-010** 测试：`DurationPredictorConfig` 默认值 `in_channels=192, n_layers=2`
- [ ] **T-011** 实现：`DurationPredictorConfig` dataclass
- [ ] **T-012** 测试：`HiFiGANConfig` 默认值 `upsample_rates=(8,8,2,2)`, 乘积等于 256
- [ ] **T-013** 实现：`HiFiGANConfig` dataclass
- [ ] **T-014** 测试：`TrainConfig` 默认值 `batch_size=16, lr=1e-4, epochs=200`
- [ ] **T-015** 实现：`TrainConfig` dataclass
- [ ] **T-016** 测试：`load_config(path)` 能从 JSON 文件加载并返回各 Config 对象
- [ ] **T-017** 实现：`load_config()` 函数，读取 `configs/default.json` 并构造 Config 对象

### 1.3 音素符号表 `src/text/symbols.py`

- [ ] **T-018** 测试：`PHONEME_LIST` 包含 PAD `"_"` 和基本英文音素（如 `"AA"`, `"AE"`, `"AH"` 等 ARPAbet）
- [ ] **T-019** 实现：定义 `PAD`、`PHONEME_LIST`（ARPAbet 音素 + 标点 + 特殊符号）
- [ ] **T-020** 测试：`SYMBOL_TO_ID["_"]` == 0，`ID_TO_SYMBOL[0]` == `"_"`
- [ ] **T-021** 实现：构建 `SYMBOL_TO_ID` 和 `ID_TO_SYMBOL` 双向映射
- [ ] **T-022** 测试：`text_to_sequence("hello", language="en")` 返回非空 int 列表，所有值在 `[0, vocab_size)` 范围内
- [ ] **T-023** 实现：`text_to_sequence()` 函数（调用 phonemizer → 查表映射）
- [ ] **T-024** 测试：`sequence_to_text(text_to_sequence("hello"))` 返回音素字符串
- [ ] **T-025** 实现：`sequence_to_text()` 函数

### 1.4 文本清洗 `src/text/cleaners.py`

- [ ] **T-026** 测试：`basic_cleaners("Hello, World!")` 返回 `"hello world"`（小写化 + 去除标点）
- [ ] **T-027** 实现：`basic_cleaners()` 函数
- [ ] **T-028** 测试：`english_cleaners("It's $100.")` 能处理缩写和数字
- [ ] **T-029** 实现：`english_cleaners()` 函数

### 1.5 文本转音素 `src/text/phonemizer.py`

- [ ] **T-030** 测试：`english_to_phonemes("hello")` 返回非空音素列表，每个元素是合法音素字符串
- [ ] **T-031** 实现：`english_to_phonemes()` 使用 gruut
- [ ] **T-032** 测试：`chinese_to_phonemes("你好")` 返回包含拼音的列表
- [ ] **T-033** 实现：`chinese_to_phonemes()` 使用 pypinyin
- [ ] **T-034** 测试：`text_to_phonemes("hello", "en")` 调用英文路径，`text_to_phonemes("你好", "zh")` 调用中文路径
- [ ] **T-035** 实现：`text_to_phonemes()` 统一入口

### 1.6 梅尔频谱提取 `src/audio/mel.py`

- [ ] **T-036** 测试：`MelSpectrogramExtractor(config)` 能实例化
- [ ] **T-037** 实现：`MelSpectrogramExtractor.__init__()`，初始化 torchaudio MelSpectrogram 变换
- [ ] **T-038** 测试：`wav_to_mel(waveform)` 输入 `(1, 22050)` 随机波形，输出 shape 为 `(80, T)` 且 `T > 0`
- [ ] **T-039** 实现：`wav_to_mel()` 方法
- [ ] **T-040** 测试：`extract(wav_path)` 能从实际 wav 文件提取梅尔频谱，shape 为 `(80, T)`
- [ ] **T-041** 实现：`extract()` 方法（加载 wav → 重采样 → wav_to_mel）

---

## 阶段二：模型实现

### 2.1 文本编码器 `src/model/text_encoder.py`

- [ ] **T-042** 测试：`TextEncoder(config)` 能实例化，参数量在合理范围（< 5M）
- [ ] **T-043** 实现：`TextEncoder.__init__()`，包含 Embedding + PositionalEncoding + 2 层 TransformerEncoderLayer
- [ ] **T-044** 测试：`forward(x, x_lengths)` 输入 `x=(2, 10)` int, `x_lengths=(2,)`，输出 shape 为 `(2, 10, 192)` 和 mask `(2, 1, 10)`
- [ ] **T-045** 实现：`TextEncoder.forward()`，包含 embedding → positional encoding → transformer → mask 生成
- [ ] **T-046** 测试：不同长度输入的 mask 正确（短序列被 mask 的位置输出为 0）
- [ ] **T-047** 实现：mask 生成逻辑，确保 padding 位置不参与注意力计算
- [ ] **T-048** 编写原理文档 `docs/text_encoder.md`

### 2.2 时长预测器 `src/model/duration_predictor.py`

- [ ] **T-049** 测试：`DurationPredictor(config)` 能实例化
- [ ] **T-050** 实现：`DurationPredictor.__init__()`，2 层 Conv1d + ReLU + LayerNorm + Linear
- [ ] **T-051** 测试：`forward(x, x_mask)` 输入 `(2, 10, 192)`，输出 shape `(2, 10)`
- [ ] **T-052** 实现：`DurationPredictor.forward()`
- [ ] **T-053** 测试：`regulate_length(encoder_output, durations)` 输入 `(1, 3, 192)` + durations `[2, 3, 1]`，输出 shape `(1, 6, 192)`
- [ ] **T-054** 实现：`regulate_length()` 函数（按 duration 重复展开每个时间步）
- [ ] **T-055** 测试：batch 内不同 duration 总和时，输出正确 padding 到最大长度
- [ ] **T-056** 实现：batch 场景下的 regulate_length 填充逻辑
- [ ] **T-057** 编写原理文档 `docs/duration_predictor.md`

### 2.3 VAE 解码器 `src/model/decoder.py`

- [ ] **T-058** 测试：`Decoder(config)` 能实例化
- [ ] **T-059** 实现：`Decoder.__init__()`，线性投影 + 6 层前馈网络（Linear + ReLU）+ 输出投影到 80 维
- [ ] **T-060** 测试：`forward(z)` 输入 `(2, 50, 96)`，输出 shape `(2, 80, 50)`
- [ ] **T-061** 实现：`Decoder.forward()`
- [ ] **T-062** 测试：VAE 采样逻辑——给定 `mu` 和 `log_var`，`reparameterize(mu, log_var)` 输出 shape 与 mu 一致
- [ ] **T-063** 实现：`reparameterize()` 函数（mu + exp(0.5 * log_var) * eps）
- [ ] **T-064** 编写原理文档 `docs/vae_decoder.md`

### 2.4 VITS 主模型 `src/model/vits.py`

- [ ] **T-065** 测试：`VITS(config)` 能实例化，包含 text_encoder / duration_predictor / decoder 子模块
- [ ] **T-066** 实现：`VITS.__init__()`，组装子模块 + mu/log_var 投影层
- [ ] **T-067** 测试：`forward()` 训练模式，输入 phoneme_ids `(2, 10)` + mel_targets `(2, 80, 50)` + 对应 lengths，返回 dict 包含 `mel_output / duration_pred / mu / log_var / z`
- [ ] **T-068** 实现：`VITS.forward()` 训练前向传播（encoder → duration_pred → regulate → mu/logvar → sample → decode）
- [ ] **T-069** 测试：`infer()` 推理模式，输入 phoneme_ids `(1, 10)`，返回 mel shape `(1, 80, M)` 且 `M > 0`
- [ ] **T-070** 实现：`VITS.infer()`（encoder → predict_duration → regulate → sample → decode，无需 teacher forcing）
- [ ] **T-071** 测试：模型总参数量在 3-5M 范围内
- [ ] **T-072** 调整参数使模型大小符合规格

### 2.5 HiFi-GAN 声码器 `src/model/hifigan.py`

- [ ] **T-073** 测试：`HiFiGAN(config)` 能实例化，参数量在 1-2M 范围
- [ ] **T-074** 实现：`HiFiGAN.__init__()`，包含 pre_conv + 4 层上采样（ConvTranspose1d）+ 残差块（ResBlock）+ post_conv
- [ ] **T-075** 测试：`ResBlock` 子模块输入输出 shape 一致
- [ ] **T-076** 实现：`ResBlock`（多膨胀率卷积 + 残差连接）
- [ ] **T-077** 测试：`forward(mel)` 输入 `(2, 80, 50)`，输出 shape `(2, 1, 50*256)` = `(2, 1, 12800)`
- [ ] **T-078** 实现：`HiFiGAN.forward()`，依次通过上采样层 + 残差块 + tanh 激活
- [ ] **T-079** 测试：输出波形值范围在 `[-1, 1]`
- [ ] **T-080** 确认 tanh 最终激活层
- [ ] **T-081** 编写原理文档 `docs/hifigan.md`

---

## 阶段三：数据与训练

### 3.1 数据预处理 `src/preprocess.py`

- [ ] **T-082** 测试：`preprocess_ljspeech()` 给定一个含 3 条样本的 mock 数据目录，生成 `mels/`、`phonemes/`、`filelists/` 目录
- [ ] **T-083** 实现：`preprocess_ljspeech()` 主流程（读 metadata.csv → 遍历处理）
- [ ] **T-084** 测试：生成的 mel `.npy` 文件 shape 为 `(80, T)`，phoneme `.npy` 为 1D int 数组
- [ ] **T-085** 实现：单条样本的预处理逻辑（text → phonemes → ids → save; wav → mel → save）
- [ ] **T-086** 测试：`train.txt` 和 `val.txt` 文件正确划分，val 比例约 5%
- [ ] **T-087** 实现：文件列表划分逻辑

### 3.2 数据集 `src/dataset.py`

- [ ] **T-088** 测试：`TTSDataset(filelist_path)` 能实例化，`len()` 返回文件列表行数
- [ ] **T-089** 实现：`TTSDataset.__init__()` 和 `__len__()`
- [ ] **T-090** 测试：`__getitem__(0)` 返回 dict 包含 `phoneme_ids` (1D Tensor)、`mel` (2D Tensor, 80×M)、`duration` (1D Tensor)
- [ ] **T-091** 实现：`TTSDataset.__getitem__()`，从 .npy 文件加载数据
- [ ] **T-092** 测试：`collate_fn(batch)` 将 3 条不同长度样本 padding 为统一 batch，验证各字段 shape 正确
- [ ] **T-093** 实现：`collate_fn()`，按 batch 内最大长度 padding
- [ ] **T-094** 测试：DataLoader 能正常迭代一个 batch
- [ ] **T-095** 实现：确保 DataLoader 与 collate_fn 集成正确

### 3.3 损失函数 `src/train.py` (loss 部分)

- [ ] **T-096** 测试：`compute_loss()` 输入 mock model_output 和 targets，返回 dict 包含 `total_loss / mel_loss / duration_loss / kl_loss`，各值为标量 Tensor
- [ ] **T-097** 实现：`compute_loss()` 函数（L1 mel_loss + MSE duration_loss + KL kl_loss）
- [ ] **T-098** 测试：KL loss 当 mu=0, log_var=0 时约等于 0
- [ ] **T-099** 实现：KL 散度计算 `-0.5 * sum(1 + log_var - mu^2 - exp(log_var))`

### 3.4 训练循环 `src/train.py` (train 部分)

- [ ] **T-100** 测试：`train()` 在 mock 小数据（3 条）上跑 2 个 epoch 不报错，loss 有输出
- [ ] **T-101** 实现：训练主循环（加载数据 → 初始化模型/优化器 → epoch 循环 → forward → loss → backward → step）
- [ ] **T-102** 测试：checkpoint 保存逻辑——每 `save_interval` 轮保存 `.pt` 文件
- [ ] **T-103** 实现：checkpoint 保存（model state_dict + optimizer state_dict + epoch + loss）
- [ ] **T-104** 测试：验证集评估逻辑——验证 loss 被正确计算且不更新梯度
- [ ] **T-105** 实现：验证循环（torch.no_grad + eval mode）
- [ ] **T-106** 测试：checkpoint 加载 & 断点续训，epoch 从上次继续
- [ ] **T-107** 实现：`resume_from_checkpoint()` 逻辑
- [ ] **T-108** 测试：设备选择逻辑（MPS 可用时用 MPS，否则 CPU）
- [ ] **T-109** 实现：设备自动检测 `torch.backends.mps.is_available()`

### 3.5 实际训练

- [ ] **T-110** 下载 LJSpeech 数据集到 `data/raw/LJSpeech-1.1/`
- [ ] **T-111** 运行 `preprocess.py` 生成预处理数据
- [ ] **T-112** 运行训练脚本，监控 loss 曲线，确认 loss 下降 ≥ 50%
- [ ] **T-113** 保存最终模型 `models/vits_final.pt`
- [ ] **T-114** 编写原理文档 `docs/training_pipeline.md`
- [ ] **T-115** 编写原理文档 `docs/mel_spectrogram.md`

---

## 阶段四：推理与服务

### 4.1 推理引擎 `src/infer.py`

- [ ] **T-116** 测试：`TTSInferencer(vits_path, hifigan_path)` 能加载模型权重并实例化
- [ ] **T-117** 实现：`TTSInferencer.__init__()`，加载 VITS + HiFiGAN 模型
- [ ] **T-118** 测试：`synthesize("hello world", "en")` 返回 1D numpy 数组，值在 `[-1, 1]`，长度 > 0
- [ ] **T-119** 实现：`synthesize()` 方法（text → phonemes → ids → VITS infer → HiFiGAN → numpy）
- [ ] **T-120** 测试：`synthesize_to_file("hello", "output.wav")` 生成有效 wav 文件，可被 `wave` 库读取
- [ ] **T-121** 实现：`synthesize_to_file()` 方法（synthesize → scipy.io.wavfile.write）
- [ ] **T-122** 测试：推理 20 字英文文本耗时 < 2 秒（Mac M1/M2）
- [ ] **T-123** 必要时做推理优化（torch.no_grad, eval mode, torch.compile）

### 4.2 模型导出 `src/export.py`

- [ ] **T-124** 测试：VITS 模型能导出为 TorchScript 格式并重新加载推理
- [ ] **T-125** 实现：`export_vits_torchscript()` 函数
- [ ] **T-126** 测试：HiFiGAN 模型能导出为 TorchScript 格式并重新加载推理
- [ ] **T-127** 实现：`export_hifigan_torchscript()` 函数

### 4.3 Web API `server.py`

- [ ] **T-128** 测试：`GET /api/health` 返回 `{"status": "ok", "model_loaded": true}`
- [ ] **T-129** 实现：Flask app + health 路由
- [ ] **T-130** 测试：`POST /api/tts` 请求 `{"text": "hello", "language": "en"}` 返回 `audio/wav` 响应，status 200
- [ ] **T-131** 实现：TTS 路由（接收 JSON → TTSInferencer.synthesize → 返回 wav 二进制流）
- [ ] **T-132** 测试：缺少 text 字段时返回 400 错误
- [ ] **T-133** 实现：请求参数校验

### 4.4 Web 前端 `webui/`

- [ ] **T-134** 创建 `webui/index.html`：包含文本输入框、语言选择（en/zh）、合成按钮、音频播放器、下载按钮
- [ ] **T-135** 创建 `webui/style.css`：基础样式
- [ ] **T-136** 实现前端 JS：点击合成 → fetch POST /api/tts → 播放音频 + 提供下载
- [ ] **T-137** 在 `server.py` 中添加静态文件服务，`GET /` 返回 index.html

### 4.5 一键运行脚本

- [ ] **T-138** 创建 `run.sh`：安装依赖 → 启动 Flask 服务 → 打开浏览器

---

## 阶段五：验收与优化

### 5.1 模型训练验收

- [ ] **T-139** 验收：训练脚本在 Mac 上无报错运行
- [ ] **T-140** 验收：训练 loss 从初始值下降 ≥ 50%
- [ ] **T-141** 验收：验证集 loss 连续 10 轮无明显上升
- [ ] **T-142** 验收：`models/vits_final.pt` 文件存在且可加载

### 5.2 推理验收

- [ ] **T-143** 验收：输入 "hello world" 生成有效 .wav 文件且可播放
- [ ] **T-144** 验收：音频时长与文本长度匹配（0.2-0.4 秒/字）
- [ ] **T-145** 验收：合成 20 字英文 < 2 秒

### 5.3 整体验收

- [ ] **T-146** 验收：Web 界面能正常打开
- [ ] **T-147** 验收：文本能转换为语音并在页面播放
- [ ] **T-148** 验收：音频能下载保存

---

## 任务依赖关系

```
T-001~003 (项目初始化)
    │
    ▼
T-004~017 (配置模块) ─────────────────────────────────┐
    │                                                   │
    ▼                                                   │
T-018~025 (音素符号表)                                   │
    │                                                   │
    ▼                                                   │
T-026~029 (文本清洗)                                     │
    │                                                   │
    ▼                                                   │
T-030~035 (文本转音素)                                   │
    │                                                   │
    ▼                                                   │
T-036~041 (梅尔频谱) ◄─────────────────────────────────┘
    │
    ├──────────────────────────────┐
    ▼                              ▼
T-042~048 (文本编码器)        T-073~081 (HiFi-GAN)
    │                              │
    ▼                              │
T-049~057 (时长预测器)              │
    │                              │
    ▼                              │
T-058~064 (VAE 解码器)             │
    │                              │
    ▼                              │
T-065~072 (VITS 主模型) ◄─────────┘
    │
    ▼
T-082~095 (数据预处理 + 数据集)
    │
    ▼
T-096~109 (损失函数 + 训练循环)
    │
    ▼
T-110~115 (实际训练)
    │
    ├──────────────────────────────┐
    ▼                              ▼
T-116~123 (推理引擎)         T-124~127 (模型导出)
    │
    ▼
T-128~138 (Web API + 前端 + run.sh)
    │
    ▼
T-139~148 (验收)
```

---

## 统计

| 阶段 | 任务数 | 范围 |
|------|--------|------|
| 阶段一：基础设施 | 41 | T-001 ~ T-041 |
| 阶段二：模型实现 | 40 | T-042 ~ T-081 |
| 阶段三：数据与训练 | 34 | T-082 ~ T-115 |
| 阶段四：推理与服务 | 23 | T-116 ~ T-138 |
| 阶段五：验收与优化 | 10 | T-139 ~ T-148 |
| **总计** | **148** | |

---

*创建日期：2026/04/22*
