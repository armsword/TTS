# TTS 技术方案 (plan.md)

## 1. 技术选型

| 类别 | 选型 | 版本要求 | 说明 |
|------|------|----------|------|
| 语言 | Python | >= 3.9 | 深度学习生态最成熟 |
| 深度学习框架 | PyTorch | >= 2.0 | 支持 MPS 加速 |
| 音频处理 | torchaudio | >= 2.0 | 与 PyTorch 深度集成 |
| 文本转音素（英文） | gruut | latest | 英文 G2P |
| 文本转音素（中文） | pypinyin | latest | 中文拼音转换 |
| 数值计算 | numpy | latest | 基础数值库 |
| 信号处理 | scipy | latest | 音频信号处理 |
| 可视化 | matplotlib | latest | 训练曲线、频谱图 |
| Web 服务 | Flask | latest | 轻量 API 服务 |
| 测试框架 | pytest | latest | TDD 驱动开发 |

---

## 2. 目录结构

```
TTS/
├── spec.md                     # 需求规格文档
├── plan.md                     # 本技术方案
├── CLAUDE.md                   # 项目指令
├── tasks.md                    # 原子任务列表
├── README.md                   # 使用说明
├── requirements.txt            # Python 依赖
├── run.sh                      # 一键运行脚本
│
├── docs/                       # 原理文档
│   ├── text_encoder.md         # 文本编码器原理
│   ├── vae_decoder.md          # VAE 解码器原理
│   ├── duration_predictor.md   # 时长预测器原理
│   ├── hifigan.md              # HiFi-GAN 声码器原理
│   ├── mel_spectrogram.md      # 梅尔频谱原理
│   └── training_pipeline.md    # 训练流程原理
│
├── data/                       # 数据目录
│   ├── raw/                    # 原始数据（LJSpeech 下载到这里）
│   │   └── LJSpeech-1.1/
│   │       ├── wavs/
│   │       └── metadata.csv
│   └── processed/              # 预处理后数据
│       ├── mels/               # 梅尔频谱 .npy 文件
│       ├── phonemes/           # 音素序列 .npy 文件
│       └── filelists/          # 训练/验证集划分
│           ├── train.txt
│           └── val.txt
│
├── configs/                    # 配置文件
│   └── default.json            # 默认超参数配置
│
├── src/                        # 源代码
│   ├── __init__.py
│   ├── config.py               # 配置加载与数据类
│   ├── text/                   # 文本处理模块
│   │   ├── __init__.py
│   │   ├── symbols.py          # 音素符号表
│   │   ├── cleaners.py         # 文本清洗
│   │   └── phonemizer.py       # 文本转音素（G2P）
│   ├── audio/                  # 音频处理模块
│   │   ├── __init__.py
│   │   └── mel.py              # 梅尔频谱提取
│   ├── model/                  # 模型定义
│   │   ├── __init__.py
│   │   ├── text_encoder.py     # 文本编码器（Transformer）
│   │   ├── duration_predictor.py # 时长预测器
│   │   ├── decoder.py          # VAE 解码器
│   │   ├── vits.py             # VITS 主模型（组装各子模块）
│   │   └── hifigan.py          # HiFi-GAN 声码器
│   ├── dataset.py              # Dataset & DataLoader
│   ├── preprocess.py           # 数据预处理脚本
│   ├── train.py                # 训练脚本
│   ├── infer.py                # 推理脚本
│   └── export.py               # 模型导出（ONNX/TorchScript）
│
├── models/                     # 保存的模型权重
│   └── .gitkeep
│
├── tests/                      # 测试代码
│   ├── __init__.py
│   ├── test_symbols.py         # 音素符号表测试
│   ├── test_phonemizer.py      # G2P 测试
│   ├── test_mel.py             # 梅尔频谱测试
│   ├── test_text_encoder.py    # 文本编码器测试
│   ├── test_duration_predictor.py
│   ├── test_decoder.py         # 解码器测试
│   ├── test_vits.py            # VITS 模型集成测试
│   ├── test_hifigan.py         # HiFi-GAN 测试
│   ├── test_dataset.py         # 数据集测试
│   └── test_infer.py           # 推理端到端测试
│
├── webui/                      # Web 前端
│   ├── index.html
│   └── style.css
│
└── server.py                   # Flask API 服务
```

---

## 3. 核心数据模型

### 3.1 配置数据类

```python
@dataclass
class AudioConfig:
    sample_rate: int = 22050        # 采样率
    n_fft: int = 1024               # FFT 窗口大小
    hop_length: int = 256           # 帧移（决定上采样倍率 = sample_rate / hop_length）
    win_length: int = 1024          # 窗长
    n_mels: int = 80                # 梅尔频带数
    mel_fmin: float = 0.0           # 梅尔最低频率
    mel_fmax: float = 8000.0        # 梅尔最高频率

@dataclass
class TextEncoderConfig:
    vocab_size: int = 256           # 音素词表大小
    hidden_dim: int = 192           # 隐藏层维度
    n_layers: int = 2               # Transformer 层数
    n_heads: int = 4                # 注意力头数
    ff_dim: int = 768               # 前馈层维度
    dropout: float = 0.1

@dataclass
class DecoderConfig:
    in_channels: int = 96           # VAE 隐变量维度
    hidden_dim: int = 192           # 隐藏层维度
    out_channels: int = 80          # 输出梅尔维度
    n_layers: int = 6               # 前馈层数

@dataclass
class DurationPredictorConfig:
    in_channels: int = 192          # 输入维度（= TextEncoder hidden_dim）
    hidden_dim: int = 128           # 隐藏层维度
    n_layers: int = 2               # 卷积层数
    kernel_size: int = 3
    dropout: float = 0.1

@dataclass
class HiFiGANConfig:
    in_channels: int = 80           # 梅尔频带数
    upsample_rates: list = (8, 8, 2, 2)       # 上采样倍率，乘积 = 256 = hop_length
    upsample_kernel_sizes: list = (16, 16, 4, 4)
    resblock_kernel_sizes: list = (3, 7, 11)   # 残差块卷积核
    resblock_dilation_sizes: list = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    initial_channel: int = 256

@dataclass
class TrainConfig:
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 200
    max_seq_len: int = 512          # 最大序列长度
    val_ratio: float = 0.05         # 验证集比例
    save_interval: int = 10         # 每 N 轮保存
    log_interval: int = 100         # 每 N 步打印日志
    seed: int = 42
    device: str = "mps"             # mac 默认用 MPS
```

### 3.2 核心张量流

```
符号说明：B=batch, T=文本长度, C=通道数, M=梅尔帧数, W=波形采样点数

文本输入 (B, T)                     # int64, 音素 ID 序列
    │
    ▼ TextEncoder
文本隐表示 (B, T, 192)              # float32, Transformer 输出
    │
    ├─▶ DurationPredictor → 时长 (B, T)     # float32, 每个音素对应帧数
    │
    ▼ 长度调节器（Length Regulator）：按时长展开
声学隐表示 (B, M, 192)              # float32, 帧级特征
    │
    ▼ VAE μ/σ 投影 → 采样
隐变量 z (B, M, 96)                 # float32, VAE bottleneck
    │
    ▼ Decoder
梅尔频谱 (B, 80, M)                 # float32, 梅尔频谱
    │
    ▼ HiFi-GAN
波形 (B, 1, W)                      # float32, W = M × 256
```

---

## 4. 接口定义

### 4.1 文本处理接口

```python
# src/text/symbols.py
PAD: str = "_"                              # 填充符号
PHONEME_LIST: list[str]                     # 所有音素列表
SYMBOL_TO_ID: dict[str, int]                # 音素 → ID 映射
ID_TO_SYMBOL: dict[int, str]                # ID → 音素 映射

def text_to_sequence(text: str, language: str = "en") -> list[int]:
    """文本 → 音素 ID 序列"""

def sequence_to_text(sequence: list[int]) -> str:
    """音素 ID 序列 → 可读文本（调试用）"""
```

```python
# src/text/phonemizer.py
def english_to_phonemes(text: str) -> list[str]:
    """英文文本 → 音素列表，使用 gruut"""

def chinese_to_phonemes(text: str) -> list[str]:
    """中文文本 → 拼音音素列表，使用 pypinyin"""

def text_to_phonemes(text: str, language: str = "en") -> list[str]:
    """统一入口：文本 → 音素列表"""
```

```python
# src/text/cleaners.py
def basic_cleaners(text: str) -> str:
    """基础文本清洗：小写化、去除特殊字符"""

def english_cleaners(text: str) -> str:
    """英文文本清洗"""
```

### 4.2 音频处理接口

```python
# src/audio/mel.py
class MelSpectrogramExtractor:
    def __init__(self, config: AudioConfig): ...

    def extract(self, wav_path: str) -> torch.Tensor:
        """从 wav 文件提取梅尔频谱
        返回: (n_mels, n_frames) 的 Tensor
        """

    def wav_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """从波形张量提取梅尔频谱
        输入: (1, samples)
        返回: (n_mels, n_frames)
        """
```

### 4.3 模型接口

```python
# src/model/text_encoder.py
class TextEncoder(nn.Module):
    def __init__(self, config: TextEncoderConfig): ...

    def forward(self, x: Tensor, x_lengths: Tensor) -> tuple[Tensor, Tensor]:
        """
        输入: x (B, T) 音素 ID, x_lengths (B,) 实际长度
        输出: (encoder_output (B, T, hidden_dim), x_mask (B, 1, T))
        """
```

```python
# src/model/duration_predictor.py
class DurationPredictor(nn.Module):
    def __init__(self, config: DurationPredictorConfig): ...

    def forward(self, x: Tensor, x_mask: Tensor) -> Tensor:
        """
        输入: x (B, T, hidden_dim), x_mask (B, 1, T)
        输出: durations (B, T) 预测的每音素帧数（log domain）
        """

def regulate_length(encoder_output: Tensor, durations: Tensor) -> tuple[Tensor, Tensor]:
    """长度调节器：将文本级特征按时长展开为帧级特征
    输入: encoder_output (B, T, C), durations (B, T)
    输出: (regulated (B, M, C), mel_lengths (B,))
    """
```

```python
# src/model/decoder.py
class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig): ...

    def forward(self, z: Tensor) -> Tensor:
        """
        输入: z (B, M, in_channels) 隐变量
        输出: mel (B, 80, M) 梅尔频谱
        """
```

```python
# src/model/vits.py
class VITS(nn.Module):
    def __init__(self, config: VITSConfig): ...

    def forward(self, phoneme_ids: Tensor, phoneme_lengths: Tensor,
                mel_targets: Tensor, mel_lengths: Tensor) -> dict:
        """训练前向传播
        返回: {
            "mel_output": Tensor,        # 预测梅尔 (B, 80, M)
            "duration_pred": Tensor,     # 预测时长 (B, T)
            "mu": Tensor,                # VAE 均值
            "log_var": Tensor,           # VAE 对数方差
            "z": Tensor,                 # 采样隐变量
        }
        """

    def infer(self, phoneme_ids: Tensor, phoneme_lengths: Tensor) -> Tensor:
        """推理：文本 → 梅尔频谱
        返回: mel (1, 80, M)
        """
```

```python
# src/model/hifigan.py
class HiFiGAN(nn.Module):
    def __init__(self, config: HiFiGANConfig): ...

    def forward(self, mel: Tensor) -> Tensor:
        """梅尔频谱 → 波形
        输入: mel (B, 80, M)
        输出: waveform (B, 1, M * hop_length)
        """
```

### 4.4 数据集接口

```python
# src/dataset.py
class TTSDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path: str, config: AudioConfig): ...

    def __getitem__(self, index: int) -> dict:
        """返回单条训练样本
        {
            "phoneme_ids": Tensor,   # (T,)
            "mel": Tensor,           # (80, M)
            "duration": Tensor,      # (T,) 每音素对应帧数（训练用）
        }
        """

    def __len__(self) -> int: ...

def collate_fn(batch: list[dict]) -> dict:
    """将变长样本填充为统一 batch
    返回: {
        "phoneme_ids": (B, T_max),
        "phoneme_lengths": (B,),
        "mel": (B, 80, M_max),
        "mel_lengths": (B,),
        "duration": (B, T_max),
    }
    """
```

### 4.5 训练接口

```python
# src/train.py
def train(config_path: str) -> None:
    """训练入口
    1. 加载配置
    2. 初始化 Dataset / DataLoader
    3. 初始化模型 + 优化器
    4. 训练循环：forward → loss → backward → step
    5. 验证 & 保存 checkpoint
    """

def compute_loss(model_output: dict, targets: dict) -> dict:
    """计算训练损失
    返回: {
        "total_loss": Tensor,
        "mel_loss": Tensor,          # 梅尔重建 L1 损失
        "duration_loss": Tensor,     # 时长预测 MSE 损失
        "kl_loss": Tensor,           # VAE KL 散度
    }
    """
```

### 4.6 推理接口

```python
# src/infer.py
class TTSInferencer:
    def __init__(self, vits_path: str, hifigan_path: str, device: str = "cpu"): ...

    def synthesize(self, text: str, language: str = "en") -> np.ndarray:
        """文本 → 波形 numpy 数组
        返回: waveform (samples,) float32, 范围 [-1, 1]
        """

    def synthesize_to_file(self, text: str, output_path: str, language: str = "en") -> None:
        """文本 → 保存为 .wav 文件"""
```

### 4.7 预处理接口

```python
# src/preprocess.py
def preprocess_ljspeech(data_dir: str, output_dir: str, config: AudioConfig) -> None:
    """LJSpeech 数据预处理
    1. 读取 metadata.csv
    2. 文本 → 音素序列 → 保存 .npy
    3. 音频 → 梅尔频谱 → 保存 .npy
    4. 生成 train.txt / val.txt 文件列表
    """
```

### 4.8 Web API 接口

```python
# server.py
# POST /api/tts
# 请求体: {"text": "hello world", "language": "en"}
# 响应: audio/wav 二进制流

# GET /api/health
# 响应: {"status": "ok", "model_loaded": true}
```

---

## 5. 损失函数设计

```
总损失 = mel_loss + duration_loss + kl_loss

mel_loss      = L1Loss(predicted_mel, target_mel)          # 梅尔重建损失
duration_loss = MSELoss(predicted_duration, target_duration)  # 时长预测损失
kl_loss       = KL(N(mu, sigma) || N(0, 1))                # VAE 正则化
```

HiFi-GAN 声码器单独训练时的损失：
```
vocoder_loss = L1Loss(predicted_mel, target_mel) + L1Loss(predicted_wav, target_wav)
```

---

## 6. 实施阶段

### 阶段一：基础设施（预计 1 天）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 1.1 | 项目初始化：requirements.txt、configs/default.json | 可安装的 Python 环境 |
| 1.2 | 配置模块 `src/config.py` | 各模块配置 dataclass |
| 1.3 | 音素符号表 `src/text/symbols.py` | SYMBOL_TO_ID / ID_TO_SYMBOL 映射 |
| 1.4 | 文本清洗 `src/text/cleaners.py` | basic_cleaners / english_cleaners |
| 1.5 | 文本转音素 `src/text/phonemizer.py` | english_to_phonemes / text_to_phonemes |
| 1.6 | 梅尔频谱提取 `src/audio/mel.py` | MelSpectrogramExtractor |

### 阶段二：模型实现（预计 2-3 天）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 2.1 | 文本编码器 `src/model/text_encoder.py` | TextEncoder (2层 Transformer) |
| 2.2 | 时长预测器 `src/model/duration_predictor.py` | DurationPredictor + regulate_length |
| 2.3 | VAE 解码器 `src/model/decoder.py` | Decoder (6层前馈) |
| 2.4 | VITS 主模型 `src/model/vits.py` | VITS（组装各子模块，训练+推理） |
| 2.5 | HiFi-GAN 声码器 `src/model/hifigan.py` | HiFiGAN（4层上采样+残差块） |

### 阶段三：数据与训练（预计 1-2 天）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 3.1 | 数据预处理 `src/preprocess.py` | LJSpeech → mels + phonemes + filelists |
| 3.2 | 数据集 `src/dataset.py` | TTSDataset + collate_fn |
| 3.3 | 训练脚本 `src/train.py` | 完整训练循环 + loss 计算 + checkpoint |
| 3.4 | LJSpeech 实际训练 | models/vits_final.pt + models/hifigan_final.pt |

### 阶段四：推理与服务（预计 1 天）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 4.1 | 推理引擎 `src/infer.py` | TTSInferencer: 文本 → wav |
| 4.2 | 模型导出 `src/export.py` | ONNX / TorchScript 导出 |
| 4.3 | Web API `server.py` | Flask 服务 + /api/tts 接口 |
| 4.4 | Web 前端 `webui/` | 输入文本 → 播放/下载音频 |

### 阶段五：验收与优化（预计 1 天）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 5.1 | 按验收标准逐项测试 | 验收报告 |
| 5.2 | 推理速度优化（torch.compile 等） | 满足 20 字 < 2 秒 |
| 5.3 | 编写原理文档 docs/ | 各模块详细原理说明 |

---

## 7. 关键技术决策

| 决策点 | 方案 | 理由 |
|--------|------|------|
| 时长对齐 | 使用外部工具（Montreal Forced Aligner）预提取时长，或训练时用注意力机制估算 | 简化版先用注意力对齐，降低依赖 |
| VAE 采样 | 重参数化技巧（reparameterization trick） | 标准做法，允许梯度反传 |
| HiFi-GAN 训练 | 先联合训练，如效果不佳再拆分单独训练 | 减少训练复杂度 |
| 设备选择 | 优先 MPS，fallback 到 CPU | Mac 原生加速 |
| 混合精度 | 默认关闭，MPS 兼容性问题时再开启 | MPS 对 fp16 支持有限 |

---

## 8. 风险预案

| 风险 | 预案 |
|------|------|
| MPS 不支持某些算子 | fallback 到 CPU 执行该算子，其余保持 MPS |
| 训练不收敛 | 1) 检查数据预处理 2) 降低 lr 3) 先只训练子模块验证 |
| 梅尔频谱数值范围不对 | 统一使用 log-mel，范围 clip 到 [-12, 2] |
| 内存不足 | 减小 batch_size 到 4，缩短 max_seq_len 到 256 |

---

*创建日期：2026/04/22*
