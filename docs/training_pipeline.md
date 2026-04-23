# 训练流程文档

## 1. 概述

本文档描述 VITS 模型的完整训练流程，包括数据预处理、模型训练和推理。

## 2. 数据预处理

### 2.1 数据集格式

LJSpeech 数据集格式：
```
data_dir/
├── metadata.csv          # 格式: filename|text
├── wavs/                 # 音频文件目录
│   ├── 1.wav
│   ├── 2.wav
│   └── ...
```

### 2.2 预处理步骤

1. 读取 `metadata.csv` 获取文本和对应音频文件
2. 对每个音频文件提取梅尔频谱 (80 mel bins)
3. 将文本转换为音素 ID 序列
4. 保存梅尔频谱和音素 ID 为 `.npy` 文件
5. 生成 `train.txt` 和 `val.txt` 文件列表

### 2.3 运行预处理

```bash
python -m src.preprocess --data_dir data/raw/LJSpeech-1.1 --output_dir data/processed
```

## 3. 模型训练

### 3.1 训练命令

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train.py \
    --train_filelist data/processed/filelists/train.txt \
    --val_filelist data/processed/filelists/val.txt \
    --data_dir data/processed \
    --checkpoint_dir data/checkpoints \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 16
```

### 3.2 训练输出

- Checkpoints 保存在 `data/checkpoints/`
- 文件名格式: `checkpoint_epoch_{N}.pt`
- 包含: model_state_dict, optimizer_state_dict, epoch, train_loss

### 3.3 训练损失

训练过程包含三个损失项：
1. **Mel Loss (L1)**: 梅尔频谱重建损失
2. **Duration Loss (MSE)**: 时长预测损失
3. **KL Loss**: VAE KL 散度损失

总损失 = Mel Loss + Duration Loss + KL Loss

## 4. 推理

### 4.1 推理命令

```python
from infer import TTSInferencer

inferencer = TTSInferencer(
    vits_path='data/checkpoints/checkpoint_epoch_200.pt',
    hifigan_path='data/models/hifigan.pt'
)

# 合成音频
waveform = inferencer.synthesize("hello world", "en")

# 保存为 WAV 文件
inferencer.synthesize_to_file("hello world", "output.wav", "en")
```

### 4.2 Web API

启动 Flask 服务：
```bash
python server.py
```

API 端点：
- `GET /api/health` - 健康检查
- `POST /api/tts` - 合成语音

## 5. 在 Mac 上运行

由于 PyTorch 的 Transformer 在 MPS 后端不支持某些操作，需要设置环境变量：

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

这会让不支持的操作回退到 CPU 运行。

## 6. 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 200 | 训练轮数 |
| lr | 1e-4 | 学习率 |
| batch_size | 16 | 批次大小 |
| save_interval | 10 | checkpoint 保存间隔 |

## 7. 模型文件

训练完成后，最终模型保存在：
- `data/checkpoints/checkpoint_epoch_200.pt` (或最后保存的 epoch)
- `models/vits_final.pt` (手动复制)
