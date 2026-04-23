"""数据预处理模块 - LJSpeech 数据集预处理"""
import os
import sys
import numpy as np
from pathlib import Path

# 添加 src 目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from audio.mel import MelSpectrogramExtractor
from text.symbols import text_to_sequence
from config import AudioConfig


def preprocess_ljspeech(data_dir: str, output_dir: str):
    """预处理 LJSpeech 数据集

    Args:
        data_dir: LJSpeech 数据集根目录（包含 metadata.csv 和 wavs/）
        output_dir: 输出目录
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # 创建输出目录
    mels_dir = output_dir / "mels"
    phonemes_dir = output_dir / "phonemes"
    filelists_dir = output_dir / "filelists"

    mels_dir.mkdir(parents=True, exist_ok=True)
    phonemes_dir.mkdir(parents=True, exist_ok=True)
    filelists_dir.mkdir(parents=True, exist_ok=True)

    # 读取 metadata
    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.exists():
        for fname in ["metadata.csv"]:
            if (data_dir / fname).exists():
                metadata_path = data_dir / fname
                break

    # 读取数据行
    items = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                filename = parts[0]
                text = parts[1]
                items.append((filename, text))

    # 配置
    audio_config = AudioConfig()
    mel_extractor = MelSpectrogramExtractor(audio_config)

    # 处理每条样本
    processed_items = []
    for filename, text in items:
        # 查找音频文件（filename 可能是 "1.wav" 或 "1"）
        wav_path = None
        base_filename = filename

        # 尝试不同的路径组合
        possible_paths = [
            data_dir / "wavs" / base_filename,
            data_dir / "wavs" / f"{base_filename}.wav",
            data_dir / base_filename,
            data_dir / f"{base_filename}.wav",
        ]

        for path in possible_paths:
            if path.exists():
                wav_path = path
                break

        if wav_path is None:
            print(f"Warning: Cannot find audio for {filename}")
            continue

        # 提取梅尔频谱
        try:
            mel = mel_extractor.extract(str(wav_path))
            mel = mel.numpy()
        except Exception as e:
            print(f"Warning: Failed to extract mel for {filename}: {e}")
            continue

        # 文本转音素 ID
        try:
            phoneme_ids = text_to_sequence(text, language="en")
        except Exception as e:
            print(f"Warning: Failed to convert text for {filename}: {e}")
            continue

        # 保存
        base_name = Path(filename).stem
        mel_path = mels_dir / f"{base_name}.npy"
        phoneme_path = phonemes_dir / f"{base_name}.npy"

        np.save(mel_path, mel)
        np.save(phoneme_path, np.array(phoneme_ids, dtype=np.int32))

        # 记录
        processed_items.append((base_name, text))

    # 生成 filelist
    create_filelist_from_items(processed_items, str(filelists_dir))

    print(f"Processed {len(processed_items)} samples")


def create_filelist_from_items(items, output_dir):
    """从处理好的数据生成 filelist

    Args:
        items: List of (base_name, text) tuples
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)

    all_items = []
    for base_name, text in items:
        all_items.append(f"{base_name}.npy|{text}\n")

    # 打乱
    np.random.seed(42)
    indices = np.random.permutation(len(all_items))
    all_items = [all_items[i] for i in indices]

    # 划分训练集和验证集 (5%)
    val_size = max(1, int(len(all_items) * 0.05))
    val_items = all_items[:val_size]
    train_items = all_items[val_size:]

    # 保存
    with open(output_dir / "train.txt", "w", encoding="utf-8") as f:
        f.writelines(train_items)

    with open(output_dir / "val.txt", "w", encoding="utf-8") as f:
        f.writelines(val_items)

    return str(output_dir / "train.txt"), str(output_dir / "val.txt")


def create_filelist(input_path, val_ratio=0.05):
    """从已有 filelist 划分训练集和验证集

    Args:
        input_path: 输入 filelist 路径
        val_ratio: 验证集比例

    Returns:
        train_path, val_path
    """
    input_path = Path(input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        items = [line.strip() for line in f if line.strip()]

    # 打乱
    np.random.seed(42)
    indices = np.random.permutation(len(items))
    items = [items[i] for i in indices]

    # 划分
    val_size = max(1, int(len(items) * val_ratio))
    val_items = items[:val_size]
    train_items = items[val_size:]

    # 保存
    output_dir = input_path.parent
    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_items) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_items) + "\n")

    return str(train_path), str(val_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess LJSpeech dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="LJSpeech data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    preprocess_ljspeech(args.data_dir, args.output_dir)
