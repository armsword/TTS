"""TTS 数据集模块"""
import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


class TTSDataset(torch.utils.data.Dataset):
    """TTS 数据集

    从 filelist 读取数据，返回 (phoneme_ids, mel, duration) 元组
    """

    def __init__(self, filelist_path: str, data_dir: str):
        """初始化数据集

        Args:
            filelist_path: filelist 路径，每行格式: "filename.npy|text"
            data_dir: 数据目录，包含 mels/ 和 phonemes/ 子目录
        """
        self.data_dir = Path(data_dir)
        self.mels_dir = self.data_dir / "mels"
        self.phonemes_dir = self.data_dir / "phonemes"

        # 读取 filelist
        with open(filelist_path, "r", encoding="utf-8") as f:
            self.items = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个样本

        Args:
            idx: 样本索引

        Returns:
            dict 包含:
                - phoneme_ids: 音素 ID 序列 (time,)
                - mel: 梅尔频谱 (n_mels, time)
                - duration: 时长（从 mel 长度计算）
        """
        line = self.items[idx]
        parts = line.split("|")
        filename = parts[0]
        text = parts[1] if len(parts) > 1 else ""

        # 加载梅尔频谱
        mel_path = self.mels_dir / filename
        mel = np.load(mel_path)  # shape: (n_mels, time)

        # 加载音素 ID
        base_name = Path(filename).stem
        phoneme_path = self.phonemes_dir / f"{base_name}.npy"
        phoneme_ids = np.load(phoneme_path)  # shape: (time,)

        # 计算 duration（梅尔时间步数）
        duration = mel.shape[1]

        return {
            "phoneme_ids": torch.from_numpy(phoneme_ids).long(),
            "mel": torch.from_numpy(mel).float(),
            "duration": torch.tensor(duration, dtype=torch.long),
            "text": text,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """将不同长度的样本 padding 到统一长度

    Args:
        batch: 样本列表

    Returns:
        padding 后的 batch
    """
    # 找出最大长度
    max_phoneme_len = max(item["phoneme_ids"].shape[0] for item in batch)
    max_mel_len = max(item["mel"].shape[1] for item in batch)

    batch_size = len(batch)
    n_mels = batch[0]["mel"].shape[0]

    # Padding 后的 tensor
    phoneme_ids_padded = torch.zeros(batch_size, max_phoneme_len, dtype=torch.long)
    mel_padded = torch.zeros(batch_size, n_mels, max_mel_len, dtype=torch.float32)

    # 处理 duration（如果有）
    has_duration = "duration" in batch[0]
    if has_duration:
        durations = torch.zeros(batch_size, dtype=torch.long)

    texts = []

    for i, item in enumerate(batch):
        phoneme_len = item["phoneme_ids"].shape[0]
        mel_len = item["mel"].shape[1]

        phoneme_ids_padded[i, :phoneme_len] = item["phoneme_ids"]
        mel_padded[i, :, :mel_len] = item["mel"]
        if has_duration:
            durations[i] = item["duration"]
        texts.append(item.get("text", ""))

    result = {
        "phoneme_ids": phoneme_ids_padded,
        "mel": mel_padded,
        "texts": texts,
    }

    if has_duration:
        result["duration"] = durations

    return result


def get_dataloader(filelist_path: str, data_dir: str, batch_size: int = 16,
                    shuffle: bool = True, num_workers: int = 0) -> torch.utils.data.DataLoader:
    """创建 DataLoader

    Args:
        filelist_path: filelist 路径
        data_dir: 数据目录
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数

    Returns:
        DataLoader
    """
    dataset = TTSDataset(filelist_path, data_dir)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
