"""T-088~T-095: 数据集测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import tempfile


def test_tts_dataset_instantiation(tmp_path):
    """T-088: TTSDataset(filelist_path) 能实例化，len() 返回文件列表行数"""
    from dataset import TTSDataset

    # 创建 mock filelist
    filelist_dir = tmp_path / "filelists"
    filelist_dir.mkdir(parents=True)

    # 创建 5 条数据
    items = [f"{i}.npy|hello\n" for i in range(5)]
    with open(filelist_dir / "train.txt", "w") as f:
        f.writelines(items)

    dataset = TTSDataset(str(filelist_dir / "train.txt"), str(tmp_path / "data"))
    assert len(dataset) == 5


def test_tts_dataset_getitem(tmp_path):
    """T-090: __getitem__(0) 返回 dict 包含 phoneme_ids (1D Tensor)、mel (2D Tensor, 80×M)、duration (1D Tensor)"""
    from dataset import TTSDataset

    # 创建 mock 数据目录
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    # 先创建子目录
    os.makedirs(data_dir / "mels", exist_ok=True)
    os.makedirs(data_dir / "phonemes", exist_ok=True)

    # 创建 mock mel 和 phoneme 文件
    mel = np.random.randn(80, 50).astype(np.float32)
    phoneme_ids = np.array([10, 20, 30, 40], dtype=np.int32)

    np.save(data_dir / "mels" / "0.npy", mel)
    np.save(data_dir / "phonemes" / "0.npy", phoneme_ids)

    # 创建 filelist
    filelist_dir = tmp_path / "filelists"
    filelist_dir.mkdir(parents=True)
    with open(filelist_dir / "train.txt", "w") as f:
        f.write("0.npy|hello\n")

    dataset = TTSDataset(str(filelist_dir / "train.txt"), str(data_dir))

    item = dataset[0]

    assert "phoneme_ids" in item
    assert "mel" in item
    assert isinstance(item["phoneme_ids"], torch.Tensor)
    assert item["phoneme_ids"].dim() == 1
    assert item["mel"].shape[0] == 80


def test_collate_fn(tmp_path):
    """T-092: collate_fn(batch) 将 3 条不同长度样本 padding 为统一 batch"""
    from dataset import collate_fn

    # 创建 3 个不同长度的样本
    batch = [
        {"phoneme_ids": torch.tensor([1, 2, 3]), "mel": torch.randn(80, 30)},
        {"phoneme_ids": torch.tensor([4, 5]), "mel": torch.randn(80, 20)},
        {"phoneme_ids": torch.tensor([6, 7, 8, 9]), "mel": torch.randn(80, 40)},
    ]

    result = collate_fn(batch)

    assert result["phoneme_ids"].shape[0] == 3
    assert result["mel"].shape[0] == 3
    assert result["mel"].shape[1] == 80
    # 最大长度是 4
    assert result["phoneme_ids"].shape[1] == 4


def test_dataloader_integration(tmp_path):
    """T-094: DataLoader 能正常迭代一个 batch"""
    from dataset import TTSDataset, collate_fn
    from torch.utils.data import DataLoader

    # 创建 mock 数据
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    os.makedirs(data_dir / "mels", exist_ok=True)
    os.makedirs(data_dir / "phonemes", exist_ok=True)

    filelist_dir = tmp_path / "filelists"
    filelist_dir.mkdir(parents=True)

    # 创建 3 条数据
    items = []
    for i in range(3):
        mel = np.random.randn(80, 30 + i * 5).astype(np.float32)
        phoneme_ids = np.array([i + 1, i + 2, i + 3], dtype=np.int32)
        np.save(data_dir / "mels" / f"{i}.npy", mel)
        np.save(data_dir / "phonemes" / f"{i}.npy", phoneme_ids)
        items.append(f"{i}.npy|hello{i}\n")

    with open(filelist_dir / "train.txt", "w") as f:
        f.writelines(items)

    dataset = TTSDataset(str(filelist_dir / "train.txt"), str(data_dir))
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    # 迭代一个 batch
    batch = next(iter(dataloader))
    assert batch["phoneme_ids"].shape[0] == 2
    assert batch["mel"].shape[0] == 2
