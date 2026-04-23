"""T-082~T-087: 数据预处理测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import json
import tempfile


def test_preprocess_ljspeech_structure(tmp_path):
    """T-082: preprocess_ljspeech() 给定一个含 3 条样本的 mock 数据目录，生成 mels/、phonemes/、filelists/ 目录"""
    from preprocess import preprocess_ljspeech

    # 创建 mock 数据目录
    data_dir = tmp_path / "mock_ljspeech"
    data_dir.mkdir()

    # 创建 mock metadata.csv
    metadata = """filename|text
1.wav|hello world
2.wav|hello there
3.wav|good morning
"""
    with open(data_dir / "metadata.csv", "w") as f:
        f.write(metadata)

    # 创建 mock wav 文件
    import scipy.io.wavfile as wavfile
    for i in range(1, 4):
        sample_rate = 22050
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        wavfile.write(data_dir / f"{i}.wav", sample_rate, waveform)

    # 运行预处理
    preprocess_ljspeech(str(data_dir), str(tmp_path / "output"))

    # 验证输出目录
    output_dir = tmp_path / "output"
    assert (output_dir / "mels").is_dir()
    assert (output_dir / "phonemes").is_dir()
    assert (output_dir / "filelists").is_dir()


def test_mel_file_shape(tmp_path):
    """T-084: 生成的 mel .npy 文件 shape 为 (80, T)"""
    from preprocess import preprocess_ljspeech

    data_dir = tmp_path / "mock_ljspeech"
    data_dir.mkdir()

    metadata = """filename|text
1.wav|hello
"""
    with open(data_dir / "metadata.csv", "w") as f:
        f.write(metadata)

    import scipy.io.wavfile as wavfile
    sample_rate = 22050
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration))
    waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    wavfile.write(data_dir / "1.wav", sample_rate, waveform)

    preprocess_ljspeech(str(data_dir), str(tmp_path / "output"))

    # 检查 mel 文件
    mel = np.load(tmp_path / "output" / "mels" / "1.npy")
    assert mel.shape[0] == 80  # n_mels


def test_filelist_split(tmp_path):
    """T-086: train.txt 和 val.txt 文件正确划分，val 比例约 5%"""
    from preprocess import create_filelist

    # 创建 mock filelist
    filelist_dir = tmp_path / "filelists"
    filelist_dir.mkdir()

    # 创建 20 条数据
    items = [f"{i}.npy|hello\n" for i in range(20)]
    with open(filelist_dir / "all.txt", "w") as f:
        f.writelines(items)

    # 划分
    train_path, val_path = create_filelist(str(filelist_dir / "all.txt"), val_ratio=0.05)

    with open(train_path) as f:
        train_lines = len(f.readlines())

    with open(val_path) as f:
        val_lines = len(f.readlines())

    # 验证大约 5% 是验证集
    assert val_lines >= 1  # 至少 1 条
    assert train_lines + val_lines == 20
