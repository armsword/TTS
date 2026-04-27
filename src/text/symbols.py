"""音素符号表 - 定义 ARPAbet 音素集合及与 ID 的双向映射"""
from typing import List

# PAD 符号用于填充
PAD = "_"

# ARPAbet 音素列表（CMU Pronouncing Dictionary）
PHONEME_LIST = [
    PAD,  # 0: padding

    # 元音 (Monophthongs)
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",

    # 辅音 (Stops)
    "P", "B", "T", "D", "K", "G",

    # 辅音 (Fricatives)
    "F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH",

    # 辅音 (Affricates)
    "CH", "JH",

    # 辅音 (Nasals)
    "M", "N", "NG",

    # 辅音 (Liquids)
    "L", "R",

    # 辅音 (Glides)
    "W", "Y",

    # 标点符号
    "!", "?", ".", ",", ";", ":",
    "-", "'", "\"",

    # 特殊符号
    "[PAD]", "[BOS]", "[EOS]", "[UNK]",
]

# 双向映射
SYMBOL_TO_ID = {symbol: idx for idx, symbol in enumerate(PHONEME_LIST)}
ID_TO_SYMBOL = {idx: symbol for idx, symbol in enumerate(PHONEME_LIST)}


def text_to_sequence(text: str, language: str = "en") -> List[int]:
    """将文本转换为音素 ID 序列

    Args:
        text: 输入文本
        language: 语言代码 ("en" 或 "zh")

    Returns:
        音素 ID 列表
    """
    from text.phonemizer import text_to_phonemes

    phonemes = text_to_phonemes(text, language)
    sequence = [SYMBOL_TO_ID.get(p, SYMBOL_TO_ID[PAD]) for p in phonemes]
    return sequence


def sequence_to_text(sequence: List[int]) -> str:
    """将音素 ID 序列转换回音素字符串

    Args:
        sequence: 音素 ID 列表

    Returns:
        音素字符串（空格分隔）
    """
    return " ".join([ID_TO_SYMBOL[idx] for idx in sequence])
