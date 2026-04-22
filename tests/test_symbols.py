"""T-018~T-025: 音素符号表测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from text.symbols import PAD, PHONEME_LIST, SYMBOL_TO_ID, ID_TO_SYMBOL


def test_phoneme_list_contains_pad_and_arpabet():
    """T-018: PHONEME_LIST 包含 PAD "_" 和基本英文音素（如 "AA", "AE", "AH" 等 ARPAbet）"""
    assert PAD == "_"
    assert "_" in PHONEME_LIST
    # 检查基本 ARPAbet 音素
    basic_phonemes = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
    for p in basic_phonemes:
        assert p in PHONEME_LIST, f"Missing phoneme: {p}"


def test_symbol_to_id_mapping():
    """T-020: SYMBOL_TO_ID["_"] == 0, ID_TO_SYMBOL[0] == "_" """
    assert SYMBOL_TO_ID["_"] == 0
    assert ID_TO_SYMBOL[0] == "_"


def test_bidirectional_mapping_consistency():
    """SYMBOL_TO_ID 和 ID_TO_SYMBOL 双向映射一致"""
    for symbol, idx in SYMBOL_TO_ID.items():
        assert ID_TO_SYMBOL[idx] == symbol


def test_text_to_sequence_returns_valid_ids():
    """T-022: text_to_sequence("hello", language="en") 返回非空 int 列表，所有值在 [0, vocab_size) 范围内"""
    from text.symbols import text_to_sequence
    from config import TextEncoderConfig

    config = TextEncoderConfig()
    result = text_to_sequence("hello", language="en")
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, int) for x in result)
    assert all(0 <= x < config.vocab_size for x in result)


def test_sequence_to_text_roundtrip():
    """T-024: sequence_to_text(text_to_sequence("hello")) 返回音素字符串"""
    from text.symbols import text_to_sequence, sequence_to_text

    ids = text_to_sequence("hello", language="en")
    text = sequence_to_text(ids)
    assert isinstance(text, str)
    assert len(text) > 0
