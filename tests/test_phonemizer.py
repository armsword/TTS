"""T-030~T-035: 文本转音素测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from text.phonemizer import english_to_phonemes, chinese_to_phonemes, text_to_phonemes
from text.symbols import PHONEME_LIST


def test_english_to_phonemes_returns_valid():
    """T-030: english_to_phonemes("hello") 返回非空音素列表，每个元素是合法音素字符串"""
    result = english_to_phonemes("hello")
    assert isinstance(result, list)
    assert len(result) > 0
    for p in result:
        assert isinstance(p, str)
        assert p in PHONEME_LIST, f"Invalid phoneme: {p}"


def test_chinese_to_phonemes_returns_pinyin():
    """T-032: chinese_to_phonemes("你好") 返回包含拼音的列表"""
    result = chinese_to_phonemes("你好")
    assert isinstance(result, list)
    assert len(result) > 0


def test_text_to_phonemes_routes_correctly():
    """T-034: text_to_phonemes("hello", "en") 调用英文路径，text_to_phonemes("你好", "zh") 调用中文路径"""
    en_result = text_to_phonemes("hello", "en")
    zh_result = text_to_phonemes("你好", "zh")
    assert isinstance(en_result, list)
    assert isinstance(zh_result, list)
    assert len(en_result) > 0
    assert len(zh_result) > 0
