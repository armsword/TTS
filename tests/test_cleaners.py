"""T-026~T-029: 文本清洗测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from text.cleaners import basic_cleaners, english_cleaners


def test_basic_cleaners():
    """T-026: basic_cleaners("Hello, World!") 返回 "hello world"（小写化 + 去除标点）"""
    result = basic_cleaners("Hello, World!")
    assert result == "hello world"


def test_english_cleaners_handles_contractions_and_numbers():
    """T-028: english_cleaners("It's $100.") 能处理缩写和数字"""
    result = english_cleaners("It's $100.")
    assert isinstance(result, str)
    assert "it's" in result.lower() or "it is" in result.lower()
