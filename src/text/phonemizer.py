"""文本转音素模块 - 将文本转换为音素序列"""
from typing import List
from text.symbols import PHONEME_LIST
from text.cleaners import english_cleaners


def english_to_phonemes(text: str) -> List[str]:
    """将英文文本转换为 ARPAbet 音素列表

    使用 gruut 库进行音素化

    Args:
        text: 输入英文文本

    Returns:
        音素字符串列表
    """
    try:
        from gruut import sentences as gruut_sentences

        # 清洗文本
        cleaned = english_cleaners(text)

        phonemes = []
        for sentence in gruut_sentences(cleaned, lang="en"):
            for word in sentence:
                if word.phonemes:
                    for phoneme in word.phonemes:
                        # gruut 返回的是 IPA 格式，需要转换为 ARPAbet
                        # 这里简化处理，直接使用映射
                        arpabet = ipa_to_arpabet(phoneme)
                        if arpabet:
                            phonemes.append(arpabet)

        if phonemes:
            return phonemes

    except ImportError:
        pass

    # 回退方案：简单音素映射
    return fallback_english_phonemes(text)


def ipa_to_arpabet(ipa: str) -> str:
    """IPA 到 ARPAbet 的简单映射"""
    # 简化映射表
    ipa_map = {
        'ə': 'AH', 'æ': 'AE', 'ɑ': 'AA', 'ɔ': 'AO', 'ʊ': 'UH',
        'u': 'UW', 'ɪ': 'IH', 'i': 'IY', 'ɛ': 'EH', 'eɪ': 'EY',
        'aɪ': 'AY', 'oʊ': 'OW', 'aʊ': 'AW', 'ɔɪ': 'OY',
        'ɜ': 'ER', 'ʌ': 'AH', 'n': 'N', 't': 'T', 's': 'S',
        'l': 'L', 'r': 'R', 'd': 'D', 'm': 'M', 'k': 'K',
        'h': 'HH', 'w': 'W', 'v': 'V', 'f': 'F', 'z': 'Z',
        'b': 'B', 'p': 'P', 'g': 'G', 'ŋ': 'NG', 'θ': 'TH',
        'ð': 'DH', 'ʃ': 'SH', 'ʒ': 'ZH', 'tʃ': 'CH', 'dʒ': 'JH',
        'j': 'Y',
    }
    return ipa_map.get(ipa, '')


def fallback_english_phonemes(text: str) -> List[str]:
    """简单的英文音素回退方案"""
    # 简化处理：基于字母的简单映射
    letter_to_phoneme = {
        'a': 'AE', 'b': 'B', 'c': 'K', 'd': 'D', 'e': 'EH',
        'f': 'F', 'g': 'G', 'h': 'HH', 'i': 'IH', 'j': 'JH',
        'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'o': 'AO',
        'p': 'P', 'q': 'K', 'r': 'R', 's': 'S', 't': 'T',
        'u': 'AH', 'v': 'V', 'w': 'W', 'x': 'K', 'y': 'Y',
        'z': 'Z',
    }

    cleaned = english_cleaners(text).lower()
    phonemes = []
    for char in cleaned:
        if char in letter_to_phoneme:
            phonemes.append(letter_to_phoneme[char])
        elif char.isalpha():
            phonemes.append('AH')  # 默认元音
        # 跳过空格和非字母字符

    return phonemes if phonemes else ['AH']


def chinese_to_phonemes(text: str) -> List[str]:
    """将中文文本转换为拼音列表

    使用 pypinyin 库进行拼音转换

    Args:
        text: 输入中文文本

    Returns:
        拼音字符串列表
    """
    try:
        from pypinyin import pinyin
        from pypinyin.style import ToneConverter

        # 获取不带声调的拼音
        pinyins = pinyin(text, style=0)  # 0 = 普通拼音风格
        result = [p[0] for p in pinyins if p[0]]

        if result:
            return result

    except ImportError:
        pass

    # 回退方案：简单按字符分割
    return list(text)


def text_to_phonemes(text: str, language: str = "en") -> List[str]:
    """统一的文本转音素入口

    Args:
        text: 输入文本
        language: 语言代码 ("en" 或 "zh")

    Returns:
        音素字符串列表
    """
    if language == "en":
        return english_to_phonemes(text)
    elif language == "zh":
        return chinese_to_phonemes(text)
    else:
        # 默认使用英文
        return english_to_phonemes(text)
