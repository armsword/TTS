"""文本清洗模块 - 将原始文本转换为可用于音素化的格式"""
import re
import string


def basic_cleaners(text: str) -> str:
    """基础清洗：小写化 + 去除标点

    Args:
        text: 输入文本

    Returns:
        清洗后的文本
    """
    # 小写化
    text = text.lower()
    # 去除标点（保留空格）
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def english_cleaners(text: str) -> str:
    """英文清洗：处理缩写和数字

    Args:
        text: 输入文本

    Returns:
        清洗后的文本
    """
    # 小写化
    text = text.lower()

    # 处理缩写（简单替换）- 在去除标点之前处理
    contractions = {
        "it's": "it is",
        "i'm": "i am",
        "you're": "you are",
        "he's": "he is",
        "she's": "she is",
        "we're": "we are",
        "they're": "they are",
        "i've": "i have",
        "you've": "you have",
        "we've": "we have",
        "they've": "they have",
        "i'll": "i will",
        "you'll": "you will",
        "he'll": "he will",
        "she'll": "she will",
        "we'll": "we will",
        "they'll": "they will",
        "i'd": "i would",
        "you'd": "you would",
        "he'd": "he would",
        "she'd": "she would",
        "we'd": "we would",
        "they'd": "they would",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "won't": "will not",
        "wouldn't": "would not",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "can't": "can not",
        "couldn't": "could not",
        "shouldn't": "should not",
        "mightn't": "might not",
        "mustn't": "must not",
    }

    words = text.split()
    cleaned_words = []
    for word in words:
        lower_word = word.lower()
        if lower_word in contractions:
            cleaned_words.extend(contractions[lower_word].split())
        else:
            cleaned_words.append(word)

    # 处理数字（简化处理：将数字替换为数字的英文单词）
    text = ' '.join(cleaned_words)

    # 去除标点（保留空格）
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 简单数字转换
    def replace_numbers(match):
        num = int(match.group())
        return str(num)  # 简化：保留数字字符串

    text = re.sub(r'\d+', replace_numbers, text)

    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text
