# -*- coding: utf-8 -*-
"""
信息抽取模块：抽取来自网页内容

Author: wjianxz
Date: 2025-11-13
"""
import re


def clean_crawled_text(text: str) -> str:
    """
    清洗爬取的网页文本，去除 HTML 标签、脚本、样式、注释，
    并过滤明显无意义的英文乱码或填充文本。
    """
    if not isinstance(text, str):
        return ""

    # 1. 去除 HTML 注释
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # 2. 去除 script 和 style 标签及其内容
    text = re.sub(
        r"<(script|style).*?>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE
    )

    # 3. 去除所有 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)

    # 4. 初步清理空白
    text = re.sub(r"\s+", " ", text).strip()

    # 5. 移除明显无效的英文乱码（逐词判断）
    def is_meaningless_word(word: str) -> bool:
        word = word.strip(".,;:!?\"'()[]{}")
        if not word or len(word) < 2:
            return True
        # 全为非字母数字（纯符号）
        if not re.search(r"[a-zA-Z0-9]", word):
            return True
        # 纯数字（可选保留，这里暂时保留）
        if word.isdigit():
            return False
        # 纯英文字母
        if word.isalpha() and word.islower():
            # 常见有效短词白名单（可扩展）
            valid_short_words = {
                "epa",
                "who",
                "cdc",
                "ai",
                "tv",
                "us",
                "uk",
                "eu",
                "ceo",
                "gdp",
                "covid",
            }
            if len(word) <= 2 or (len(word) <= 4 and word not in valid_short_words):
                # 检查是否为重复字符（如 aaaa, zzz）
                if len(set(word.lower())) == 1:
                    return True
                # 检查是否为键盘序列（简单规则）
                keyboard_seqs = ["qwerty", "asdf", "zxcv", "qwer", "abcd"]
                if any(seq in word.lower() for seq in keyboard_seqs):
                    return True
                # 默认认为短小全小写英文无意义（可根据场景调整）
                return True
        # 包含数字+字母但像乱码（如 x123y, abc123def 通常有效，不删）
        return False

    # 按空格分词，过滤无意义词
    words = text.split()
    filtered_words = []
    for w in words:
        if not is_meaningless_word(w):
            filtered_words.append(w)

    text = " ".join(filtered_words)

    # 再次清理多余空白
    text = re.sub(r"\s+", " ", text).strip()

    return text
