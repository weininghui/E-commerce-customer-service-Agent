"""章节名归一化与正文清洗。

把解析出的各种章节别名（如"商品名称/通用名"）统一到标准章节，
并清理正文中的空白、断行与全角空格，保证分块质量一致。
"""

from __future__ import annotations

import re


SECTION_ALIASES = {
    "产品名称": "产品名称",
    "商品名称": "产品名称",
    "成份": "成分",
    "成分": "成分",
    "性状": "性状",
    "规格": "规格",
    "使用说明": "使用说明",
    "使用方法": "使用说明",
    "注意事项": "注意事项",
    "商品参数": "商品参数",
    "参数": "商品参数",
    "卖点": "卖点",
    "功效": "功效",
    "搭配建议": "搭配建议",
    "穿搭指南": "搭配建议",
    "尺码指南": "尺码指南",
    "面料": "面料",
    "保存方法": "保存方法",
    "贮藏": "保存方法",
    "产地": "产地",
    "售后FAQ": "售后FAQ",
    "退换货": "售后FAQ",
    "物流说明": "售后FAQ",
    "包装": "包装",
    "有效期": "有效期",
    "执行标准": "执行标准",
    "备案编号": "备案编号",
    "生产企业": "生产企业",
}


def compact_text(text: str) -> str:
    """压缩空白：合并连续空格、去除多余空行。

    Args:
        text: 原始文本。

    Returns:
        清洗后的文本。
    """
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_section_name(raw: str) -> str:
    """把原始章节名映射到标准章节名。

    Args:
        raw: 解析出的章节名。

    Returns:
        标准章节名（无别名时原样返回）。
    """
    name = raw.replace("【", "").replace("】", "")
    name = re.sub(r"\s+", "", name)
    return SECTION_ALIASES.get(name, name)


def normalize_body_text(text: str) -> str:
    """归一化正文：统一换行、修复断词、清理行首尾空白。

    Args:
        text: 原始正文。

    Returns:
        归一化后的正文。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\S)-\n(\S)", r"\1\2", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return compact_text(text)
