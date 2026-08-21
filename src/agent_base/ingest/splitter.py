"""官方切分器封装（P24）：doc_type → 切分策略档位映射。

P24a 规则层：MD 主路径统一用官方 ``MarkdownHeaderTextSplitter`` 按标题层级
切章节（章节名自动进 metadata），超长章节再用官方
``RecursiveCharacterTextSplitter``（中文 separators）递归切。

设计对齐 Dify 通用模式：分隔符 + 最大长度 + 重叠三参数。
不做 AI 动态规划（ROADMAP P24 明确排除）；参数档位按 doc_type 规则映射，
未知类型回退默认档位，保证行为可复现、可回归。

用法::

    from agent_base.ingest.splitter import split_markdown_by_type, CHUNK_PROFILES

    chunks = split_markdown_by_type("faq", md_text)   # list[Document]
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


# ── doc_type → 切分档位（P24a 规则层白名单） ────────────────────────────────
# mode: section（章节优先）/ flat（纯文本递归）
CHUNK_PROFILES: dict[str, dict[str, Any]] = {
    "faq": {
        "chunk_size": 512,
        "chunk_overlap": 60,
        "separators": ["\n\n", "\n", "Q:", "A:", "。", "；", "，", ""],
        "mode": "section",
    },
    "metadata_doc": {
        "chunk_size": 256,
        "chunk_overlap": 30,
        "separators": ["\n\n", "\n", "|", "。", "；", "，", ""],
        "mode": "section",
    },
    "ingredient": {
        "chunk_size": 256,
        "chunk_overlap": 30,
        "separators": ["\n\n", "\n", "|", "。", "；", "，", ""],
        "mode": "section",
    },
    "material": {
        "chunk_size": 256,
        "chunk_overlap": 30,
        "separators": ["\n\n", "\n", "|", "。", "；", "，", ""],
        "mode": "section",
    },
    "product_detail": {
        "chunk_size": 900,
        "chunk_overlap": 120,
        "separators": ["\n\n", "\n", "。", "；", "，", ""],
        "mode": "section",
    },
    "product_longdoc": {
        "chunk_size": 900,
        "chunk_overlap": 120,
        "separators": ["\n\n", "\n", "。", "；", "，", ""],
        "mode": "section",
    },
    "guide": {
        "chunk_size": 900,
        "chunk_overlap": 120,
        "separators": ["\n\n", "\n", "。", "；", "，", ""],
        "mode": "section",
    },
    "fashion_guide": {
        "chunk_size": 900,
        "chunk_overlap": 120,
        "separators": ["\n\n", "\n", "。", "；", "，", ""],
        "mode": "section",
    },
    "policy": {
        "chunk_size": 512,
        "chunk_overlap": 60,
        "separators": ["\n\n", "\n", "。", "；", "，", ""],
        "mode": "section",
    },
    "origin_cert": {
        "chunk_size": 512,
        "chunk_overlap": 60,
        "separators": ["\n\n", "\n", "。", "；", "，", ""],
        "mode": "section",
    },
}

# 未知 doc_type 的兜底档位
DEFAULT_PROFILE: dict[str, Any] = {
    "chunk_size": 900,
    "chunk_overlap": 120,
    "separators": ["\n\n", "\n", "。", "；", "，", ""],
    "mode": "section",
}


def get_profile(doc_type: str) -> dict[str, Any]:
    """取 doc_type 对应的切分档位（未知类型回退默认）。

    P30：运营可在管理端为某 doc_type 自定义分隔符/块大小/重叠，
    存于 chunk_profile_overrides 表；有覆盖时覆盖优先于代码默认。
    """
    base = dict(CHUNK_PROFILES.get(doc_type or "", DEFAULT_PROFILE))
    try:
        from agent_base.storage.pg import chunk_override_get

        ov = chunk_override_get(doc_type or "")
        if ov:
            if ov.get("chunk_size"):
                base["chunk_size"] = int(ov["chunk_size"])
            if ov.get("chunk_overlap") is not None:
                base["chunk_overlap"] = int(ov["chunk_overlap"])
            if ov.get("separators"):
                base["separators"] = [str(s) for s in ov["separators"]]
    except Exception:
        pass
    return base


def build_md_section_splitter() -> MarkdownHeaderTextSplitter:
    """构造按标题层级切章节的官方 splitter。

    Returns:
        MarkdownHeaderTextSplitter 实例（#/##/### → H1/H2/H3 metadata）。
    """
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")]
    )


def build_recursive_splitter(
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> RecursiveCharacterTextSplitter:
    """构造章节内递归切分器（中文分隔符优先级）。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or ["\n\n", "\n", "。", "；", "，", ""],
    )


def _section_name(meta: dict[str, Any]) -> str:
    """从 MarkdownHeaderTextSplitter metadata 提取章节名（H3 > H2 > H1）。"""
    for key in ("H3", "H2", "H1"):
        if meta.get(key):
            return str(meta[key])
    return "概述"


def split_markdown_by_type(
    doc_type: str,
    text: str,
    strip_front_matter: bool = True,
    profile: dict[str, Any] | None = None,
) -> list[Document]:
    """按 doc_type 档位切分 MD 文本（官方切分器，P24a 主路径）。

    Args:
        doc_type: 文档类型（faq / metadata_doc / product_detail ...）。
        text: 完整 MD 文本（可含 front matter blockquote）。
        strip_front_matter: 是否剥离 ``>`` 开头的 front matter 行。
        profile: 可选自定义档位（预览试验参数，未保存），缺省走 get_profile。

    Returns:
        Document 列表；每个 Document.metadata 含 ``section``（章节名），
        超长章节已按档位递归切分。
    """
    if profile is None:
        profile = get_profile(doc_type)
    if strip_front_matter:
        body_lines = [line for line in text.splitlines() if not line.strip().startswith(">")]
        body = "\n".join(body_lines).strip()
    else:
        body = text.strip()
    if not body:
        return []

    section_splitter = build_md_section_splitter()
    recursive = build_recursive_splitter(
        profile["chunk_size"], profile["chunk_overlap"], profile["separators"]
    )

    documents: list[Document] = []
    for sec in section_splitter.split_text(body):
        section = _section_name(sec.metadata)
        content = sec.page_content.strip()
        if not content:
            continue
        if len(content) <= profile["chunk_size"]:
            # 标题拼回正文（旧版行为）：标题词参与向量化；
            # 同时 section 保留在 metadata 供过滤/展示。
            documents.append(
                Document(
                    page_content=f"{section}\n{content}" if section != "概述" else content,
                    metadata={"section": section},
                )
            )
            continue
        # 超长章节：递归切分，保留章节名
        for piece in recursive.split_text(content):
            piece = piece.strip()
            if piece:
                documents.append(
                    Document(
                        page_content=f"{section}\n{piece}" if section != "概述" else piece,
                        metadata={"section": section},
                    )
                )
    return documents


def split_plain_text_by_type(
    doc_type: str,
    text: str,
) -> list[str]:
    r"""按 doc_type 档位切分纯文本（无标题结构时用，上传/入库路径兜底）。

    段落优先（对齐旧行为与 Dify 通用模式）：``\\n\\n`` 分隔的段落各成一块，
    超长段落才用官方 ``RecursiveCharacterTextSplitter``（中文 separators）递归切分。

    Args:
        doc_type: 文档类型。
        text: 纯文本内容。

    Returns:
        切分后的文本列表。
    """
    profile = get_profile(doc_type)
    recursive = build_recursive_splitter(
        profile["chunk_size"], profile["chunk_overlap"], profile["separators"]
    )
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    if len(paragraphs) <= 1:
        # 单段：整段递归切分（中文分隔符边界）
        return [p for p in (x.strip() for x in recursive.split_text(paragraphs[0])) if p]
    # 多段：段落优先，超长段落递归切
    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= profile["chunk_size"]:
            chunks.append(para)
            continue
        chunks.extend(p for p in (x.strip() for x in recursive.split_text(para)) if p)
    return chunks
