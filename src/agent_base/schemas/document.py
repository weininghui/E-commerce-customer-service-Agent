"""文档核心数据模型。

定义 PDF 解析产出的三类数据结构：
- SourceSpan：来源定位（文件 + 页码区间）
- ProductDocument：整份文档的元信息
- ProductChunk：分块后的检索单元

这些数据类是解析器、向量化、检索链路之间传递的标准格式。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SourceSpan:
    """来源定位：标记一个文本块来自哪个文件、哪些页码。"""

    source_file: str
    source_path: str
    page_start: int
    page_end: int

    def to_dict(self) -> dict[str, Any]:
        """转为 dict，便于 JSON 序列化。

        Returns:
            包含 source_file/source_path/page_start/page_end 的字典。
        """
        return asdict(self)


@dataclass(slots=True)
class ProductDocument:
    """文档级元数据：记录整份 PDF 的标识、来源与商品摘要。"""

    doc_id: str
    source_file: str
    source_path: str
    pages: int
    product_name: str = "unknown"
    product_spec: str = "unknown"
    approval_number: str = "unknown"
    category: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)



@dataclass(slots=True)
class ProductChunk:
    """分块单元：检索命中的最小粒度，包含文本、章节与来源。"""

    chunk_id: str
    doc_id: str
    text: str
    section: str
    source: SourceSpan
    product_name: str = "unknown"
    product_spec: str = "unknown"
    category: str = "unknown"
    section_index: int = 0
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """转为 JSON 友好的完整字典（含嵌套 source）。"""
        payload = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "section": self.section,
            "source": self.source.to_dict(),
            "product_name": self.product_name,
            "product_spec": self.product_spec,
            "category": self.category,
            "section_index": self.section_index,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }
        return payload

    def flat_metadata(self) -> dict[str, Any]:
        """转为扁平元数据字典，过滤 None 值，用于向量库写入。"""
        metadata = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "section": self.section,
            "product_name": self.product_name,
            "product_spec": self.product_spec,
            "category": self.category,
            "section_index": self.section_index,
            "chunk_index": self.chunk_index,
            **self.source.to_dict(),
            **self.metadata,
        }
        return {k: v for k, v in metadata.items() if v is not None}
