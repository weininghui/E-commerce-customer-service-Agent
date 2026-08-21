"""多文档类型 MD 知识入库（P14 多文档知识库 / P24 官方切分器）。

读取 data/ecommerce/md/*.md（front matter blockquote 标注 doc_type），
按 doc_type 档位用官方 ``MarkdownHeaderTextSplitter`` + ``RecursiveCharacterTextSplitter``
切块（见 ``agent_base.ingest.splitter``），打上 doc_type/section/source metadata，
用 ollama bge-m3 向量化写入 Qdrant ecommerce_chunks collection。

用法：uv run python -m agent_base.ingest.markdown
"""

from __future__ import annotations

import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_base.embeddings import build_embeddings
from agent_base.ingest.splitter import split_markdown_by_type
from agent_base.vectorstore import build_vector_store


MD_DIR = Path("data/ecommerce/md")
COLLECTION = "ecommerce_chunks"
QDRANT_URL = "http://localhost:6333"


def parse_md(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """解析单个 MD：front matter + 按 doc_type 档位官方切块（P24a）。

    Args:
        path: MD 文件路径。

    Returns:
        (metadata, chunks) 元组，chunk 为 {text, metadata}，metadata 含 section。
    """
    text = path.read_text(encoding="utf-8")
    meta_line = ""
    m = re.search(r"^> doc_type: (.+)$", text, re.MULTILINE)
    if m:
        meta_line = m.group(1)
    meta: dict[str, Any] = {}
    parts = meta_line.split("|")
    if parts and parts[0].strip():
        # 第一个字段是 doc_type（如 "product_longdoc | brand: ..."）
        meta["doc_type"] = parts[0].strip()
    for part in parts[1:]:
        if ":" in part:
            k, v = part.split(":", 1)
            meta[k.strip()] = v.strip()
    meta.setdefault("doc_type", "general")

    # P24a：官方 MarkdownHeaderTextSplitter 按标题层级切章节，
    # 超长章节由 RecursiveCharacterTextSplitter（中文 separators）递归切分。
    documents = split_markdown_by_type(meta.get("doc_type", ""), text)
    chunks: list[dict[str, Any]] = []
    for doc in documents:
        _flush(doc.page_content, doc.metadata.get("section", "概述"), meta, path, chunks)
    return meta, chunks


def _flush(
    text: str,
    section: str,
    meta: dict[str, Any],
    path: Path,
    chunks: list[dict[str, Any]],
) -> None:
    """把单个 chunk 写入列表（带 doc_type/section 元数据）。"""
    text = text.strip()
    if not text:
        return
    chunk_meta = dict(meta)
    chunk_meta["section"] = section or "概述"
    chunk_meta["source_file"] = path.name
    chunk_meta["chunk_type"] = "md_" + meta.get("doc_type", "general")
    chunks.append({"text": text, "metadata": chunk_meta})


def main() -> None:
    """执行 MD 知识入库：解析 → 切块 → 向量化 → Qdrant。"""
    all_chunks: list[dict[str, Any]] = []
    metas: dict[str, Any] = {}
    for path in sorted(MD_DIR.glob("*.md")):
        meta, chunks = parse_md(path)
        metas[path.name] = meta
        all_chunks.extend(chunks)
        print(f"  {path.name}: doc_type={meta.get('doc_type')} chunks={len(chunks)}")

    if not all_chunks:
        raise RuntimeError(f"无 MD 文档可入库: {MD_DIR}")

    # ── P16b: 给每个 chunk 补顶层 chunk_id + doc_id（与 Qdrant UUID 一致） ──
    for c in all_chunks:
        cid = str(uuid.uuid5(uuid.NAMESPACE_URL, c["metadata"]["source_file"] + c["text"][:40]))
        c["chunk_id"] = cid
        if "chunk_id" not in c["metadata"]:
            c["metadata"]["chunk_id"] = cid
# doc_id = 源文件名（用于 PG 文档分组）
        if "doc_id" not in c["metadata"]:
            c["metadata"]["doc_id"] = c["metadata"].get("source_file", "")

    # 构建向量库实例（写入统一由 ingest_document_from_chunks 完成）
    emb = build_embeddings(provider=os.getenv("EMBEDDING_PROVIDER", "ollama"))
    store = build_vector_store(
        provider="qdrant", collection=COLLECTION, embedding_function=emb, url=QDRANT_URL,
    )

    # ── P16b: documents 落库（PG truth source） ──
    from agent_base.storage.documents import ingest_document_from_chunks

# 按 source_file 分组 chunk（一个 md 文件对应一篇文档）
    by_file: dict[str, list[dict[str, Any]]] = {}
    file_content: dict[str, str] = {}
    for path in sorted(MD_DIR.glob("*.md")):
        key = path.name
        file_content[key] = path.read_text(encoding="utf-8")
        by_file[key] = [c for c in all_chunks if c["metadata"].get("source_file") == key]

    written = 0
    for fname, fchunks in by_file.items():
        if not fchunks:
            continue
        try:
            ingest_document_from_chunks(
                doc_id=fname,
                content=file_content.get(fname, ""),
                chunks=fchunks,
                vector_store=store,
                skip_tag_check=True,  # P19 D1：内部导入器豁免，对外 API 强制
            )
            written += 1
        except Exception as e:
            print(f"  PG documents: {fname} skipped ({e})")
    print(f"PG documents: {written}/{len(by_file)} md files written")


if __name__ == "__main__":
    main()
