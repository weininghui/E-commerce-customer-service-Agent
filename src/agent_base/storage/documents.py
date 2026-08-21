"""Document landing orchestration service (P16b).

Shared by API endpoints (``api/main.py``) and ingest scripts
(``ingest/ecommerce.py``, ``ingest/markdown.py``).

The core invariant: PG ``documents`` is the truth source (content + chunk_ids);
Qdrant is a projection.  Every write follows:
  1. parse content → chunks
  2. PG doc_upsert (content + chunk_ids)   ← truth
  3. Qdrant add_texts                       ← projection
  4. Qdrant failure → rollback PG + compensate-delete Qdrant

Importers call ``ingest_document_from_chunks()`` when they already have
their own chunk-generation logic (which must not change to preserve the
search baseline).  The API ``POST /api/documents/ingest`` calls
``ingest_document()`` which uses the generic paragraph splitter.
"""

from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from typing import Any

from agent_base.indexing.vector_index import _qdrant_point_id

logger = logging.getLogger(__name__)


def _parse_doc_header(content: str) -> dict[str, str]:
    """解析文档首行 ``> doc_type: X | brand: Y | category: Z``（键序不固定）。

    Args:
        content: 文档全文。

    Returns:
        首行键值对（小写 key → 值）；非 header 行返回空字典。
    """
    first = ""
    for line in (content or "").splitlines():
        if line.strip():
            first = line.strip()
            break
    if not first.startswith(">"):
        return {}
    kv: dict[str, str] = {}
    for part in first.lstrip(">").split("|"):
        m = re.match(r"\s*([a-zA-Z_]+)\s*:\s*(.+?)\s*$", part)
        if m:
            kv[m.group(1).lower()] = m.group(2).strip()
    return kv


@lru_cache(maxsize=1)
def _catalog_products() -> list[dict[str, str]]:
    """从 catalog 表读取商品清单（name/brand/category/price_band），供分类字段推导。"""
    try:
        from agent_base.storage.pg import _conn

        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, name, brand, category, price_band FROM catalog")
            return [
                {
                    "id": str(r[0]),
                    "name": str(r[1] or ""),
                    "brand": str(r[2] or ""),
                    "category": str(r[3] or ""),
                    "price_band": str(r[4] or ""),
                }
                for r in cur.fetchall()
            ]
    except Exception:
        return []


def _extract_multi_tags(content: str, keywords: list[str]) -> list[str]:
    """按关键词从内容提取多值标签（去重、保持顺序）。"""
    tags = [k for k in keywords if k in content]
    return tags


def _build_doc_fields(
    content: str,
    category: str = "",
    doc_type: str = "",
    filename: str = "",
) -> dict[str, Any]:
    """推导全部分类字段（对齐 ecommerce.yaml filter_fields + doc_type）。

    - doc_type / brand / category：优先文档首行 header，回退调用方传入值
    - product_name / price_band：catalog 商品全名出现在正文时补全
    - skin_types / style / season：按内容关键词提取（多值数组，Qdrant 数组匹配）

    Args:
        content: 文档全文。
        category: 调用方传入的类目（header 缺失时兜底）。
        doc_type: 调用方传入的文档类型（header 缺失时兜底）。
        filename: 文件名（商品名出现在文件名时优先作为主商品）。

    Returns:
        分类字段字典（空值不写入）。
    """
    header = _parse_doc_header(content)
    fields: dict[str, Any] = {}
    doc_type = header.get("doc_type") or doc_type
    if doc_type:
        fields["doc_type"] = doc_type
    brand = header.get("brand") or ""
    cat = header.get("category") or category
    if brand:
        fields["brand"] = brand
    if cat:
        fields["category"] = cat

    # catalog 匹配：优先文件名命中的商品，否则取正文出现次数最多的商品
    candidates = [p for p in _catalog_products() if p["name"] and p["name"] in content]
    product = next((p for p in candidates if p["name"] and p["name"] in (filename or "")), None)
    if product is None and candidates:
        product = max(candidates, key=lambda p: content.count(p["name"]))
    if product:
        fields["product_name"] = product["name"]
        if not fields.get("brand"):
            fields["brand"] = product["brand"]
        if not fields.get("category"):
            fields["category"] = product["category"]
        if product["price_band"]:
            fields["price_band"] = product["price_band"]

    skin_types = _extract_multi_tags(content, ["全肤质", "油皮", "干皮", "敏感肌", "混合皮", "中性皮", "痘痘肌"])
    if skin_types:
        fields["skin_types"] = skin_types
    style = _extract_multi_tags(content, ["基础百搭", "通勤", "休闲", "法式", "复古", "温柔", "显瘦", "慵懒", "浪漫"])
    if style:
        fields["style"] = style
    season = _extract_multi_tags(content, ["春季", "夏季", "秋季", "冬季", "春夏", "秋冬", "春秋", "四季", "夏天", "冬天"])
    if season:
        fields["season"] = season
    return fields


class TagNotApprovedError(RuntimeError):
    """P19: 文档未通过精审（无标签/pending/returned），禁止入库。"""


# ── 公开 API ────────────────────────────────────────────────────────────────


def ingest_document_from_chunks(
    doc_id: str,
    content: str,
    chunks: list[dict[str, Any]],
    vector_store: Any,
    category: str = "",
    metadata: dict[str, Any] | None = None,
    summary_store: Any | None = None,
    skip_tag_check: bool = False,
) -> dict[str, Any]:
    """Land a document whose chunks were already generated by the caller.

    **Chunk ids / texts / metadata must not change** — this preserves the
    existing 143-chunk search baseline (src_hit = 0.76).

    When ``summary_store`` is provided and ``summary_index.enabled``,
    per-chunk LLM summaries are generated and upserted into the summary
    collection, with stale summaries deleted via diff-sync.

    Args:
        doc_id: Unique document identifier (product ID / FAQ ID / md filename).
        content: Full original text stored in PG as truth source.
        chunks: List of ``{chunk_id, text, metadata}`` dicts.
        vector_store: Qdrant / Chroma vector store instance.
        category: Optional category label.
        metadata: Optional extra PG metadata blob.
        summary_store: Optional Qdrant summary vector store (enables P18 summary sync).
        skip_tag_check: True 时跳过 approved 标签检查（内部导入器豁免，D1）。

    Returns:
        ``{"status": "ingested", "doc_id": ..., "version": ..., "chunk_count": ...}``

    Raises:
        RuntimeError: PG rollback happened (vector write failed).
    """
    from agent_base.storage.pg import doc_upsert, doc_versions
    from agent_base.storage.cache import invalidate_pattern

    # P19 D1: 硬约束——对外入库必须 approved；内部导入器显式豁免
    if not skip_tag_check:
        _ensure_approved(doc_id)

    # v0.27.3：chunk_id 去重——相同段落（sha256 相同）只索引一次，
    # 避免 PG chunk_ids 数组出现重复导致与 Qdrant 唯一点数不一致
    unique_chunks: dict[str, dict[str, Any]] = {}
    for c in chunks:
        unique_chunks.setdefault(c["chunk_id"], c)
    chunks = list(unique_chunks.values())
    chunk_ids = [c["chunk_id"] for c in chunks]

    # v0.27.3：入库前读旧版本 chunk_ids，写新向量后按差集删旧，
    # 保证 ingest/导入器对已存在 doc_id 不残留孤儿向量
    old_chunk_ids: list[str] = []
    for v in doc_versions(doc_id):
        old_chunk_ids.extend(v.get("chunk_ids", []))

# 1. 先写 PG（真相源）
    new_version = doc_upsert(doc_id, chunk_ids, metadata=metadata or {"category": category}, content=content)

# 2. 再写 Qdrant 投影
    texts = [c["text"] for c in chunks]
    metas = [c.get("metadata", {}) for c in chunks]
    point_ids = [_qdrant_point_id(cid) for cid in chunk_ids]

    try:
        vector_store.add_texts(texts=texts, metadatas=metas, ids=point_ids)
    except Exception as exc:
# 补偿：删除可能已部分写入的向量
        _compensate_delete(vector_store, point_ids)
# 回滚 PG
        _rollback_pg(doc_id, new_version)
        logger.error("Vector write failed for %s v%d, PG rolled back: %s", doc_id, new_version, exc)
        raise RuntimeError(f"Vector write failed, PG rolled back: {exc}") from exc

    # 3. 删旧向量（差集：仅删不在新版本中的旧 chunk，避免误删同 ID 新向量）
    new_id_set = set(chunk_ids)
    old_to_delete = [cid for cid in old_chunk_ids if cid not in new_id_set]
    if old_to_delete:
        _compensate_delete(vector_store, [_qdrant_point_id(cid) for cid in old_to_delete])

    # 4. P19 D3: 按 approved 标签 strategy 路由附加索引（摘要/父子等）
    _route_indexing_by_tag(
        doc_id=doc_id,
        content=content,
        chunks=chunks,
        old_chunk_ids=old_chunk_ids,
        summary_store=summary_store,
    )

    invalidate_pattern("rag:cache:*")
    return {
        "status": "ingested",
        "doc_id": doc_id,
        "version": new_version,
        "chunk_count": len(chunk_ids),
        "chunk_ids": chunk_ids,
    }


def ingest_document(
    doc_id: str,
    content: str,
    vector_store: Any,
    category: str = "",
    summary_store: Any | None = None,
    skip_tag_check: bool = False,
    doc_type: str = "",
    filename: str = "",
) -> dict[str, Any]:
    """Land a document using the generic paragraph splitter.

    For imports that already have their own chunk logic (products / FAQ / md),
    prefer ``ingest_document_from_chunks()`` to preserve the chunk baseline.
    P24a: 上传/精审入库路径按 doc_type 档位用官方切分器
    （``agent_base.ingest.splitter``），替代旧的 400/200 自研滑窗。

    Args:
        doc_id: Unique document identifier.
        content: Full document text.
        vector_store: Qdrant / Chroma vector store instance.
        category: Optional category label.
        summary_store: Optional Qdrant summary vector store (P18 summary sync).
        skip_tag_check: True 时跳过 approved 标签检查（内部导入器豁免，D1）。
        doc_type: 精审确认的文档类型（决定切分档位；空串走默认档位）。
        filename: 原始文件名（存入 metadata.doc_name，文档管理展示用）。

    Returns:
        ``{"status": "ingested", "doc_id": ..., "version": ..., "chunk_count": ...}``
    """
    # 全部分类字段（doc_type/brand/category/product_name/price_band/skin_types/style/season）
    # 推导后并入每个 chunk 的 metadata，供检索元数据过滤使用
    chunks = _parse_content_to_chunks(
        doc_id,
        content,
        category,
        doc_type=doc_type,
        filename=filename,
        extra_metadata=_build_doc_fields(content, category=category, doc_type=doc_type, filename=filename),
    )
    metadata: dict[str, Any] = {"category": category}
    if filename:
        metadata["doc_name"] = filename
        # P27：把文档名同步进每个 chunk 的 metadata，Qdrant 点直接携带
        # doc_name，来源卡展示文档名而不再暴露 api://{hash} 内部 ID。
        for c in chunks:
            c["metadata"]["doc_name"] = filename
    return ingest_document_from_chunks(
        doc_id=doc_id,
        content=content,
        chunks=chunks,
        vector_store=vector_store,
        category=category,
        metadata=metadata,
        summary_store=summary_store,
        skip_tag_check=skip_tag_check,
    )


def _ensure_approved(doc_id: str) -> None:
    """P19 D1: 非 approved 标签（含无标签/pending/returned）禁止入库。

    Args:
        doc_id: Document ID.

    Raises:
        RuntimeError: 文档未通过精审。
    """
    tag = None
    try:
        from agent_base.knowledge_factory import get_tag
        tag = get_tag(doc_id)
    except Exception:
        tag = None
    if tag is None or tag.status != "approved":
        status = tag.status if tag is not None else "无标签"
        raise TagNotApprovedError(
            f"文档 {doc_id} 未通过精审（status={status}），禁止入库；"
            "请先在管理端完成打标精审"
        )


def sync_document_summaries(
    doc_id: str,
    chunks: list[dict[str, Any]],
    summary_store: Any,
) -> None:
    """P18：为文档重新生成摘要并 diff 同步摘要存储。

    API 更新/恢复端点的公开入口（不经过 ``ingest_document_from_chunks``）；
    自行从 PG 版本读取旧 chunk ids，调用方只需 doc_id + 新 chunks。
    未启用摘要索引时为 no-op。
    ``retrieval.summary_index.enabled`` is true.

    Args:
        doc_id: Document identifier.
        chunks: Current chunks after update/restore.
        summary_store: Qdrant summary vector store.
    """
    from agent_base.storage.pg import doc_versions

    old_chunk_ids: list[str] = []
    for v in doc_versions(doc_id):
        old_chunk_ids.extend(v.get("chunk_ids", []))
    _sync_summaries(doc_id, chunks, old_chunk_ids, summary_store)


# ── 内部辅助函数 ────────────────────────────────────────────────────────────


def _parse_content_to_chunks(
    doc_id: str,
    content: str,
    category: str = "",
    doc_type: str = "",
    filename: str = "",
    extra_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """按 doc_type 档位切分文档（P24a 官方切分器，共享 API ingest / 精审入库）。

    MD 内容走官方 ``MarkdownHeaderTextSplitter`` + ``RecursiveCharacterTextSplitter``
    （中文 separators）；纯文本走官方递归切分。chunk ids 为 sha256 确定性生成：
    同内容 → 同 id（幂等），内容变更 → 新 id（更新差集安全）。

    Args:
        doc_id: Document identifier.
        content: Full document text.
        category: Optional category tag.
        doc_type: Document type（决定切分档位）。
        filename: 原始文件名（并入 chunk metadata，来源卡展示用）。
        extra_metadata: 额外分类字段（doc_type/brand/category/price_band/skin_types 等），
            合并进每个 chunk 的 metadata。

    Returns:
        List of ``{chunk_id, text, metadata}`` dicts.
    """
    from agent_base.ingest.splitter import split_markdown_by_type, split_plain_text_by_type

    stripped = content.strip()
    if not stripped:
        return []
    # 含 Markdown 标题（##/###）走章节切分；否则按纯文本档位切分
    # P30: 切分时保留真实章节名（splitter 已把 H3>H2>H1 提取到 metadata.section），
    # 来源卡显示真实章节而不是 category 占位（"上传文档"）。
    if "##" in stripped or "# " in stripped:
        documents = split_markdown_by_type(doc_type, content)
        paragraphs = [d.page_content for d in documents]
        sections = [str(d.metadata.get("section") or "") for d in documents]
    else:
        paragraphs = split_plain_text_by_type(doc_type, stripped)
        sections = [""] * len(paragraphs)

    chunks: list[dict[str, Any]] = []
    for idx, para in enumerate(paragraphs):
        # v0.27.3：完整 64 hex，避免 64bit 截断在海量 chunk 下的生日碰撞
        digest = hashlib.sha256(para.encode()).hexdigest()
        chunk_id = _qdrant_point_id(f"{doc_id}:chunk:{digest}")
        meta: dict[str, Any] = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "section": sections[idx] if idx < len(sections) and sections[idx] else (category or ""),
            "source_file": f"api://{doc_id}",
            "chunk_type": "ingested",
        }
        if filename:
            meta["doc_name"] = filename
        if extra_metadata:
            meta.update(extra_metadata)
        chunks.append({
            "chunk_id": chunk_id,
            "text": para,
            "metadata": meta,
        })
    return chunks


def _sync_summaries(
    doc_id: str,
    chunks: list[dict[str, Any]],
    old_chunk_ids: list[str],
    summary_store: Any,
) -> None:
    """P18：文档更新后同步各 chunk 摘要。

    为当前 chunks 生成新摘要并 upsert 进 summary_store，
    再删除旧摘要点（diff-sync，与 dense 向量同逻辑）。

    Args:
        doc_id: Document identifier.
        chunks: Current chunks after update.
        old_chunk_ids: All chunk_ids from previous versions.
        summary_store: Qdrant summary vector store.
    """
    try:
        enabled = False
        try:
            from agent_base.config import load_yaml, deep_get
            cfg = load_yaml("configs/app.yaml")
            enabled = deep_get(cfg, "retrieval.summary_index.enabled", False)
        except Exception:
            pass
        if not enabled:
            return
    except Exception:
        return

    try:
        from agent_base.retrieval.summarizer import generate_summaries
        summaries = generate_summaries(chunks)
        if not summaries:
            return

# upsert 新摘要
        s_texts = [s["summary"] for s in summaries]
        s_ids = [s["summary_id"] for s in summaries]
        s_metas = [s["metadata"] for s in summaries]
        summary_store.add_texts(texts=s_texts, metadatas=s_metas, ids=s_ids)

# 删除旧摘要点（差集：旧 chunk_ids - 新 chunk_ids）
        new_cids = {c.get("chunk_id", "") for c in chunks}
        old_cids = set(old_chunk_ids)
        stale_cids = old_cids - new_cids
        if stale_cids:
            from agent_base.indexing.vector_index import _qdrant_point_id
            stale_ids = [_qdrant_point_id(f"summary:{cid}") for cid in stale_cids]
            summary_store.delete(ids=stale_ids)
    except Exception:
        logger.warning("Summary sync failed for %s (non-fatal)", doc_id, exc_info=True)


def _route_indexing_by_tag(
    doc_id: str,
    content: str,
    chunks: list[dict[str, Any]],
    old_chunk_ids: list[str],
    summary_store: Any | None,
) -> None:
    """P19 D3: 按 approved 标签的 strategy 路由附加索引。

    strategy 动作（CONTRACT-P19 §8）：
      - ``summary_index`` → LLM 每块摘要（总闸开关 + 标签双条件，复用 P18 差集同步）
      - ``parent_child`` → 父文档 docstore 幂等写入
      - ``self_query`` / ``hypothetical_variants`` → 无 ingest 附加动作：
        self_query 的 metadata 已随 chunk 写入，检索侧用 AttributeInfo 白名单；
        FAQ 变体由导入器在 chunk 生成阶段产出（产品/FAQ 天然覆盖）。

    Args:
        doc_id: Document ID.
        content: Full original text.
        chunks: Current chunks after ingest.
        old_chunk_ids: Chunk ids from previous versions (for summary diff-sync).
        summary_store: Optional Qdrant summary vector store.
    """
    try:
        from agent_base.knowledge_factory import get_tag
        tag = get_tag(doc_id)
    except Exception:
        return
    if tag is None or tag.status != "approved":
        return
    strategy = tag.strategy or []

    try:
        from agent_base.config import deep_get, load_yaml
        cfg = load_yaml("configs/app.yaml")
        retrieval_cfg = cfg.get("retrieval", {})
    except Exception:
        retrieval_cfg = {}

    if "summary_index" in strategy and summary_store is not None:
        if deep_get(retrieval_cfg, "summary_index.enabled", False):
            _sync_summaries(doc_id, chunks, old_chunk_ids, summary_store)

    if "parent_child" in strategy:
        _sync_parent_docstore(doc_id, content, chunks)


def _sync_parent_docstore(
    doc_id: str,
    content: str,
    chunks: list[dict[str, Any]],
) -> None:
    """P19 D3: 为 parent_child 策略写入父文档 docstore 条目（幂等 upsert）。

    父文档锚点取 product_card chunk（缺省用首 chunk），key 与
    ``populate_parent_docstore`` 约定一致（chunk_id → Document）。

    Args:
        doc_id: Document ID.
        content: Full original text (fallback parent text).
        chunks: Current chunks after ingest.
    """
    try:
        from langchain_core.documents import Document

        from agent_base.storage.docstore import PGDocStore, populate_parent_docstore

        anchor = next(
            (c for c in chunks if c.get("metadata", {}).get("chunk_type") == "product_card"),
            chunks[0] if chunks else None,
        )
        if anchor is None:
            return
        parent = Document(
            page_content=content or anchor.get("text", ""),
            metadata={
                "chunk_id": anchor.get("chunk_id", ""),
                "doc_id": doc_id,
            },
        )
        store = PGDocStore(table_name="docstore_parent")
        populate_parent_docstore(store, [parent])
    except Exception:
        logger.warning("Parent docstore sync failed for %s (non-fatal)", doc_id, exc_info=True)


def _compensate_delete(vector_store: Any, point_ids: list[str]) -> None:
    """尽力删除向量（部分写入后的补偿）。"""
    try:
        vector_store.delete(ids=point_ids)
    except Exception:
        pass


def _rollback_pg(doc_id: str, version: int) -> None:
    """删除 PG 中指定版本行（向量失败后的回滚）。"""
    try:
        from agent_base.storage.pg import _conn
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM documents WHERE doc_id=%s AND version=%s",
                (doc_id, version),
            )
    except Exception:
        pass


