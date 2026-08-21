"""P20 知识入库暂存服务（上传=暂存+自动预审 → 精审 approved → 自动入库）。

与 P19 打标状态机的衔接：
  - 上传 → ``document_staging`` pending（即时启发式建议；数据中台推送可携带评估建议）
  - 精审确认（tags/apply）→ document_strategy approved → ``ingest_document``
    真正入库（PG documents → Qdrant → 摘要路由 D3）
  - 打回 → staging returned + reason；重传同内容 → round+1 重新预审
"""

from __future__ import annotations

import hashlib
from typing import Any


def stage_uploaded_document(
    filename: str,
    content: str,
    category: str = "",
    suggestion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """上传文档 → 暂存 + 自动预审 → 进入待审队列。

    Args:
        filename: 上传文件名。
        content: 解析出的文档全文。
        category: 文档分类（可选）。
        suggestion: 数据中台评估建议（可选）——{type, confidence, reasoning, strategy}。
            提供时优先采用（source=data_platform），否则用即时启发式；
            两条路径都不调 LLM（主项目只精审一次）。

    Returns:
        P20 契约 §2.2 响应 dict（action=staged/skipped/staged_updated）。
    """
    from agent_base.knowledge_factory import DocTag, STRATEGY_MAP, pre_review_document
    from agent_base.storage.pg import (
        _conn,
        staging_find_by_content,
        staging_find_by_filename,
        staging_upsert,
    )

    def _build_tag() -> DocTag:
        """数据中台建议优先，否则启发式；绝不调 LLM。"""
        if suggestion:
            s_type = str(suggestion.get("type") or "").strip()
            s_conf = float(suggestion.get("confidence") or 0.8)
            s_reason = str(suggestion.get("reasoning") or "数据中台评估建议")
            s_strategy = (
                list(suggestion.get("strategy") or [])
                if suggestion.get("strategy") is not None
                else None
            )
            first_review: dict[str, Any] = {
                "type": s_type,
                "strategy": s_strategy or list(STRATEGY_MAP.get(s_type, ["default_vector"])),
                "confidence": s_conf,
                "reasoning": s_reason,
                "source": "data_platform",
            }
            return DocTag(
                doc_id="",
                doc_type=s_type,
                strategy=s_strategy or list(STRATEGY_MAP.get(s_type, ["default_vector"])),
                status="pending_fine_review",
                first_review=first_review,
                confidence=s_conf,
                reasoning=s_reason,
            )
        return pre_review_document(
            content[:2000], filename=filename, llm_cfg={"provider": "none"}
        )

    doc_id = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # 去重 1：已入库（documents 真相源）
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT doc_id FROM documents WHERE content=%s AND status='active' "
                "ORDER BY version DESC LIMIT 1",
                (content,),
            )
            row = cur.fetchone()
            dup_doc_id = row[0] if row else None
    except Exception:
        dup_doc_id = None
    if dup_doc_id:
        return {
            "doc_id": dup_doc_id,
            "filename": filename,
            "category": category,
            "status": "skipped",
            "action": "skipped",
            "doc_type": "",
            "confidence": 0.0,
            "message": "相同内容已在知识库中，跳过上传。",
        }

    # 去重 2：暂存区已有同内容 → skipped
    existing = None
    try:
        existing = staging_find_by_content(content)
    except Exception:
        existing = None
    if existing:
        return {
            "doc_id": existing["doc_id"],
            "filename": filename,
            "category": category or existing.get("category", ""),
            "status": existing.get("status", "pending"),
            "action": "skipped",
            "doc_type": existing.get("first_review", {}).get("type", ""),
            "confidence": float(existing.get("first_review", {}).get("confidence", 0.0)),
            "review_round": existing.get("review_round", 1),
            "message": "相同内容已在精审队列中，跳过重复上传。",
        }

    # 同名文件内容不同 → 更新暂存（round 保留；returned 重传 → round+1 重新预审）
    same_name = None
    try:
        same_name = staging_find_by_filename(filename)
    except Exception:
        same_name = None
    if same_name:
        review_round = int(same_name.get("review_round") or 1)
        prev_reason = ""
        if same_name.get("status") == "returned":
            review_round += 1
            # BUG-25: 读上轮短期记忆，注入 prev_reject_reason（精审抽屉展示 + LLM 复核）
            try:
                from agent_base.storage.review_memory import load_memory

                memory = load_memory(same_name["doc_id"], max(1, review_round - 1))
                if memory and memory.get("reject_reason"):
                    prev_reason = memory["reject_reason"]
            except Exception:
                pass
        # 主项目只精审一次：上传路径不再调 LLM 预审（避免阻塞+慢），
        # 仅用即时启发式给出建议类型；LLM 建议在精审抽屉按需触发
        # （数据中台承担上游评估/清洗职责）。
        tag = _build_tag()
        first_review = dict(tag.first_review or {})
        if prev_reason:
            first_review["prev_reject_reason"] = prev_reason
            first_review["memory_round"] = review_round - 1
        staging_upsert(
            doc_id=same_name["doc_id"],
            content=content,
            filename=filename,
            category=category or same_name.get("category", ""),
            status="pending",
            review_round=review_round,
            first_review=first_review,
            reject_reason="",
        )
        return {
            "doc_id": same_name["doc_id"],
            "filename": filename,
            "category": category or same_name.get("category", ""),
            "status": "pending",
            "action": "staged_updated",
            "doc_type": tag.doc_type,
            "confidence": float(tag.first_review.get("confidence", 0.0)),
            "review_round": review_round,
            "message": f"同名文件内容已更新并重新预审（round {review_round}），请确认入库。",
        }

    # 新文档：暂存 + 自动预审
    tag = _build_tag()
    staging_upsert(
        doc_id=doc_id,
        content=content,
        filename=filename,
        category=category,
        status="pending",
        review_round=1,
        first_review=tag.first_review,
    )
    return {
        "doc_id": doc_id,
        "filename": filename,
        "category": category,
        "status": "pending",
        "action": "staged",
        "doc_type": tag.doc_type,
        "confidence": float(tag.first_review.get("confidence", 0.0)),
        "review_round": 1,
        "message": "已进入精审队列，请在 AI 运营台确认入库。",
    }


def approve_and_ingest(
    doc_id: str,
    doc_type: str,
    strategy: list[str] | None,
    reviewer: str,
    vector_store: Any,
    summary_store: Any | None,
) -> dict[str, Any]:
    """精审确认：approved 标签 + 自动入库（PG → Qdrant → 摘要路由 D3）。

    Args:
        doc_id: Staging doc_id（内容 sha256）。
        doc_type: 精审确认的文档类型。
        strategy: 精审确认的索引策略。
        reviewer: 审核人。
        vector_store: 生产向量库（由 API 层注入，避免循环依赖）。
        summary_store: 摘要向量库（可选）。

    Returns:
        P20 契约 §2.3 响应 dict。

    Raises:
        RuntimeError: 暂存文档不存在。
    """
    from agent_base.knowledge_factory import DocTag, apply_tag, persist_tag
    from agent_base.storage.documents import ingest_document
    from agent_base.storage.pg import staging_get

    staging = staging_get(doc_id)
    if staging is None:
        raise RuntimeError(f"暂存文档不存在: {doc_id}")

    review_round = int(staging.get("review_round") or 1)
    category = staging.get("category", "")

    # 1. 打标 approved（document_strategy）
    tag = DocTag(
        doc_id=doc_id,
        doc_type=doc_type,
        strategy=list(strategy or []),
        review_round=review_round,
        first_review=staging.get("first_review") or {},
    )
    tag = apply_tag(doc_id, tag, reviewer=reviewer)
    persist_tag(tag)

    # P27：审核通过 → 清除短期记忆（记忆释放，TTL 之外主动清）
    try:
        from agent_base.storage.review_memory import clear_memory
        clear_memory(doc_id, review_round)
    except Exception:
        pass

    # 2. 自动入库（此时已 approved，精审门放行）
    result = ingest_document(
        doc_id=doc_id,
        content=staging["content"],
        vector_store=vector_store,
        category=category,
        summary_store=summary_store,
        doc_type=doc_type,
        filename=staging.get("filename", ""),
    )
    # 3. 同步暂存状态 → approved：待审队列不再展示，数据中台可查询最终结果
    try:
        from agent_base.storage.pg import staging_upsert
        staging_upsert(
            doc_id=doc_id,
            content=staging["content"],
            filename=staging.get("filename", ""),
            category=category,
            status="approved",
            review_round=review_round,
            first_review=staging.get("first_review") or {},
        )
    except Exception:
        pass
    return {
        "status": "approved",
        "doc_id": doc_id,
        "ingested": True,
        "chunk_count": result.get("chunk_count", 0),
        "review_round": review_round,
        "message": "已确认并入库。",
    }
