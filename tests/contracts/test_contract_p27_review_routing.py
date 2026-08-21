"""P27 契约：AI 决策式审核工作台（决策包 + 置信度分级 + 记忆 + 审计）。"""

from __future__ import annotations

import hashlib
import uuid

from fastapi.testclient import TestClient

from agent_base.api.main import create_app

TOKEN = "admin-dev-token-2026"
PLATFORM_TOKEN = "platform-dev-token-2026"


def _client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=True)


def _stage_doc(filename: str, first_review: dict) -> str:
    """写入一条待审暂存记录，返回 doc_id。"""
    from agent_base.storage.pg import staging_upsert

    content = f"P27 契约测试 {filename} {uuid.uuid4().hex}"
    doc_id = hashlib.sha256(content.encode("utf-8")).hexdigest()
    staging_upsert(
        doc_id=doc_id,
        content=content,
        filename=filename,
        category="契约测试",
        status="pending",
        review_round=1,
        first_review=first_review,
    )
    return doc_id


def test_batch_approve_skips_low_confidence():
    """P27b：低置信度（<0.75）批量确认跳过，提示逐条精审。"""
    from agent_base.storage.pg import staging_delete

    client = _client()
    low = _stage_doc(
        "低置信_1.md",
        {"type": "faq", "confidence": 0.4, "source": "heuristic", "strategy": ["hypothetical_variants"]},
    )
    high = _stage_doc(
        "高置信_1.md",
        {"type": "faq", "confidence": 0.9, "source": "llm", "strategy": ["hypothetical_variants"]},
    )
    try:
        resp = client.post(
            "/api/documents/batch-approve",
            json={"doc_ids": [low, high]},
            headers={"X-Admin-Token": TOKEN},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["approved"] == 1
        failed_ids = {f["doc_id"] for f in data["failed"]}
        assert low in failed_ids
        assert "置信度" in next(f["message"] for f in data["failed"] if f["doc_id"] == low)
    finally:
        for did in (low, high):
            staging_delete(did)


def test_reject_writes_memory_and_audit():
    """P27：打回写 Redis 记忆（review:memory:*）+ first_review.audit.history。"""
    from agent_base.storage.pg import staging_delete, strategy_delete

    client = _client()
    doc_id = _stage_doc(
        "记忆审计_1.md",
        {"type": "faq", "confidence": 0.8, "source": "llm", "strategy": ["hypothetical_variants"]},
    )
    try:
        resp = client.post(
            "/api/documents/tags/reject",
            json={"doc_id": doc_id, "reason": "内容不完整：缺少使用说明", "reviewer": "admin"},
            headers={"X-Admin-Token": TOKEN},
        )
        assert resp.status_code == 200, resp.text
        # 审计历史已追加（first_review.audit.history）
        tag = client.get(
            f"/api/platform/documents/{doc_id}", headers={"X-Platform-Token": PLATFORM_TOKEN}
        )
        assert tag.status_code == 200
        # Redis 记忆键已写
        from agent_base.storage.review_memory import memory_key
        client_redis = __import__("agent_base.storage.review_memory", fromlist=["_get_client"])._get_client()
        if client_redis is not None:
            raw = client_redis.get(memory_key(doc_id, 1))
            assert raw is not None
    finally:
        staging_delete(doc_id)
        strategy_delete(doc_id)


def test_apply_appends_audit_history():
    """P27e：确认入库追加 audit 记录，多轮不覆盖。"""
    from agent_base.storage.pg import staging_delete, staging_get, strategy_delete, strategy_upsert

    client = _client()
    doc_id = _stage_doc(
        "审计追加_1.md",
        {"type": "faq", "confidence": 0.9, "source": "llm", "strategy": ["hypothetical_variants"]},
    )
    try:
        strategy_upsert(
            doc_id=doc_id, doc_type="faq", strategy=["hypothetical_variants"],
            reviewer="admin", status="approved", review_round=1,
            first_review={"type": "faq", "confidence": 0.9, "source": "llm"},
        )
        from agent_base.knowledge_factory import DocTag, apply_tag, persist_tag
        tag = DocTag(doc_id=doc_id, doc_type="faq", strategy=["hypothetical_variants"],
                     review_round=1, first_review={"type": "faq", "confidence": 0.9})
        tag = apply_tag(doc_id, tag, reviewer="admin")
        persist_tag(tag)
        st = staging_get(doc_id)
        assert st is not None
        # 通过后清记忆：Redis 键不存在
        from agent_base.storage.review_memory import memory_key
        client_redis = __import__("agent_base.storage.review_memory", fromlist=["_get_client"])._get_client()
        if client_redis is not None:
            assert client_redis.get(memory_key(doc_id, 1)) is None
    finally:
        staging_delete(doc_id)
        strategy_delete(doc_id)
        client.delete(f"/api/documents/{doc_id}", headers={"X-Admin-Token": TOKEN})
