"""契约：待审队列批量操作 + 数据中台推回闭环。

覆盖：
1. 批量打回（batch-reject）：原因必填，状态 returned + 原因入档
2. 数据中台状态查询（GET /api/platform/documents/{doc_id}）：打回原因可见（推回闭环）
3. 批量确认（batch-approve）：按暂存建议入库，暂存状态 → approved，队列不再展示
4. 批量丢弃（batch-discard）：从队列移除，平台查询 404
5. 清空已打回（returned/clear）：删除全部 returned 记录，不可恢复

依赖真实 PG + Qdrant（与 P16 契约一致）；使用独立测试 doc_id，不污染现有数据。
"""

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

    content = f"契约测试内容 {filename} {uuid.uuid4().hex}"
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


def test_batch_reject_then_platform_status():
    """批量打回 → returned + 原因；数据中台可查到原因（推回闭环）。"""
    from agent_base.storage.pg import staging_delete

    client = _client()
    docs = [
        _stage_doc("批量打回_1.md", {"type": "faq", "confidence": 0.7, "source": "heuristic"}),
        _stage_doc("批量打回_2.md", {"type": "faq", "confidence": 0.7, "source": "heuristic"}),
    ]
    try:
        resp = client.post(
            "/api/documents/batch-reject",
            json={"doc_ids": docs, "reason_code": "内容不完整", "reason": "缺少使用说明"},
            headers={"X-Admin-Token": TOKEN},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["rejected"] == 2

        # 数据中台侧可查状态 + 原因
        for did in docs:
            r = client.get(f"/api/platform/documents/{did}", headers={"X-Platform-Token": PLATFORM_TOKEN})
            assert r.status_code == 200, r.text
            data = r.json()
            assert data["status"] == "returned"
            assert "内容不完整" in data["reject_reason"]
            assert "缺少使用说明" in data["reject_reason"]

        # 已打回列表可见
        q = client.get(
            "/api/documents/review-queue?status=returned",
            headers={"X-Admin-Token": TOKEN},
        ).json()
        assert set(docs) <= {x["doc_id"] for x in q["queue"]}
    finally:
        for did in docs:
            staging_delete(did)


def test_batch_approve_and_discard():
    """批量确认入库（暂存 → approved）与批量丢弃（队列移除）。"""
    from agent_base.storage.pg import staging_delete, staging_get

    client = _client()
    doc_id = _stage_doc(
        "批量确认_1.md",
        {
            "type": "product_detail",
            "confidence": 0.8,
            "source": "heuristic",
            "strategy": ["parent_child"],
        },
    )
    doc_discard = _stage_doc("批量丢弃_1.md", {"type": "faq", "confidence": 0.5, "source": "heuristic"})
    try:
        # 批量确认
        resp = client.post(
            "/api/documents/batch-approve",
            json={"doc_ids": [doc_id]},
            headers={"X-Admin-Token": TOKEN},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["approved"] == 1, resp.json()
        # 暂存状态 → approved（队列不再展示）
        st = staging_get(doc_id)
        assert st is not None and st["status"] == "approved"
        q = client.get(
            "/api/documents/review-queue?status=pending_fine_review",
            headers={"X-Admin-Token": TOKEN},
        ).json()
        assert doc_id not in {x["doc_id"] for x in q["queue"]}
        # 平台侧状态 = approved
        pr = client.get(f"/api/platform/documents/{doc_id}", headers={"X-Platform-Token": PLATFORM_TOKEN})
        assert pr.status_code == 200 and pr.json()["status"] == "approved"

        # 批量丢弃
        r = client.post(
            "/api/documents/batch-discard",
            json={"doc_ids": [doc_discard]},
            headers={"X-Admin-Token": TOKEN},
        )
        assert r.status_code == 200 and r.json()["discarded"] == 1
        assert staging_get(doc_discard) is None
        r404 = client.get(f"/api/platform/documents/{doc_discard}", headers={"X-Platform-Token": PLATFORM_TOKEN})
        assert r404.status_code == 404
    finally:
        # 清理入库数据（PG 文档 + Qdrant 向量 + 打标）
        client.delete(f"/api/documents/{doc_id}", headers={"X-Admin-Token": TOKEN})
        staging_delete(doc_id)
        staging_delete(doc_discard)
        from agent_base.storage.pg import strategy_delete
        strategy_delete(doc_id)


def test_batch_reject_requires_reason():
    """批量打回原因必填（模板 + 说明至少其一）。"""
    client = _client()
    resp = client.post(
        "/api/documents/batch-reject",
        json={"doc_ids": ["x"], "reason_code": "", "reason": ""},
        headers={"X-Admin-Token": TOKEN},
    )
    assert resp.status_code == 400


def test_clear_returned_records():
    """清空已打回：returned 暂存 + 打标全部删除，数据中台查询 404。"""
    from agent_base.storage.pg import staging_delete

    client = _client()
    docs = [
        _stage_doc("清空已打回_1.md", {"type": "faq", "confidence": 0.6, "source": "heuristic"}),
        _stage_doc("清空已打回_2.md", {"type": "guide", "confidence": 0.6, "source": "heuristic"}),
    ]
    try:
        # 先打回
        resp = client.post(
            "/api/documents/batch-reject",
            json={"doc_ids": docs, "reason_code": "内容不完整", "reason": "测试清空"},
            headers={"X-Admin-Token": TOKEN},
        )
        assert resp.status_code == 200 and resp.json()["rejected"] == 2
        q = client.get(
            "/api/documents/review-queue?status=returned",
            headers={"X-Admin-Token": TOKEN},
        ).json()
        assert set(docs) <= {x["doc_id"] for x in q["queue"]}

        # 清空
        r = client.post("/api/documents/returned/clear", json={}, headers={"X-Admin-Token": TOKEN})
        assert r.status_code == 200, r.text
        assert r.json()["cleared"] >= 2
        q2 = client.get(
            "/api/documents/review-queue?status=returned",
            headers={"X-Admin-Token": TOKEN},
        ).json()
        assert all(d not in {x["doc_id"] for x in q2["queue"]} for d in docs)
        # 数据中台查询 404（记录已清空）
        for did in docs:
            r404 = client.get(f"/api/platform/documents/{did}", headers={"X-Platform-Token": PLATFORM_TOKEN})
            assert r404.status_code == 404
    finally:
        for did in docs:
            staging_delete(did)


def test_clear_returned_only_strategy_table():
    """清空已打回：仅存在于打标队列（document_strategy）的 returned 也要清掉。"""
    from agent_base.storage.pg import strategy_delete, strategy_upsert

    client = _client()
    doc_id = f"p24strategyonly_{uuid.uuid4().hex[:12]}"
    strategy_upsert(
        doc_id=doc_id,
        doc_type="faq",
        strategy=["default_vector"],
        reviewer="admin",
        status="returned",
        review_round=1,
        first_review={"type": "faq", "confidence": 0.5, "source": "heuristic"},
        reject_reason="测试：仅打标队列记录",
    )
    try:
        # 列表可见（来自 strategy 表）
        q = client.get(
            "/api/documents/review-queue?status=returned",
            headers={"X-Admin-Token": TOKEN},
        ).json()
        assert doc_id in {x["doc_id"] for x in q["queue"]}
        # 清空
        r = client.post("/api/documents/returned/clear", json={}, headers={"X-Admin-Token": TOKEN})
        assert r.status_code == 200, r.text
        assert r.json()["cleared"] >= 1
        q2 = client.get(
            "/api/documents/review-queue?status=returned",
            headers={"X-Admin-Token": TOKEN},
        ).json()
        assert doc_id not in {x["doc_id"] for x in q2["queue"]}
        r404 = client.get(f"/api/platform/documents/{doc_id}", headers={"X-Platform-Token": PLATFORM_TOKEN})
        assert r404.status_code == 404
    finally:
        strategy_delete(doc_id)


def test_clear_returned_partial_selection():
    """清空已打回支持指定 doc_ids（前端多选删除），未选中的保留。"""
    from agent_base.storage.pg import staging_delete

    client = _client()
    docs = [
        _stage_doc("部分删除_1.md", {"type": "faq", "confidence": 0.6, "source": "heuristic"}),
        _stage_doc("部分删除_2.md", {"type": "guide", "confidence": 0.6, "source": "heuristic"}),
    ]
    try:
        resp = client.post(
            "/api/documents/batch-reject",
            json={"doc_ids": docs, "reason_code": "内容不完整", "reason": "测试部分删除"},
            headers={"X-Admin-Token": TOKEN},
        )
        assert resp.status_code == 200 and resp.json()["rejected"] == 2

        # 只删第一篇
        r = client.post(
            "/api/documents/returned/clear",
            json={"doc_ids": [docs[0]]},
            headers={"X-Admin-Token": TOKEN},
        )
        assert r.status_code == 200, r.text
        assert r.json()["cleared"] == 1

        q = client.get(
            "/api/documents/review-queue?status=returned",
            headers={"X-Admin-Token": TOKEN},
        ).json()
        remaining = {x["doc_id"] for x in q["queue"]}
        assert docs[0] not in remaining
        assert docs[1] in remaining
        # 未删除的仍可被数据中台查到
        r2 = client.get(f"/api/platform/documents/{docs[1]}", headers={"X-Platform-Token": PLATFORM_TOKEN})
        assert r2.status_code == 200 and r2.json()["status"] == "returned"
    finally:
        for did in docs:
            staging_delete(did)
