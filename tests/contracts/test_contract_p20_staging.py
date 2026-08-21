"""Contract P20: 知识入库暂存（上传=暂存+自动预审 → 精审自动入库）。"""

from __future__ import annotations

import hashlib
import uuid

TOKEN = "admin-dev-token-2026"


def _tag(doc_type: str = "product_detail"):
    from agent_base.knowledge_factory import DocTag
    return DocTag(
        doc_id="", doc_type=doc_type,
        strategy=["summary_index"], status="pending_fine_review",
        first_review={"type": doc_type, "confidence": 0.7, "source": "heuristic"},
    )


def test_stage_new_document(monkeypatch):
    """新文档 → 暂存 pending + 自动预审 + action=staged。"""
    from agent_base.knowledge_factory import pre_review_document as _orig  # noqa: F401
    from agent_base.storage import staging

    upserted: dict = {}
    monkeypatch.setattr("agent_base.knowledge_factory.pre_review_document", lambda *a, **k: _tag())
    monkeypatch.setattr("agent_base.storage.pg._conn", lambda: (_ for _ in ()).throw(RuntimeError("no pg")))
    monkeypatch.setattr("agent_base.storage.pg.staging_find_by_content", lambda content: None)
    monkeypatch.setattr("agent_base.storage.pg.staging_upsert", lambda **kw: upserted.update(kw))

    content = "# 全新成分文档\n\n玻尿酸 5% 浓度说明。"
    result = staging.stage_uploaded_document(filename="成分_新文档.md", content=content, category="知识")
    assert result["action"] == "staged"
    assert result["status"] == "pending"
    assert result["doc_id"] == hashlib.sha256(content.encode()).hexdigest()
    assert result["doc_type"] == "product_detail"
    assert upserted["status"] == "pending"
    assert upserted["review_round"] == 1
    assert upserted["first_review"]["source"] == "heuristic"


def test_stage_skips_duplicate_in_documents(monkeypatch):
    """已入库内容 → action=skipped。"""
    from agent_base.storage import staging

    class FakeCursor:
        def execute(self, sql, params=None):
            return None

        def fetchone(self):
            return ("doc-existing",)

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("agent_base.storage.pg._conn", lambda: FakeConn())
    result = staging.stage_uploaded_document(
        filename="dup.md", content="已入库内容", category="",
    )
    assert result["action"] == "skipped"
    assert result["doc_id"] == "doc-existing"


def test_stage_returned_reupload_round_plus_one(monkeypatch):
    """同名文件内容更新（原 returned）→ round+1 重新预审。"""
    from agent_base.storage import staging

    upserted: dict = {}
    monkeypatch.setattr(
        "agent_base.knowledge_factory.pre_review_document",
        lambda *a, **k: _tag("faq"),
    )
    monkeypatch.setattr("agent_base.storage.pg._conn", lambda: (_ for _ in ()).throw(RuntimeError("no pg")))
    monkeypatch.setattr("agent_base.storage.pg.staging_find_by_content", lambda content: None)
    monkeypatch.setattr(
        "agent_base.storage.pg.staging_find_by_filename",
        lambda filename: {"doc_id": "x", "status": "returned", "review_round": 2, "category": "知识"},
    )
    monkeypatch.setattr("agent_base.storage.pg.staging_upsert", lambda **kw: upserted.update(kw))

    result = staging.stage_uploaded_document(filename="retry.md", content="修改后的内容", category="知识")
    assert result["action"] == "staged_updated"
    assert upserted["review_round"] == 3
    assert upserted["status"] == "pending"
    assert upserted["reject_reason"] == ""


def test_stage_same_content_in_staging_skipped(monkeypatch):
    """暂存区同内容 → action=skipped（不重复进队列）。"""
    from agent_base.storage import staging

    monkeypatch.setattr("agent_base.storage.pg._conn", lambda: (_ for _ in ()).throw(RuntimeError("no pg")))
    monkeypatch.setattr(
        "agent_base.storage.pg.staging_find_by_content",
        lambda content: {
            "doc_id": "x", "filename": "a.md", "category": "", "status": "pending",
            "review_round": 1, "first_review": {"type": "faq", "confidence": 0.7},
        },
    )
    result = staging.stage_uploaded_document(filename="b.md", content="相同内容", category="")
    assert result["action"] == "skipped"
    assert result["doc_id"] == "x"


def test_approve_and_ingest(monkeypatch):
    """精审确认 → approved 标签 + 自动入库。"""
    from agent_base.storage import staging

    monkeypatch.setattr(
        "agent_base.storage.pg.staging_get",
        lambda doc_id: {
            "content": "# 待入库内容", "review_round": 1, "category": "知识",
            "first_review": {"type": "faq", "confidence": 0.7},
        },
    )
    upserted: dict = {}
    monkeypatch.setattr("agent_base.storage.pg.strategy_upsert", lambda **kw: upserted.update(kw))
    from agent_base.knowledge_factory import DocTag
    monkeypatch.setattr(
        "agent_base.knowledge_factory.get_tag",
        lambda doc_id: DocTag(
            doc_id=doc_id, doc_type="faq", strategy=["hypothetical_variants"],
            status="approved",
        ),
    )

    class FakeVS:
        def add_texts(self, texts, metadatas, ids):
            pass

        def delete(self, ids):
            pass

    result = staging.approve_and_ingest(
        doc_id="doc1", doc_type="faq", strategy=["hypothetical_variants"],
        reviewer="alice", vector_store=FakeVS(), summary_store=None,
    )
    assert result["status"] == "approved"
    assert result["ingested"] is True
    assert upserted["status"] == "approved"
    assert upserted["reviewer"] == "alice"


def test_api_upload_stages_without_403(monkeypatch):
    """API 层：上传新文档返回 200 staged（不再 403）。"""
    from fastapi.testclient import TestClient

    from agent_base.api.main import create_app
    from agent_base.storage.pg import staging_delete

    monkeypatch.setattr(
        "agent_base.knowledge_factory.pre_review_document",
        lambda *a, **k: _tag("metadata_doc"),
    )
    client = TestClient(create_app(), raise_server_exceptions=True)
    content = "# 接口上传测试\n\n唯一内容标记 {}".format(uuid.uuid4().hex)
    doc_id = hashlib.sha256(content.encode()).hexdigest()
    try:
        resp = client.post(
            "/api/upload",
            files={"file": ("api_test.md", content.encode("utf-8"), "text/markdown")},
            data={"category": "测试"},
            headers={"X-Admin-Token": TOKEN},
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["action"] == "staged"
        assert payload["doc_id"] == doc_id
        assert payload["status"] == "pending"
    finally:
        staging_delete(doc_id)
