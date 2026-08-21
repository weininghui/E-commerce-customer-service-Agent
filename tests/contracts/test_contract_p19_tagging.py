"""Contract P19: 打标精审状态机 + 入库硬约束（D1/D2/D5）。"""

from __future__ import annotations

import uuid

import pytest

TOKEN = "admin-dev-token-2026"


def test_pre_review_heuristic_faq():
    """启发式预审：FAQ 信号 → faq + 默认 strategy，状态 pending_fine_review。"""
    from agent_base.knowledge_factory import pre_review_document

    tag = pre_review_document(
        "Q: 支持七天无理由退货吗？\nA: 支持，签收后7天内未拆封可退",
        llm_cfg={"provider": "none"},
    )
    assert tag.status == "pending_fine_review"
    assert tag.doc_type == "faq"
    assert "hypothetical_variants" in tag.strategy
    assert tag.first_review["source"] == "heuristic"


def test_pre_review_llm_adopted_when_confident(monkeypatch):
    """LLM 高置信度（>=0.5）采用 agent 建议。"""
    from agent_base.knowledge_factory import pre_review_document

    monkeypatch.setattr(
        "agent_base.knowledge_factory._llm_classify",
        lambda content, cfg, prev_reject_reason="": ("metadata_doc", 0.82, "命中成分表", "approve", "", []),
    )
    tag = pre_review_document("随便一段内容", llm_cfg={"provider": "langchain"})
    assert tag.doc_type == "metadata_doc"
    assert tag.first_review["confidence"] == 0.82
    assert tag.first_review["source"] == "llm"


def test_pre_review_llm_low_confidence_falls_back(monkeypatch):
    """LLM 低置信度（<0.5）回退启发式，agent 不生效。"""
    from agent_base.knowledge_factory import pre_review_document

    monkeypatch.setattr(
        "agent_base.knowledge_factory._llm_classify",
        lambda content, cfg, prev_reject_reason="": ("faq", 0.2, "低置信", "review", "", []),
    )
    tag = pre_review_document(
        "白色纯棉T恤\n类目: T恤 | 价位: 入门\n面料: 纯棉\n尺码: S-XL\n风格: 基础百搭",
        llm_cfg={"provider": "langchain"},
    )
    assert tag.doc_type == "product_detail"


def test_apply_reject_resubmit_state_machine(monkeypatch):
    """精审 approve → 打回 returned → 重提 round+1。"""
    from agent_base.knowledge_factory import (
        DocTag, apply_tag, persist_tag, reject_tag, submit_document_for_review,
    )

    calls: dict = {}
    monkeypatch.setattr(
        "agent_base.storage.pg.strategy_upsert",
        lambda **kw: calls.update(kw),
    )
    monkeypatch.setattr("agent_base.storage.pg.strategy_get", lambda doc_id: None)

    tag = DocTag(doc_id="doc1", doc_type="faq", strategy=["hypothetical_variants"])
    tag = apply_tag("doc1", tag, reviewer="alice")
    assert tag.status == "approved"
    persist_tag(tag)
    assert calls["status"] == "approved"
    assert calls["review_round"] == 1

    tag = reject_tag("doc1", tag, reviewer="alice", reason="格式不对")
    assert tag.status == "returned"
    assert tag.reject_reason == "格式不对"
    persist_tag(tag)
    assert calls["status"] == "returned"

    monkeypatch.setattr(
        "agent_base.storage.pg.strategy_get",
        lambda doc_id: {
            "status": "returned", "review_round": 1, "doc_type": "faq",
            "strategy": ["hypothetical_variants"], "first_review": {},
        },
    )
    resub = submit_document_for_review("doc1")
    assert resub.status == "pending_fine_review"
    assert resub.review_round == 2
    persist_tag(resub)
    assert calls["review_round"] == 2
    assert calls["status"] == "pending_fine_review"


def test_ingest_gate_blocks_unapproved(monkeypatch):
    """D1 硬约束：无 approved 标签入库 → TagNotApprovedError。"""
    from agent_base.storage.documents import TagNotApprovedError, ingest_document_from_chunks

    monkeypatch.setattr("agent_base.storage.pg.doc_versions", lambda doc_id: [])
    monkeypatch.setattr("agent_base.storage.pg.doc_upsert", lambda *a, **k: 1)
    monkeypatch.setattr("agent_base.knowledge_factory.get_tag", lambda doc_id: None)

    with pytest.raises(TagNotApprovedError):
        ingest_document_from_chunks(
            doc_id="unapproved1", content="x", chunks=[], vector_store=object(),
        )


def test_ingest_gate_passes_approved(monkeypatch):
    """approved 标签放行，正常入库。"""
    from agent_base.knowledge_factory import DocTag
    from agent_base.storage import documents

    monkeypatch.setattr("agent_base.storage.pg.doc_versions", lambda doc_id: [])
    monkeypatch.setattr("agent_base.storage.pg.doc_upsert", lambda *a, **k: 1)
    monkeypatch.setattr(
        "agent_base.knowledge_factory.get_tag",
        lambda doc_id: DocTag(doc_id=doc_id, doc_type="faq", strategy=[], status="approved"),
    )
    monkeypatch.setattr("agent_base.storage.cache.invalidate_pattern", lambda *a, **k: None)

    class FakeVS:
        def __init__(self):
            self.added = False

        def add_texts(self, texts, metadatas, ids):
            self.added = True

        def delete(self, ids):
            pass

    vs = FakeVS()
    chunks = [{"chunk_id": "c1", "text": "x", "metadata": {"doc_id": "d1", "chunk_id": "c1"}}]
    result = documents.ingest_document_from_chunks(
        doc_id="d1", content="x", chunks=chunks, vector_store=vs,
    )
    assert result["status"] == "ingested"
    assert vs.added


def test_seed_legacy_tags(monkeypatch):
    """D2 存量迁移：无标签文档批量 approved（reviewer=migration）。"""
    from agent_base.knowledge_factory import seed_legacy_tags

    upserted: list[dict] = []
    monkeypatch.setattr("agent_base.storage.pg.list_document_ids", lambda: ["docA", "docB"])
    monkeypatch.setattr("agent_base.storage.pg.strategy_get", lambda doc_id: None)
    monkeypatch.setattr(
        "agent_base.storage.pg.strategy_upsert",
        lambda **kw: upserted.append(kw),
    )

    n = seed_legacy_tags()
    assert n == 2
    assert all(u["status"] == "approved" for u in upserted)
    assert all(u["reviewer"] == "migration" for u in upserted)


def test_heuristic_uses_config_signals(monkeypatch):
    """D5：信号词可配置，启发式读取 tagging.yaml。"""
    from agent_base.knowledge_factory import _heuristic_classify

    monkeypatch.setattr(
        "agent_base.knowledge_factory._load_tagging_cfg",
        lambda: {"doc_types": {"faq": {"signals": ["补货", "预售"]}}},
    )
    assert _heuristic_classify("补货 预售 咨询") == "faq"


def test_api_ingest_403_without_approval():
    """D1 API 层：未打标文档 ingest → 403。"""
    from fastapi.testclient import TestClient

    from agent_base.api.main import create_app

    client = TestClient(create_app())
    doc_id = f"p19gate_{uuid.uuid4().hex[:8]}"
    resp = client.post(
        "/api/documents/ingest",
        json={"doc_id": doc_id, "content": "X1\n\nX2"},
        headers={"X-Admin-Token": TOKEN},
    )
    assert resp.status_code == 403


def test_route_summary_index_when_tag_and_switch_on(monkeypatch):
    """D3: strategy=summary_index + 总闸开启 → 摘要同步执行。"""
    from agent_base.knowledge_factory import DocTag
    from agent_base.storage import documents

    calls = {"summary": 0, "parent": 0}
    monkeypatch.setattr(
        "agent_base.knowledge_factory.get_tag",
        lambda doc_id: DocTag(
            doc_id=doc_id, doc_type="product_detail",
            strategy=["summary_index"], status="approved",
        ),
    )
    monkeypatch.setattr(
        "agent_base.config.load_yaml",
        lambda *a, **k: {"retrieval": {"summary_index": {"enabled": True}}},
    )

    def _fake_sync(doc_id, chunks, old_chunk_ids, summary_store):
        calls["summary"] += 1

    def _fake_parent(doc_id, content, chunks):
        calls["parent"] += 1

    monkeypatch.setattr(documents, "_sync_summaries", _fake_sync)
    monkeypatch.setattr(documents, "_sync_parent_docstore", _fake_parent)

    documents._route_indexing_by_tag("d1", "x", [], [], object())
    assert calls["summary"] == 1
    assert calls["parent"] == 0


def test_route_summary_skipped_when_switch_off(monkeypatch):
    """D3: 总闸关闭时即使标签含 summary_index 也不生成摘要。"""
    from agent_base.knowledge_factory import DocTag
    from agent_base.storage import documents

    monkeypatch.setattr(
        "agent_base.knowledge_factory.get_tag",
        lambda doc_id: DocTag(
            doc_id=doc_id, doc_type="product_detail",
            strategy=["summary_index"], status="approved",
        ),
    )
    monkeypatch.setattr(
        "agent_base.config.load_yaml",
        lambda *a, **k: {"retrieval": {"summary_index": {"enabled": False}}},
    )
    calls = {"summary": 0}
    monkeypatch.setattr(
        documents, "_sync_summaries",
        lambda *a, **k: calls.__setitem__("summary", calls["summary"] + 1),
    )

    documents._route_indexing_by_tag("d1", "x", [], [], object())
    assert calls["summary"] == 0


def test_route_parent_child_writes_parent_docstore(monkeypatch):
    """D3: strategy=parent_child → 父文档 docstore 写入。"""
    from agent_base.knowledge_factory import DocTag
    from agent_base.storage import documents

    calls = {"summary": 0, "parent": 0}
    monkeypatch.setattr(
        "agent_base.knowledge_factory.get_tag",
        lambda doc_id: DocTag(
            doc_id=doc_id, doc_type="product_detail",
            strategy=["parent_child"], status="approved",
        ),
    )

    def _fake_summary(doc_id, chunks, old_chunk_ids, summary_store):
        calls["summary"] += 1

    def _fake_parent(doc_id, content, chunks):
        calls["parent"] += 1

    monkeypatch.setattr(documents, "_sync_summaries", _fake_summary)
    monkeypatch.setattr(documents, "_sync_parent_docstore", _fake_parent)

    documents._route_indexing_by_tag("d1", "x", [], [], None)
    assert calls["parent"] == 1
    assert calls["summary"] == 0
