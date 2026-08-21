"""Contract P18: per-chunk LLM summarizer (concurrency + fallback + diff-sync).

Covers the P18b regressions that the previous 90 tests missed:
  - ``generate_summaries`` must return one result per chunk with per-item
    fallback (a single LLM failure must not abort the batch);
  - ``sync_document_summaries`` must upsert new summary points and delete
    stale ones (diff-sync, same semantics as the dense vector path).
"""

from __future__ import annotations


def test_generate_summaries_per_item_fallback(monkeypatch):
    """LLM 全部失败时逐项回退原文，结果数量与输入一致。"""
    from agent_base.retrieval import summarizer

    class FakeChain:
        """Chain whose invoke always fails (simulates 429/5xx exhaustion)."""

        def invoke(self, input_dict):
            raise RuntimeError("429 rate limit exceeded")

    chain_builds = {"n": 0}

    def fake_make_summary_chain(llm_cfg=None, timeout=60):
        chain_builds["n"] += 1
        return FakeChain()

    monkeypatch.setattr(summarizer, "make_summary_chain", fake_make_summary_chain)

    chunks = [
        {"chunk_id": "c1", "text": "苹果精华液", "metadata": {"doc_id": "p1", "section": "商品参数"}},
        {"chunk_id": "c2", "text": "白色T恤", "metadata": {"doc_id": "p2", "section": "商品参数"}},
    ]
    out = summarizer.generate_summaries(chunks, max_workers=2)

    assert chain_builds["n"] == 1  # 链只构建一次，线程池复用
    assert len(out) == 2
    assert out[0]["summary"] == "苹果精华液"[:60]  # 失败回退原文截断
    assert out[1]["summary"] == "白色T恤"[:60]
    assert out[0]["metadata"]["summary_type"] == "llm_per_chunk"
    assert out[0]["chunk_id"] == "c1"
    assert out[0]["doc_id"] == "p1"
    assert out[1]["chunk_id"] == "c2"


def test_sync_document_summaries_diff_sync(monkeypatch):
    """更新文档后：新摘要 upsert，旧 chunk 摘要按差集删除。"""
    from agent_base.storage import documents

    # PG 旧版本 chunk_ids（diff 基准）
    monkeypatch.setattr(
        "agent_base.storage.pg.doc_versions",
        lambda doc_id: [{"chunk_ids": ["old1", "old2"]}],
    )
    # 开关开启
    monkeypatch.setattr(
        "agent_base.config.load_yaml",
        lambda *args, **kwargs: {"retrieval": {"summary_index": {"enabled": True}}},
    )
    # 摘要生成直接返回固定结果（不调 LLM）
    fake_summaries = [{
        "summary_id": "sid-new1",
        "summary": "s1",
        "chunk_id": "new1",
        "doc_id": "d1",
        "metadata": {"chunk_id": "new1", "doc_id": "d1"},
    }]
    monkeypatch.setattr(
        "agent_base.retrieval.summarizer.generate_summaries",
        lambda chunks, **kwargs: fake_summaries,
    )

    class FakeSummaryStore:
        """Records add_texts / delete calls for assertion."""

        def __init__(self):
            self.added_ids: list[list[str]] = []
            self.deleted_ids: list[str] = []

        def add_texts(self, texts, metadatas, ids):
            self.added_ids.append(list(ids))

        def delete(self, ids):
            self.deleted_ids.extend(ids)

    store = FakeSummaryStore()
    chunks = [{"chunk_id": "new1", "text": "新内容", "metadata": {"doc_id": "d1"}}]

    documents.sync_document_summaries("d1", chunks, store)

    assert store.added_ids == [["sid-new1"]]
    # 旧 chunk old1/old2 摘要删除；new1 是新 chunk 不误删
    assert len(store.deleted_ids) == 2
