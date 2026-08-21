"""契约 P16：文档生命周期（ingest / update / restore / delete）与向量库一致性。

覆盖 v0.24.x 验收修复的回归点：
1. ingest 后 PG 记录 + Qdrant 向量数量一致
2. update 后新向量写入、旧向量清除，Qdrant 不残留孤儿向量（含内容化 chunk_id 修复）
3. restore 后目标版本向量恢复、当前 active 版本多余向量清除
4. delete 后全部向量清除 + PG 标记 deleted
5. update 拒绝外部传入 chunk_ids

依赖真实 Qdrant + PG 服务（与运行环境一致），使用独立 doc_id 不污染现有数据。
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import uuid

import psycopg2
import pytest
from fastapi.testclient import TestClient

from agent_base.api.main import create_app


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("AGENT_BASE_COLLECTION", "ecommerce_chunks")
PG_URL = os.getenv("PG_URL", "postgresql://postgres:postgres@localhost:5432/ragdb")
TOKEN = "admin-dev-token-2026"


def _qdrant_point_id(chunk_id: str) -> str:
    """与 vector_index._qdrant_point_id 保持一致的确定性 UUID 映射。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag:{chunk_id}"))


def _scroll_doc_points(doc_id: str) -> dict[str, dict]:
    """返回该 doc 在 Qdrant 中的全部 point（id -> payload.metadata）。"""
    out: dict[str, dict] = {}
    offset = None
    while True:
        body: dict = {"limit": 500, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        req = urllib.request.Request(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            res = json.loads(r.read())["result"]
        for p in res["points"]:
            meta = (p.get("payload") or {}).get("metadata") or {}
            if meta.get("doc_id") == doc_id:
                out[p["id"]] = meta
        if res.get("next_page_offset") is None:
            break
        offset = res["next_page_offset"]
    return out


def _wait_doc_points(doc_id: str, expected: int, tries: int = 5, interval: float = 0.5) -> dict[str, dict]:
    """等待 Qdrant 写删生效后返回该 doc 的点集合（幂等重试，消除时序竞争）。"""
    points: dict[str, dict] = {}
    for _ in range(tries):
        points = _scroll_doc_points(doc_id)
        if len(points) == expected:
            return points
        time.sleep(interval)
    return points


def _pg_versions(doc_id: str) -> list[dict]:
    with psycopg2.connect(PG_URL) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT version, content, chunk_ids, status FROM documents WHERE doc_id=%s ORDER BY version",
            (doc_id,),
        )
        return [
            {"version": r[0], "content": r[1] or "", "chunk_ids": r[2] or [], "status": r[3]}
            for r in cur.fetchall()
        ]


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture()
def doc_id() -> str:
    return f"p16test_{uuid.uuid4().hex[:10]}"


@pytest.fixture(autouse=True)
def cleanup(client: TestClient, doc_id: str):
    """每个用例前后清理测试文档，避免污染。"""
    yield
    client.delete(f"/api/documents/{doc_id}", headers={"X-Admin-Token": TOKEN})
    # v0.30.2：打标记录也清理，避免测试残留标签污染文档列表
    try:
        from agent_base.storage.pg import strategy_delete
        strategy_delete(doc_id)
    except Exception:
        pass


def _post(client: TestClient, path: str, body: dict, token: str = TOKEN):
    resp = client.post(path, json=body, headers={"X-Admin-Token": token})
    return resp


def _approve(client: TestClient, doc_id: str):
    """P19 D1: 精审通过（approved）后文档才允许入库。"""
    resp = _post(
        client,
        "/api/documents/tags/apply",
        {"doc_id": doc_id, "doc_type": "product_detail"},
    )
    assert resp.status_code == 200, resp.text


def test_ingest_writes_pg_and_qdrant(client: TestClient, doc_id: str):
    """ingest 后 PG 记录与 Qdrant 向量数量一致。"""
    _approve(client, doc_id)
    content = "ALPHA-PARA\n\nBETA-PARA\n\nGAMMA-PARA"
    resp = _post(client, "/api/documents/ingest", {"doc_id": doc_id, "content": content, "category": "test"})
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["chunk_count"] == 3

    versions = _pg_versions(doc_id)
    assert len(versions) == 1
    assert versions[0]["content"] == content
    assert len(versions[0]["chunk_ids"]) == 3
    assert versions[0]["status"] == "active"

    points = _wait_doc_points(doc_id, expected=3)
    assert len(points) == 3
    for cid in versions[0]["chunk_ids"]:
        assert _qdrant_point_id(cid) in points


def test_update_replaces_vectors_without_orphans(client: TestClient, doc_id: str):
    """update 后新向量生效、旧向量清除，Qdrant 不残留孤儿向量。"""
    _approve(client, doc_id)
    _post(client, "/api/documents/ingest", {"doc_id": doc_id, "content": "A1\n\nA2\n\nA3", "category": "test"})
    resp = _post(client, "/api/documents/update", {"doc_id": doc_id, "content": "B1\n\nB2", "category": "test"})
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["version"] == 2
    assert payload["chunk_count"] == 2

    versions = _pg_versions(doc_id)
    assert len(versions) == 2
    active = [v for v in versions if v["status"] == "active"]
    assert len(active) == 1 and active[0]["version"] == 2

    # 核心断言：Qdrant 恰好只有新版本的 2 个向量，旧 3 个全部清除
    points = _wait_doc_points(doc_id, expected=2)
    assert len(points) == 2
    for cid in active[0]["chunk_ids"]:
        assert _qdrant_point_id(cid) in points


def test_restore_syncs_vectors(client: TestClient, doc_id: str):
    """restore 后目标版本向量恢复，当前 active 版本的额外向量被清除。"""
    _approve(client, doc_id)
    _post(client, "/api/documents/ingest", {"doc_id": doc_id, "content": "R1\n\nR2\n\nR3", "category": "test"})
    _post(client, "/api/documents/update", {"doc_id": doc_id, "content": "S1\n\nS2", "category": "test"})

    resp = client.post(f"/api/documents/{doc_id}/restore/1", headers={"X-Admin-Token": TOKEN})
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["vectors_synced"] == 3
    assert payload["removed_vectors"] == 2  # v2 的 2 个向量被清理

    versions = _pg_versions(doc_id)
    active = [v for v in versions if v["status"] == "active"]
    assert len(active) == 1 and active[0]["version"] == 1  # v0.30.2: 切换版本，不递增
    v1 = [v for v in versions if v["version"] == 1][0]
    assert set(active[0]["chunk_ids"]) == set(v1["chunk_ids"])

    points = _wait_doc_points(doc_id, expected=3)
    assert len(points) == 3
    for cid in v1["chunk_ids"]:
        assert _qdrant_point_id(cid) in points


def test_delete_clears_vectors_and_archives(client: TestClient, doc_id: str):
    """delete 后全部版本向量清除，PG 标记 deleted。"""
    _approve(client, doc_id)
    _post(client, "/api/documents/ingest", {"doc_id": doc_id, "content": "D1\n\nD2", "category": "test"})
    resp = client.delete(f"/api/documents/{doc_id}", headers={"X-Admin-Token": TOKEN})
    assert resp.status_code == 200, resp.text

    versions = _pg_versions(doc_id)
    assert versions and all(v["status"] == "deleted" for v in versions)
    assert _wait_doc_points(doc_id, expected=0) == {}


def test_update_rejects_chunk_ids(client: TestClient, doc_id: str):
    """update 不接受外部 chunk_ids（索引由服务端自动管理）。"""
    resp = _post(
        client,
        "/api/documents/update",
        {"doc_id": doc_id, "content": "X1", "chunk_ids": ["fake-1"]},
    )
    assert resp.status_code == 400


def test_pg_docstore_roundtrip_returns_document():
    """v0.28.2: PGDocStore mget 必须反序列化为 Document。

    官方 MultiVectorRetriever 直接返回 docstore.mget() 结果当 Document，
    此前 mget 返回原始 BYTEA memoryview 导致"命中摘要回原块"失效。
    """
    from langchain_core.documents import Document

    from agent_base.storage.docstore import PGDocStore

    store = PGDocStore(table_name="docstore_summary")
    key = f"roundtrip_{uuid.uuid4().hex[:8]}"
    doc = Document(page_content="原文块", metadata={"chunk_id": key, "doc_id": "t1"})
    try:
        store.mset([(key, doc)])
        got = store.mget([key])[0]
        assert got is not None
        assert got.page_content == "原文块"
        assert got.metadata.get("chunk_id") == key
    finally:
        store.mdelete([key])
