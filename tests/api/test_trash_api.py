"""回收站 API：删除进回收站 / 列表 / 恢复 / 彻底删除 / 过期清理。"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from agent_base.storage.pg import _conn, purge_deleted_documents


def _cleanup_doc(doc_id: str) -> None:
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM documents WHERE doc_id=%s", (doc_id,))
            cur.execute("DELETE FROM document_strategy WHERE doc_id=%s", (doc_id,))
            cur.execute("DELETE FROM document_staging WHERE doc_id=%s", (doc_id,))
    except Exception:
        pass


def _create_ingested_doc(client: TestClient, headers: dict[str, str]) -> str:
    """upload -> 精审确认入库（v1）-> 返回 doc_id。"""
    tag = uuid.uuid4().hex[:8]
    filename = f"trash_{tag}.md"
    content = f"# 回收站测试\n\n## 第一章\n内容 {tag}。".encode("utf-8")
    up = client.post(
        "/api/upload",
        headers=headers,
        files={"file": (filename, content, "text/markdown")},
        data={"category": "运营上传"},
    )
    assert up.status_code == 200, up.text
    doc_id = up.json()["doc_id"]
    apply = client.post(
        "/api/documents/tags/apply",
        json={"doc_id": doc_id, "doc_type": "product_detail", "strategy": ["default_vector"]},
        headers=headers,
    )
    assert apply.status_code == 200, apply.text
    return doc_id


def _trash_ids(client: TestClient, headers: dict[str, str]) -> set[str]:
    r = client.get("/api/documents/trash", headers=headers)
    assert r.status_code == 200, r.text
    return {d["doc_id"] for d in r.json().get("documents", [])}


def test_delete_puts_doc_in_trash(client: TestClient, headers: dict[str, str]):
    doc_id = _create_ingested_doc(client, headers)
    try:
        r = client.delete(f"/api/documents/{doc_id}", headers=headers)
        assert r.status_code == 200, r.text

        trash = client.get("/api/documents/trash", headers=headers).json()["documents"]
        item = next((d for d in trash if d["doc_id"] == doc_id), None)
        assert item is not None, "删除后文档应出现在回收站"
        assert item["status"] == "deleted"
        assert item["deleted_at"]
        assert item["remaining_days"] is not None and 0 <= item["remaining_days"] <= 30
        assert item["version_count"] >= 1
        assert item["doc_name"]

        act = client.get("/api/documents?status=active", headers=headers).json()["documents"]
        assert doc_id not in {d["doc_id"] for d in act}
    finally:
        client.delete(f"/api/documents/{doc_id}", headers=headers)
        _cleanup_doc(doc_id)


def test_recover_doc_from_trash(client: TestClient, headers: dict[str, str]):
    doc_id = _create_ingested_doc(client, headers)
    r2 = client.post(
        "/api/documents/update",
        json={"doc_id": doc_id, "content": "# 回收站测试\n\n## 第一章\nv2 内容。"},
        headers=headers,
    )
    assert r2.status_code == 200, r2.text
    client.delete(f"/api/documents/{doc_id}", headers=headers)
    try:
        r = client.post(f"/api/documents/{doc_id}/recover", json={}, headers=headers)
        assert r.status_code == 200, r.text
        assert r.json()["ok"] is True

        assert doc_id not in _trash_ids(client, headers)
        act = client.get("/api/documents?status=active", headers=headers).json()["documents"]
        assert doc_id in {d["doc_id"] for d in act}

        versions = client.get(f"/api/documents/{doc_id}/versions", headers=headers).json()["versions"]
        statuses = {v["version"]: v["status"] for v in versions}
        top = max(statuses)
        assert statuses[top] == "active"
        assert any(s == "archived" for s in statuses.values()), "历史版本应转 archived"
        top_ver = next(v for v in versions if v["version"] == top)
        assert top_ver["chunk_ids"], "恢复后应重建向量（chunk_ids 非空）"
    finally:
        client.delete(f"/api/documents/{doc_id}", headers=headers)
        _cleanup_doc(doc_id)


def test_purge_doc_permanent(client: TestClient, headers: dict[str, str]):
    doc_id = _create_ingested_doc(client, headers)
    client.delete(f"/api/documents/{doc_id}", headers=headers)
    try:
        r = client.delete(f"/api/documents/{doc_id}/purge", headers=headers)
        assert r.status_code == 200, r.text
        assert doc_id not in _trash_ids(client, headers)
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM documents WHERE doc_id=%s", (doc_id,))
            assert cur.fetchone()[0] == 0, "彻底删除后版本行应物理清除"
    finally:
        _cleanup_doc(doc_id)


def test_purge_deleted_expired_cleanup(client: TestClient, headers: dict[str, str]):
    doc_id = _create_ingested_doc(client, headers)
    try:
        old = datetime.now(timezone.utc) - timedelta(days=40)
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE documents SET status='deleted', deleted_at=%s WHERE doc_id=%s",
                (old, doc_id),
            )
        result = purge_deleted_documents(days=30)
        assert result["documents"] >= 1
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM documents WHERE doc_id=%s", (doc_id,))
            assert cur.fetchone()[0] == 0
    finally:
        _cleanup_doc(doc_id)


def test_recover_then_purge_keeps_history(client: TestClient, headers: dict[str, str]):
    """恢复后 purge 不应误删已恢复文档的历史版本。"""
    doc_id = _create_ingested_doc(client, headers)
    client.post(
        "/api/documents/update",
        json={"doc_id": doc_id, "content": "# 回收站测试\n\n## 第一章\nv2 内容。"},
        headers=headers,
    )
    client.delete(f"/api/documents/{doc_id}", headers=headers)
    r = client.delete(f"/api/documents/{doc_id}/versions/1", headers=headers)
    assert r.status_code == 200, r.text
    client.post(f"/api/documents/{doc_id}/recover", json={}, headers=headers)
    try:
        purge_deleted_documents(days=30)
        versions = client.get(f"/api/documents/{doc_id}/versions", headers=headers).json()["versions"]
        assert len(versions) >= 1
        assert all(v["status"] != "deleted" for v in versions)
        assert any(v["status"] == "archived" for v in versions)
    finally:
        client.delete(f"/api/documents/{doc_id}", headers=headers)
        _cleanup_doc(doc_id)


def test_batch_recover_and_purge(client: TestClient, headers: dict[str, str]):
    d1 = _create_ingested_doc(client, headers)
    d2 = _create_ingested_doc(client, headers)
    try:
        client.delete(f"/api/documents/{d1}", headers=headers)
        client.delete(f"/api/documents/{d2}", headers=headers)
        r = client.post("/api/documents/batch-recover", json={"doc_ids": [d1, d2]}, headers=headers)
        assert r.status_code == 200, r.text
        assert r.json()["recovered"] == 2

        client.delete(f"/api/documents/{d1}", headers=headers)
        client.delete(f"/api/documents/{d2}", headers=headers)
        r2 = client.post("/api/documents/batch-purge", json={"doc_ids": [d1, d2]}, headers=headers)
        assert r2.status_code == 200, r2.text
        assert r2.json()["purged"] == 2
        assert d1 not in _trash_ids(client, headers)
        assert d2 not in _trash_ids(client, headers)
    finally:
        _cleanup_doc(d1)
        _cleanup_doc(d2)


def test_recover_missing_returns_404(client: TestClient, headers: dict[str, str]):
    r = client.post("/api/documents/no_such_doc_trash/recover", json={}, headers=headers)
    assert r.status_code == 404

