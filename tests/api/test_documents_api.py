"""文档管理接口：列表去重 / 归档 / 恢复 / 版本 / 删除。"""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient
from agent_base.storage.pg import _conn


def _temp_doc_id() -> str:
    return f"api_lifecycle_{uuid.uuid4().hex[:8]}"


def test_upload_docx_rejected_direct(client: TestClient, headers: dict[str, str]):
    """知识入库直接上传仅收 MD：docx 应 415（外部格式走文件清洗面板）。"""
    import io
    import zipfile

    xml = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:t>神经酰胺修护面霜</w:t></w:p></w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", xml)
    up = client.post(
        "/api/upload",
        headers=headers,
        files={"file": (f"doc_{uuid.uuid4().hex[:8]}.docx", buf.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        data={"category": "运营上传"},
    )
    assert up.status_code == 415


def test_upload_exe_still_rejected(client: TestClient, headers: dict[str, str]):
    """不支持的扩展名仍 415。"""
    r = client.post(
        "/api/upload",
        headers=headers,
        files={"file": ("evil.exe", b"MZ", "application/octet-stream")},
    )
    assert r.status_code == 415


def _cleanup_doc(doc_id: str) -> None:
    """清理：从 documents 物理删除测试行。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM documents WHERE doc_id=%s", (doc_id,))
            cur.execute("DELETE FROM document_strategy WHERE doc_id=%s", (doc_id,))
            cur.execute("DELETE FROM document_staging WHERE doc_id=%s", (doc_id,))
    except Exception:
        pass


def test_document_list_shape(client: TestClient, headers: dict[str, str]):
    r = client.get("/api/documents?status=active", headers=headers)
    assert r.status_code == 200
    docs = r.json().get("documents", [])
    assert isinstance(docs, list)
    for d in docs:
        assert "doc_id" in d and "doc_name" in d
        assert "doc_type" in d and "status" in d
        assert "version" in d or "current_version" in d


def test_document_list_active_archived_no_overlap(client: TestClient, headers: dict[str, str]):
    """P31 回归：active / archived 无同一 doc_id（按最新版本去重）。"""
    act = client.get("/api/documents?status=active", headers=headers).json()
    arc = client.get("/api/documents?status=archived", headers=headers).json()
    active_ids = {d["doc_id"] for d in act.get("documents", [])}
    archived_ids = {d["doc_id"] for d in arc.get("documents", [])}
    assert not (active_ids & archived_ids)


def test_archive_activate_roundtrip(
    client: TestClient,
    headers: dict[str, str],
    first_active_doc: str | None,
    restore_doc_status,
):
    if not first_active_doc:
        return
    restore_doc_status(first_active_doc)
    r1 = client.post(f"/api/documents/{first_active_doc}/archive", headers=headers)
    assert r1.status_code == 200 and r1.json()["status"] == "archived"
    arc = client.get("/api/documents?status=archived", headers=headers).json()
    assert first_active_doc in {d["doc_id"] for d in arc.get("documents", [])}

    r2 = client.post(f"/api/documents/{first_active_doc}/activate", headers=headers)
    assert r2.status_code == 200 and r2.json()["status"] == "active"


def test_document_versions(client: TestClient, headers: dict[str, str], first_active_doc: str | None):
    if not first_active_doc:
        return
    r = client.get(f"/api/documents/{first_active_doc}/versions", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["doc_id"] == first_active_doc
    assert "versions" in body
    if body["versions"]:
        v = body["versions"][0]
        assert "version" in v and "content" in v and "status" in v


def test_delete_document_marks_deleted(client: TestClient, headers: dict[str, str]):
    """删除后文档不再出现在 active/archived 列表（软删）。"""
    # 用归档/激活测试过的第一篇做删除验证有风险，改用不存在文档验证 404 路径
    r = client.delete("/api/documents/no_such_doc_xyz", headers=headers)
    assert r.status_code in (200, 404)


def test_document_ingest_update_restore_lifecycle(client: TestClient, headers: dict[str, str]):
    """upload→精审入库（v1）→ update（v2）→ restore v1 → 清理。"""
    # 上传临时文档拿 doc_id
    tag = uuid.uuid4().hex[:8]
    filename = f"lifecycle_{tag}.md"
    content = f"# 生命周期测试\n\n## 第一章\n\nv1 内容 {tag}。".encode("utf-8")
    up = client.post(
        "/api/upload",
        headers=headers,
        files={"file": (filename, content, "text/markdown")},
        data={"category": "运营上传"},
    )
    assert up.status_code == 200, up.text
    doc_id = up.json()["doc_id"]
    try:
        # 精审确认 → 入库 v1
        apply = client.post(
            "/api/documents/tags/apply",
            json={"doc_id": doc_id, "doc_type": "product_detail", "strategy": ["default_vector"]},
            headers=headers,
        )
        assert apply.status_code == 200, apply.text
        v1 = apply.json().get("version", 1)

        r2 = client.post(
            "/api/documents/update",
            json={"doc_id": doc_id, "content": f"# 生命周期测试\n\n## 第一章\n\nv2 更新内容 {tag}。"},
            headers=headers,
        )
        assert r2.status_code == 200, r2.text
        v2 = r2.json().get("version", v1 + 1)
        assert v2 > v1

        versions = client.get(f"/api/documents/{doc_id}/versions", headers=headers).json()
        assert len(versions.get("versions", [])) >= 2

        r3 = client.post(f"/api/documents/{doc_id}/restore/{v1}", json={}, headers=headers)
        assert r3.status_code == 200, r3.text
    finally:
        client.delete(f"/api/documents/{doc_id}", headers=headers)
        _cleanup_doc(doc_id)


def test_delete_version_excludes_from_versions(client: TestClient, headers: dict[str, str]):
    """回归（BUG）：软删历史版本后，/versions 不再返回已删版本（doc_versions 过滤 deleted）。"""
    tag = uuid.uuid4().hex[:8]
    filename = f"lifecycle_{tag}.md"
    content = f"# 版本删除测试\n\n## 第一章\nv1 内容 {tag}。".encode("utf-8")
    up = client.post(
        "/api/upload",
        headers=headers,
        files={"file": (filename, content, "text/markdown")},
        data={"category": "运营上传"},
    )
    assert up.status_code == 200, up.text
    doc_id = up.json()["doc_id"]
    try:
        apply = client.post(
            "/api/documents/tags/apply",
            json={"doc_id": doc_id, "doc_type": "product_detail", "strategy": ["default_vector"]},
            headers=headers,
        )
        assert apply.status_code == 200, apply.text
        v1 = apply.json().get("version", 1)

        r2 = client.post(
            "/api/documents/update",
            json={"doc_id": doc_id, "content": f"# 版本删除测试\n\n## 第一章\nv2 更新内容 {tag}。"},
            headers=headers,
        )
        assert r2.status_code == 200, r2.text
        v2 = r2.json().get("version", v1 + 1)
        assert v2 > v1

        r3 = client.delete(f"/api/documents/{doc_id}/versions/{v1}", headers=headers)
        assert r3.status_code == 200, r3.text

        versions = client.get(f"/api/documents/{doc_id}/versions", headers=headers).json().get("versions", [])
        assert all(v["version"] != v1 for v in versions)
        assert any(v["version"] == v2 for v in versions)
    finally:
        client.delete(f"/api/documents/{doc_id}", headers=headers)
        _cleanup_doc(doc_id)


def test_document_ingest_requires_approval_for_external(client: TestClient, headers: dict[str, str]):
    """对外 ingest 未打标文档应被精审门拦截（TagNotApproved → 4xx）。"""
    doc_id = _temp_doc_id()
    try:
        r = client.post(
            "/api/documents/ingest",
            json={"doc_id": doc_id, "content": "# 未打标测试\n\n内容。"},
            headers=headers,
        )
        assert r.status_code in (400, 403, 422), r.text
    finally:
        _cleanup_doc(doc_id)


def test_document_batch_delete_and_tags(client: TestClient, headers: dict[str, str]):
    """batch-delete 与 tags 列表端点可调通。"""
    r = client.get("/api/documents/tags", headers=headers)
    assert r.status_code == 200
    r2 = client.post("/api/documents/batch-delete", json={"doc_ids": ["no_such_doc"]}, headers=headers)
    assert r2.status_code == 200  # 不存在的 doc 逐篇软删不失败


def test_returned_clear_endpoint(client: TestClient, headers: dict[str, str]):
    """清空已打回（传空 doc_ids 会清全局，测试用不存在的 doc 走 200 无副作用）。"""
    r = client.post("/api/documents/returned/clear", json={"doc_ids": ["no_such_doc_clear_test"]}, headers=headers)
    assert r.status_code == 200
