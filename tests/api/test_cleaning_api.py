"""文件清洗工作台接口测试（两段式入库第一段）。"""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient


def _upload_md(client: TestClient, headers: dict[str, str]):
    tag = uuid.uuid4().hex[:8]
    content = f"# 清洗测试 {tag}\n\n这是待清洗的正文内容。".encode("utf-8")
    return client.post(
        "/api/clean/upload",
        headers=headers,
        files={"file": (f"clean_{tag}.md", content, "text/markdown")},
    )


def test_clean_lifecycle_upload_edit_push(client: TestClient, headers: dict[str, str]):
    up = _upload_md(client, headers)
    assert up.status_code == 200, up.text
    body = up.json()
    assert body["engine"] == "direct"
    assert "清洗测试" in body["text"]
    draft_id = body["id"]

    lst = client.get("/api/clean/list", headers=headers).json()
    assert any(x["id"] == draft_id for x in lst.get("items", []))

    detail = client.get(f"/api/clean/{draft_id}", headers=headers)
    assert detail.status_code == 200
    assert detail.json()["status"] == "pending"

    edit = client.put(f"/api/clean/{draft_id}", json={"text": "人工清洗后的最终文本"}, headers=headers)
    assert edit.status_code == 200

    push = client.post(f"/api/clean/{draft_id}/push", json={"category": "商品说明"}, headers=headers)
    assert push.status_code == 200, push.text
    assert push.json()["doc_id"]

    detail2 = client.get(f"/api/clean/{draft_id}", headers=headers).json()
    assert detail2["status"] == "pushed"
    assert detail2["cleaned_text"] == "人工清洗后的最终文本"

    doc_id = push.json()["doc_id"]
    client.delete(f"/api/clean/{draft_id}", headers=headers)
    client.delete(f"/api/documents/{doc_id}", headers=headers)


def test_clean_upload_bad_type(client: TestClient, headers: dict[str, str]):
    r = client.post(
        "/api/clean/upload",
        headers=headers,
        files={"file": ("x.exe", b"MZ", "application/octet-stream")},
    )
    assert r.status_code == 400


def test_clean_docx_routes_to_parser_mock(client: TestClient, headers: dict[str, str]):
    import io
    import zipfile

    xml = (
        "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">"
        "<w:body><w:p><w:t>神经酰胺修护面霜</w:t></w:p></w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", xml)
    up = client.post(
        "/api/clean/upload",
        headers=headers,
        files={"file": (f"doc_{uuid.uuid4().hex[:8]}.docx", buf.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert up.status_code == 200, up.text
    assert up.json()["engine"] == "mock"
    assert "神经酰胺修护面霜" in up.json()["text"]
    client.delete(f"/api/clean/{up.json()["id"]}", headers=headers)


def test_clean_endpoints_require_admin(client: TestClient):
    assert client.post("/api/clean/upload").status_code in (401, 403)
    assert client.get("/api/clean/list").status_code in (401, 403)
    assert client.get("/api/clean/1").status_code in (401, 403)
    assert client.put("/api/clean/1", json={"text": "x"}).status_code in (401, 403)
    assert client.post("/api/clean/1/push", json={}).status_code in (401, 403)
    assert client.delete("/api/clean/1").status_code in (401, 403)
    assert client.post("/api/clean/1/polish").status_code in (401, 403)


def test_clean_polish_endpoint(client: TestClient, headers: dict[str, str], monkeypatch):
    """AI 整理格式：结果写回 cleaned_text；不存在 404。"""
    from agent_base import cleaning as cleaning_mod

    up = _upload_md(client, headers)
    draft_id = up.json()["id"]
    monkeypatch.setattr(cleaning_mod, "polish_clean_text", lambda text: "# 整理后\n\n## 使用方法\n- " + text.split("\n")[0])
    r = client.post(f"/api/clean/{draft_id}/polish", headers=headers)
    assert r.status_code == 200, r.text
    assert r.json()["polished"].startswith("# 整理后")
    detail = client.get(f"/api/clean/{draft_id}", headers=headers).json()
    assert detail["cleaned_text"].startswith("# 整理后")
    client.delete(f"/api/clean/{draft_id}", headers=headers)
    r2 = client.post("/api/clean/999999/polish", headers=headers)
    assert r2.status_code == 404