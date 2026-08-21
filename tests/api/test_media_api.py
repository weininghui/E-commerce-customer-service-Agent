"""图片知识库接口测试（Phase 2）：上传 / 列表 / 绑定 / 解析 / 删除。"""

from __future__ import annotations

import base64
import uuid

from fastapi.testclient import TestClient

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _upload(client: TestClient, headers: dict[str, str], filename: str | None = None):
    tag = uuid.uuid4().hex[:8]
    name = filename or f"test_{tag}.png"
    r = client.post(
        "/api/media/upload",
        headers=headers,
        files={"file": (name, PNG_1PX, "image/png")},
        data={"description": "接口测试图"},
    )
    return r


def test_media_upload_list_bind_parse_delete_lifecycle(client: TestClient, headers: dict[str, str]):
    """完整生命周期：上传 → 列表 → 绑定 → 解析入队 → 删除清理。"""
    up = _upload(client, headers)
    assert up.status_code == 200, up.text
    body = up.json()
    media_id = body["id"]
    assert body["ok"] is True and body["url"].startswith("/media/uploads/")
    try:
        lst = client.get("/api/media/list", headers=headers).json()
        ids = [x["id"] for x in lst.get("items", [])]
        assert media_id in ids

        bind = client.post(f"/api/media/{media_id}/bind", json={"product_id": "P001"}, headers=headers)
        assert bind.status_code == 200, bind.text
        assert bind.json()["product_id"] == "P001"

        parse = client.post(f"/api/media/{media_id}/parse", headers=headers)
        assert parse.status_code == 200, parse.text
        assert parse.json()["task_id"] > 0

        # 解绑
        unbind = client.post(f"/api/media/{media_id}/bind", json={"product_id": ""}, headers=headers)
        assert unbind.status_code == 200
    finally:
        dele = client.delete(f"/api/media/{media_id}", headers=headers)
        assert dele.status_code == 200, dele.text
    lst2 = client.get("/api/media/list", headers=headers).json()
    assert media_id not in [x["id"] for x in lst2.get("items", [])]


def test_media_upload_rejects_bad_type(client: TestClient, headers: dict[str, str]):
    r = client.post(
        "/api/media/upload",
        headers=headers,
        files={"file": ("evil.exe", b"MZ", "application/octet-stream")},
    )
    assert r.status_code == 400
    assert "不支持" in r.json()["detail"]


def test_media_endpoints_require_admin(client: TestClient):
    assert client.post("/api/media/upload").status_code in (401, 403)
    assert client.get("/api/media/list").status_code in (401, 403)
    assert client.delete("/api/media/1").status_code in (401, 403)


def test_media_parse_missing_record(client: TestClient, headers: dict[str, str]):
    r = client.post("/api/media/999999/parse", headers=headers)
    assert r.status_code == 404


def test_media_status_transitions(client: TestClient, headers: dict[str, str]):
    """审核状态流转：pending → approved → rejected，非法状态 400。"""
    up = _upload(client, headers)
    assert up.status_code == 200, up.text
    media_id = up.json()["id"]
    try:
        r1 = client.post(f"/api/media/{media_id}/status", json={"status": "approved"}, headers=headers)
        assert r1.status_code == 200 and r1.json()["status"] == "approved"
        r2 = client.post(f"/api/media/{media_id}/status", json={"status": "rejected"}, headers=headers)
        assert r2.status_code == 200
        r3 = client.post(f"/api/media/{media_id}/status", json={"status": "nonsense"}, headers=headers)
        assert r3.status_code == 400
        r4 = client.post("/api/media/999999/status", json={"status": "approved"}, headers=headers)
        assert r4.status_code == 404
    finally:
        client.delete(f"/api/media/{media_id}", headers=headers)