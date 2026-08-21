"""后端 API 接口测试 fixtures（tests/api/ 独立目录）。

用 TestClient 模拟前端真实调用（登录 → 带 X-Admin-Token 调接口）。
依赖真实 PG / Redis（与运行环境一致）；写操作的测试数据在用例内清理还原。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent_base.api.main import create_app
from agent_base.storage.pg import doc_set_status


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=True)


@pytest.fixture()
def admin_token(client: TestClient) -> str:
    r = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert r.status_code == 200, r.text
    return r.json()["token"]


@pytest.fixture()
def headers(admin_token: str) -> dict[str, str]:
    return {"X-Admin-Token": admin_token}


@pytest.fixture()
def agent_headers(client: TestClient) -> dict[str, str]:
    """v0.49: 客服（role=agent）登录 token——人工端接口专用。"""
    r = client.post("/api/auth/login", json={"username": "agent", "password": "agent123"})
    assert r.status_code == 200, r.text
    return {"X-Admin-Token": r.json()["token"]}


@pytest.fixture()
def user_headers(client: TestClient) -> dict[str, str]:
    """v0.51: 买家（role=user）登录 token——转人工触发/会话消息归属校验。"""
    r = client.post("/api/auth/login", json={"username": "user", "password": "user123"})
    assert r.status_code == 200, r.text
    return {"X-Admin-Token": r.json()["token"]}


@pytest.fixture()
def first_active_doc(client: TestClient, headers: dict[str, str]) -> str | None:
    """取第一篇 active 文档 doc_id（用于归档/恢复往返测试），无文档返回 None。"""
    r = client.get("/api/documents?status=active", headers=headers)
    if r.status_code != 200:
        return None
    docs = r.json().get("documents", [])
    return docs[0]["doc_id"] if docs else None


@pytest.fixture()
def restore_doc_status():
    """确保文档状态测试后还原为 active。"""
    restored: list[str] = []

    def _track(doc_id: str) -> None:
        restored.append(doc_id)

    yield _track
    for doc_id in restored:
        try:
            doc_set_status(doc_id, "active")
        except Exception:
            pass


@pytest.fixture()
def runtime_supervisor():
    """P33a：Supervisor 编排测试用运行期对象（真实服务）。"""
    from agent_base.api.main import get_runtime

    return get_runtime()
