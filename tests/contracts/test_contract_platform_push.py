"""P23: 数据中台推送文档接口契约。

验证：
  - X-Platform-Token 鉴权，与运营台 X-Admin-Token 分离
  - 推送 → 暂存（pending_fine_review），携带评估建议（source=data_platform）
  - 待审队列可见平台建议的类型/策略，精审抽屉可预填
  - 同内容重复推送 → skipped（内容 sha256 幂等）
  - 鉴权失败（无 token / 错误 token / 用运营台 token）→ 403
"""

import hashlib
import uuid

TOKEN = "admin-dev-token-2026"
PLATFORM_TOKEN = "platform-dev-token-2026"


def _client():
    from fastapi.testclient import TestClient

    from agent_base.api.main import create_app
    return TestClient(create_app(), raise_server_exceptions=True)


def test_platform_push_stages_with_suggestion():
    """数据中台推送 → 暂存 + 评估建议入队，重复推送幂等跳过。"""
    from agent_base.storage.pg import staging_delete

    client = _client()
    content = "# 平台推送\n\n唯一内容标记 {}".format(uuid.uuid4().hex)
    fname = "平台推送_{}.md".format(uuid.uuid4().hex[:8])
    doc_id = hashlib.sha256(content.encode("utf-8")).hexdigest()
    try:
        resp = client.post(
            "/api/platform/documents",
            json={
                "filename": fname,
                "content": content,
                "category": "美妆-精华",
                "doc_type": "metadata_doc",
                "strategy": ["self_query"],
                "confidence": 0.9,
                "reasoning": "成分表类文档",
            },
            headers={"X-Platform-Token": PLATFORM_TOKEN},
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["action"] == "staged"
        assert payload["doc_id"] == doc_id
        assert payload["status"] == "pending"
        assert payload["doc_type"] == "metadata_doc"

        # 待审队列可见（含平台建议类型/策略）
        q = client.get(
            "/api/documents/review-queue?status=pending_fine_review",
            headers={"X-Admin-Token": TOKEN},
        ).json()
        hit = next((x for x in q["queue"] if x["doc_id"] == doc_id), None)
        assert hit is not None
        assert hit["filename"] == fname
        assert hit["doc_type"] == "metadata_doc"
        assert hit["strategy"] == ["self_query"]
        assert hit["confidence"] == 0.9

        # 重复推送同内容 → skipped
        resp2 = client.post(
            "/api/platform/documents",
            json={"filename": fname, "content": content},
            headers={"X-Platform-Token": PLATFORM_TOKEN},
        )
        assert resp2.status_code == 200, resp2.text
        assert resp2.json()["action"] == "skipped"
    finally:
        staging_delete(doc_id)


def test_platform_push_requires_platform_token():
    """平台接口只认 X-Platform-Token；运营台 token 不可混用。"""
    client = _client()
    body = {"filename": "x.md", "content": "hello"}
    assert client.post("/api/platform/documents", json=body).status_code == 403
    assert (
        client.post(
            "/api/platform/documents", json=body, headers={"X-Admin-Token": TOKEN}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/platform/documents",
            json=body,
            headers={"X-Platform-Token": "wrong-token"},
        ).status_code
        == 403
    )


def test_platform_push_rejects_empty_body():
    """filename / content 缺失 → 400。"""
    client = _client()
    resp = client.post(
        "/api/platform/documents",
        json={"filename": "", "content": ""},
        headers={"X-Platform-Token": PLATFORM_TOKEN},
    )
    assert resp.status_code == 400
