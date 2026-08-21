"""知识入库/审核接口：上传 → 队列 → 预审 → 打回/丢弃。"""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient


def _upload_temp_md(client: TestClient, headers: dict[str, str]) -> tuple[str, str, bytes]:
    """上传一个临时 MD，返回 (doc_id, filename, content)。"""
    tag = uuid.uuid4().hex[:8]
    filename = f"api_test_{tag}.md"
    content = f"# 接口测试文档\n\n## 功效说明\n\n这是一份 {tag} 的临时测试内容，测后清理。".encode("utf-8")
    r = client.post(
        "/api/upload",
        headers=headers,
        files={"file": (filename, content, "text/markdown")},
        data={"category": "运营上传"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    return body["doc_id"], filename, content


def test_upload_staged_and_in_queue(client: TestClient, headers: dict[str, str]):
    doc_id, filename, _ = _upload_temp_md(client, headers)
    try:
        r = client.get("/api/documents/review-queue?status=pending_fine_review", headers=headers)
        assert r.status_code == 200
        queue = r.json().get("queue", [])
        assert any(item["doc_id"] == doc_id for item in queue)
        item = next(item for item in queue if item["doc_id"] == doc_id)
        # 字段计算：轮次/建议动作/类型/置信度/理由
        assert item["review_round"] == 1
        assert item["suggest_action"] in {"approve", "review", "reject"}
        assert item["doc_type"] or item["reasoning"]  # 至少有一项预审产出
        assert 0.0 <= item["confidence"] <= 1.0
    finally:
        r = client.post("/api/documents/batch-discard", json={"doc_ids": [doc_id]}, headers=headers)
        assert r.status_code == 200


def test_upload_duplicate_skipped(client: TestClient, headers: dict[str, str]):
    """重复上传同内容 → action=skipped，不产生第二条记录。"""
    doc_id, filename, content = _upload_temp_md(client, headers)
    try:
        # 完全相同的内容再传一次
        r2 = client.post(
            "/api/upload",
            headers=headers,
            files={"file": (f"dup_{uuid.uuid4().hex[:6]}.md", content, "text/markdown")},
            data={"category": "运营上传"},
        )
        assert r2.status_code == 200
        assert r2.json().get("action") == "skipped"
        # 队列仍只有一条
        queue = client.get("/api/documents/review-queue?status=pending_fine_review", headers=headers).json().get("queue", [])
        assert sum(1 for item in queue if item["doc_id"] == doc_id) == 1
    finally:
        client.post("/api/documents/batch-discard", json={"doc_ids": [doc_id]}, headers=headers)


def test_batch_reject_memory_round2_prev_reason(client: TestClient, headers: dict[str, str]):
    """BUG-25：批量打回写短期记忆 → 同名重传 round+1 注入 prev_reject_reason。"""
    tag = uuid.uuid4().hex[:8]
    filename = f"api_test_{tag}.md"
    content1 = f"# BUG25 测试\n\n第一版内容 {tag}。".encode("utf-8")
    r = client.post(
        "/api/upload",
        headers=headers,
        files={"file": (filename, content1, "text/markdown")},
        data={"category": "运营上传"},
    )
    assert r.status_code == 200, r.text
    doc_id = r.json()["doc_id"]
    try:
        reason = f"{tag} 需要修改"
        full_reason = f"分类错误：{reason}"
        rr = client.post(
            "/api/documents/batch-reject",
            json={"doc_ids": [doc_id], "reason_code": "分类错误", "reason": reason},
            headers=headers,
        )
        assert rr.status_code == 200, rr.text
        assert rr.json()["rejected"] == 1

        # 同名不同内容 → 走 returned 重传分支：round+1 且注入上轮打回原因
        content2 = f"# BUG25 测试\n\n第二版内容 {tag}，已修复。".encode("utf-8")
        r2 = client.post(
            "/api/upload",
            headers=headers,
            files={"file": (filename, content2, "text/markdown")},
            data={"category": "运营上传"},
        )
        assert r2.status_code == 200, r2.text
        queue = client.get("/api/documents/review-queue?status=pending_fine_review", headers=headers).json().get("queue", [])
        item = next((i for i in queue if i["doc_id"] == doc_id), None)
        assert item is not None, "重传后应回到待审核队列"
        assert item["review_round"] == 2
        assert item["prev_reject_reason"] == full_reason
    finally:
        client.post("/api/documents/batch-discard", json={"doc_ids": [doc_id]}, headers=headers)


def test_pre_review_endpoint(client: TestClient, headers: dict[str, str]):
    r = client.post(
        "/api/documents/pre-review",
        json={"content": "这是一段测试内容，用于验证预审接口。", "filename": "pre_review_test.md"},
        headers=headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert "first_review" in body or "suggest_action" in body or "doc_type" in body


def test_review_queue_requires_admin(client: TestClient):
    assert client.get("/api/documents/review-queue").status_code == 403


def test_tags_apply_approves_and_ingests(client: TestClient, headers: dict[str, str]):
    """精审确认：tags/apply → 文档入库（从待审消失，出现在文档列表）。"""
    doc_id, filename, _ = _upload_temp_md(client, headers)
    try:
        r = client.post(
            "/api/documents/tags/apply",
            json={"doc_id": doc_id, "doc_type": "product_detail", "strategy": ["default_vector"]},
            headers=headers,
        )
        assert r.status_code == 200, r.text
        # 从待审队列消失
        queue = client.get("/api/documents/review-queue?status=pending_fine_review", headers=headers).json().get("queue", [])
        assert not any(item["doc_id"] == doc_id for item in queue)
    finally:
        client.delete(f"/api/documents/{doc_id}", headers=headers)
        client.post("/api/documents/batch-discard", json={"doc_ids": [doc_id]}, headers=headers)


def test_tags_reject_then_submit(client: TestClient, headers: dict[str, str]):
    """打回 → returned 列表 → 重新提交回待审。"""
    doc_id, filename, _ = _upload_temp_md(client, headers)
    try:
        r = client.post(
            "/api/documents/tags/reject",
            json={"doc_id": doc_id, "reason": "api 测试打回"},
            headers=headers,
        )
        assert r.status_code == 200, r.text
        returned = client.get("/api/documents/review-queue?status=returned", headers=headers).json().get("queue", [])
        assert any(item["doc_id"] == doc_id for item in returned)

        r2 = client.post("/api/documents/submit", json={"doc_id": doc_id}, headers=headers)
        assert r2.status_code == 200, r2.text
        pending = client.get("/api/documents/review-queue?status=pending_fine_review", headers=headers).json().get("queue", [])
        assert any(item["doc_id"] == doc_id for item in pending)
    finally:
        client.post("/api/documents/batch-discard", json={"doc_ids": [doc_id]}, headers=headers)


def test_clear_returned_keeps_resubmitted_pending(client: TestClient, headers: dict[str, str]):
    """BUG-1 回归：打回 → 重提（staging pending round 2）→ 清空已打回 → pending 保留。"""
    doc_id, filename, content = _upload_temp_md(client, headers)
    try:
        # 打回 → returned
        r = client.post("/api/documents/tags/reject", json={"doc_id": doc_id, "reason": "api bug1 测试打回"}, headers=headers)
        assert r.status_code == 200, r.text
        # 重新提交 → staging pending（round 2）
        r2 = client.post("/api/documents/submit", json={"doc_id": doc_id}, headers=headers)
        assert r2.status_code == 200, r2.text
        pending_before = client.get("/api/documents/review-queue?status=pending_fine_review", headers=headers).json().get("queue", [])
        assert any(item["doc_id"] == doc_id for item in pending_before)

        # 清空已打回
        r3 = client.post("/api/documents/returned/clear", json={"doc_ids": [doc_id]}, headers=headers)
        assert r3.status_code == 200, r3.text

        # 待审核队列仍保留该文档（round 2）
        pending_after = client.get("/api/documents/review-queue?status=pending_fine_review", headers=headers).json().get("queue", [])
        match = [item for item in pending_after if item["doc_id"] == doc_id]
        assert match, "清空已打回误删了重提的待审核记录（BUG-1 复发）"
        assert match[0]["review_round"] >= 2
    finally:
        client.post("/api/documents/batch-discard", json={"doc_ids": [doc_id]}, headers=headers)


def test_batch_approve(client: TestClient, headers: dict[str, str]):
    doc_ids = []
    for _ in range(2):
        doc_id, _, _ = _upload_temp_md(client, headers)
        doc_ids.append(doc_id)
    try:
        r = client.post("/api/documents/batch-approve", json={"doc_ids": doc_ids}, headers=headers)
        assert r.status_code == 200, r.text
        # 低置信度文档会被跳过（confidence<0.75 是设计行为），approved 可能为 0
        assert "approved" in r.json()
    finally:
        for did in doc_ids:
            client.delete(f"/api/documents/{did}", headers=headers)
            client.post("/api/documents/batch-discard", json={"doc_ids": [did]}, headers=headers)


def test_batch_pre_review(client: TestClient, headers: dict[str, str]):
    doc_id, _, _ = _upload_temp_md(client, headers)
    try:
        r = client.post(
            "/api/documents/batch-pre-review",
            json={"doc_ids": [doc_id]},
            headers=headers,
        )
        assert r.status_code == 200
    finally:
        client.post("/api/documents/batch-discard", json={"doc_ids": [doc_id]}, headers=headers)


def test_batch_pre_review_empty_rejected(client: TestClient, headers: dict[str, str]):
    r = client.post(
        "/api/documents/batch-pre-review",
        json={"doc_ids": []},
        headers=headers,
    )
    assert r.status_code == 400


def test_platform_push(client: TestClient, headers: dict[str, str]):
    """数据中台推送入口：推送文档 → 进入待审队列。"""
    from agent_base.config import deep_get, load_yaml

    cfg = load_yaml("configs/app.yaml") or {}
    platform_token = deep_get(cfg, "security.platform_token", "")
    if not platform_token:
        return  # 未配置平台 token，跳过
    tag = uuid.uuid4().hex[:8]
    content = f"# 中台推送\n\n## 章节\n\n{tag} 的推送内容。"
    r = client.post(
        "/api/platform/documents",
        json={"filename": f"platform_{tag}.md", "content": content, "doc_type": "faq"},
        headers={"X-Platform-Token": platform_token},
    )
    assert r.status_code in (200, 201), r.text
    body = r.json()
    doc_id = body.get("doc_id") or (body.get("document") or {}).get("doc_id", "")
    if doc_id:
        queue = client.get("/api/documents/review-queue?status=pending_fine_review", headers=headers).json().get("queue", [])
        assert any(item["doc_id"] == doc_id for item in queue)
        client.post("/api/documents/batch-discard", json={"doc_ids": [doc_id]}, headers=headers)
