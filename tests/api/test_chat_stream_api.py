"""对话流式接口契约：请求体校验 / 事件结构（不触发完整 LLM 生成）。"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient


def test_stream_requires_question(client: TestClient):
    """P31 回归：query 不是合法字段 → 422；question 为空 → 422。"""
    r = client.post("/api/ask/stream", json={"query": "测试", "top_k": 4, "rerank": "none"})
    assert r.status_code == 422
    r2 = client.post("/api/ask/stream", json={"question": "", "top_k": 4, "rerank": "none"})
    assert r2.status_code == 422


def test_stream_rejects_bad_rerank(client: TestClient):
    r = client.post("/api/ask/stream", json={"question": "测试", "rerank": "bge-reranker"})
    assert r.status_code == 422  # rerank 值域 auto/keyword/model/none


def test_stream_accepts_valid_request(client: TestClient):
    """合法请求至少能返回事件流（含 sources/trace），不强制跑完 LLM 生成。"""
    with client.stream(
        "POST",
        "/api/ask/stream",
        json={"question": "接口测试问题", "top_k": 4, "rerank": "none", "use_cache": False},
    ) as resp:
        assert resp.status_code == 200
        # 读前几个事件，确认协议字段
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            evt = json.loads(line[6:])
            assert "type" in evt
            if evt["type"] in ("sources", "trace", "memory"):
                break
