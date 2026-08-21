"""系统统计接口：健康 / 向量 / 缓存 / 会话。"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health(client: TestClient):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert "status" in r.json()


def test_stats_vector(client: TestClient, headers: dict[str, str]):
    r = client.get("/api/stats/vector", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert "collections" in body
    cols = body["collections"]
    assert any(k in cols for k in ("ecommerce_chunks", "ecommerce_chunks_sparse", "ecommerce_summaries"))


def test_stats_cache(client: TestClient, headers: dict[str, str]):
    r = client.get("/api/stats/cache", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert "hit_rate" in body and "hits" in body and "misses" in body
    assert 0.0 <= body["hit_rate"] <= 1.0


def test_sessions(client: TestClient, headers: dict[str, str]):
    r = client.get("/api/sessions", headers=headers)
    assert r.status_code == 200
    assert "sessions" in r.json()


def test_stats_failure_events(client: TestClient, headers: dict[str, str]):
    r = client.get("/api/stats/failure-events", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert "events" in body and "total" in body
    assert isinstance(body["events"], list)
    assert body["total"] >= 0


def test_stats_failure_events_requires_admin(client: TestClient):
    r = client.get("/api/stats/failure-events")
    assert r.status_code in (401, 403)
