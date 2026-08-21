"""目录 / 检索接口：catalog、resolve、retrieve。"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_catalog(client: TestClient, headers: dict[str, str]):
    r = client.get("/api/catalog", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert "products" in body or "categories" in body or "product_count" in body


def test_catalog_resolve(client: TestClient, headers: dict[str, str]):
    r = client.get("/api/catalog/resolve?query=精华液", headers=headers)
    assert r.status_code in (200, 404)  # 识别不到商品返回 404 也算正常


def test_retrieve_returns_sources(client: TestClient, headers: dict[str, str]):
    """检索接口（不生成）：返回来源与 trace。"""
    r = client.post(
        "/api/retrieve",
        json={"question": "玻尿酸精华有什么功效", "top_k": 4, "rerank": "none"},
        headers=headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert "sources" in body or "trace" in body or "results" in body
