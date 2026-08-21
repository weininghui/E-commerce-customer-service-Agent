"""意图管理接口：列表 / 详情 / 示例追加 / 版本 / 测试。"""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient


def test_intents_list(client: TestClient, headers: dict[str, str]):
    r = client.get("/api/intents", headers=headers)
    assert r.status_code == 200
    intents = r.json().get("intents", [])
    assert len(intents) >= 1
    for it in intents:
        assert "intent" in it and "keywords" in it


def test_intent_get(client: TestClient, headers: dict[str, str]):
    r = client.get("/api/intents", headers=headers)
    intent_key = r.json()["intents"][0]["intent"]
    r2 = client.get(f"/api/intents/{intent_key}", headers=headers)
    assert r2.status_code == 200
    assert r2.json()["intent"] == intent_key


def test_intent_add_examples_and_versions(client: TestClient, headers: dict[str, str]):
    r = client.get("/api/intents", headers=headers)
    intents = r.json()["intents"]
    target = next((it for it in intents if it["intent"] != "general_qa"), intents[0])
    key = target["intent"]
    example = f"api测试问法_{uuid.uuid4().hex[:6]}"
    try:
        r = client.post(f"/api/intents/{key}/examples", json={"examples": [example]}, headers=headers)
        assert r.status_code == 200
        v = client.get(f"/api/intents/{key}/versions", headers=headers)
        assert v.status_code == 200
        assert "versions" in v.json()
    finally:
        # 清理：把 examples 还原（重传空集不可行，这里只保证不抛错）
        pass


def test_intent_put_save_and_restore(client: TestClient, headers: dict[str, str]):
    """PUT 保存关键词（保存后恢复原值）。"""
    r = client.get("/api/intents", headers=headers)
    intent = r.json()["intents"][0]
    key = intent["intent"]
    original = list(intent.get("keywords") or [])
    try:
        modified = (original + ["api临时词"])[:30]
        r2 = client.put(f"/api/intents/{key}", json={"keywords": modified}, headers=headers)
        assert r2.status_code == 200, r2.text
        after = client.get(f"/api/intents/{key}", headers=headers).json()
        assert "api临时词" in (after.get("keywords") or [])
    finally:
        client.put(f"/api/intents/{key}", json={"keywords": original}, headers=headers)


def test_intent_ai_improve_not_found(client: TestClient, headers: dict[str, str]):
    """AI 优化：不存在的意图 → 404（端点可达）。"""
    r = client.post("/api/intents/no_such_intent_xyz/ai-improve", json={}, headers=headers)
    assert r.status_code == 404


def test_intent_restore_not_found(client: TestClient, headers: dict[str, str]):
    r = client.post("/api/intents/no_such_intent_xyz/restore/1", json={}, headers=headers)
    assert r.status_code == 404


def test_intent_single_test(client: TestClient, headers: dict[str, str]):
    r = client.post("/api/intents/test", json={"question": "这个多少钱"}, headers=headers)
    assert r.status_code == 200
    assert "intent" in r.json()
