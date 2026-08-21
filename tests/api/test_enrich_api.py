"""T4：Enrich 别名扩展接入测试（enrich_alias 启用 → 检索查询含标准商品名）。"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from agent_base.retrieval.enrich import expand_aliases
from agent_base.chains.streaming import _extract_current_product


def test_expand_aliases_direct():
    """别名直接扩展：玻尿酸精华 → 追加标准商品名。"""
    out = expand_aliases("玻尿酸精华有什么功效")
    assert "玻尿酸保湿精华液" in out
    assert "玻尿酸精华" in out  # 原始词保留


def test_expand_aliases_no_match_unchanged():
    """无别名命中：原问题不变。"""
    out = expand_aliases("今天天气怎么样")
    assert out == "今天天气怎么样"


def test_retrieve_search_query_enriched(client: TestClient, headers: dict[str, str]):
    """配置启用时：检索 trace 的 search_query 含标准商品名。"""
    r = client.post(
        "/api/retrieve",
        json={"question": "玻尿酸精华适合敏感肌吗", "top_k": 4, "rerank": "none"},
        headers=headers,
    )
    assert r.status_code == 200
    trace = r.json()["trace"]
    search_query = trace.get("search_query", "")
    assert "玻尿酸保湿精华液" in search_query, f"search_query 未 enrich: {search_query}"


def test_retrieve_enrich_disabled_unchanged(client: TestClient, headers: dict[str, str]):
    """配置关闭时：search_query 不 enrich（回归保护）。"""
    with patch(
        "agent_base.config.deep_get",
        return_value=False,
    ):
        r = client.post(
            "/api/retrieve",
            json={"question": "玻尿酸精华适合敏感肌吗", "top_k": 4, "rerank": "none"},
            headers=headers,
        )
        assert r.status_code == 200
        trace = r.json()["trace"]
        search_query = trace.get("search_query", "")
        assert "玻尿酸保湿精华液" not in search_query


def test_extract_current_product_from_history():
    """从历史用户消息提取商品标准名（最近优先）。"""
    history = [
        {"role": "user", "content": "玻尿酸精华有什么功效"},
        {"role": "assistant", "content": "玻尿酸保湿精华液主打保湿"},
        {"role": "user", "content": "它适合敏感肌吗"},
    ]
    assert _extract_current_product(history) == "玻尿酸保湿精华液"


def test_extract_current_product_empty():
    """无商品匹配历史 → None。"""
    history = [
        {"role": "user", "content": "今天天气怎么样"},
        {"role": "assistant", "content": "不清楚"},
    ]
    assert _extract_current_product(history) is None


def test_resolve_question_with_product():
    """指代补全：有 current_product 时短指代问题前补商品名。"""
    from agent_base.retrieval.enrich import resolve_question

    out = resolve_question("它多少钱", current_product="玻尿酸保湿精华液")
    assert out.startswith("玻尿酸保湿精华液")
