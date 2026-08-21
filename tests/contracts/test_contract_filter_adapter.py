"""契约 P9-02：Chroma → Qdrant filter 适配单测。"""

from agent_base.retrieval.filter_adapter import chroma_to_qdrant_filter


def test_simple_kv():
    result = chroma_to_qdrant_filter({"section": "用法用量"})
    assert result == {"must": [{"key": "metadata.section", "match": {"value": "用法用量"}}]}


def test_multi_kv():
    result = chroma_to_qdrant_filter({"section": "商品参数", "product_name": "玻尿酸精华"})
    assert len(result["must"]) == 2


def test_in_operator():
    result = chroma_to_qdrant_filter({"section": {"$in": ["商品参数", "成分"]}})
    assert result["must"][0]["match"]["any"] == ["商品参数", "成分"]


def test_ne_operator():
    result = chroma_to_qdrant_filter({"section": {"$ne": "禁用"}})
    assert result["must"][0]["match"]["except"] == ["禁用"]


def test_and_combinator():
    result = chroma_to_qdrant_filter({"$and": [{"section": "商品参数"}, {"product_name": "玻尿酸精华"}]})
    assert len(result["must"]) == 2


def test_or_combinator():
    result = chroma_to_qdrant_filter({"$or": [{"section": "商品参数"}, {"section": "成分"}]})
    assert len(result["should"]) == 2


def test_nested_and_in():
    """true scenario: section $in + product_name + category = $and three conditions"""
    result = chroma_to_qdrant_filter({
        "$and": [
            {"section": {"$in": ["商品参数", "成分", "注意事项"]}},
            {"product_name": "玻尿酸精华"},
            {"product_spec": "保湿精华"},
        ]
    })
    assert len(result["must"]) == 3
    assert result["must"][0]["match"]["any"] is not None
    assert result["must"][1]["match"]["value"] == "玻尿酸精华"


def test_empty_filter():
    assert chroma_to_qdrant_filter({}) is None
    assert chroma_to_qdrant_filter(None) is None


def test_none_value():
    result = chroma_to_qdrant_filter({"section": None})
    assert result is not None
