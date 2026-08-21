"""契约 P19b：升级版关键词重排（keyword_v2）。

覆盖：
1. 电商词典分词：中文成分/品类词整词命中，英文整词；
2. 同义词/别名扩展（复用 PG alias_rules）；
3. TF-IDF 加权 + 长度归一化（稀有词权重大、长文不占优）；
4. 融合兜底：关键词信号 + 原始向量信号，语义序为主、关键词微调；
5. reranker 兼容：keyword 分支与 model 降级仍写 rerank_strategy=keyword。

全部离线、确定性、零外部依赖。
"""

from __future__ import annotations

from langchain_core.documents import Document

from agent_base.retrieval.keyword_ranker import (
    hybrid_fallback_rank,
    keyword_signal,
    tokenize,
)
from agent_base.retrieval.reranker import keyword_score, rerank_documents


def _docs():
    return [
        Document(
            page_content="花颜集玻尿酸保湿精华液，含玻尿酸，适合油皮干皮，保湿补水。",
    metadata={"section": "商品参数", "source_file": "legacy-source#P001"},
        ),
        Document(
            page_content="白色纯棉T恤，260g 重磅面料，基础百搭款。",
    metadata={"section": "商品参数", "source_file": "legacy-source#P003"},
        ),
        Document(
            page_content="退换货政策：支持 7 天无理由退换货，吊牌完整。",
            metadata={"section": "售后FAQ", "source_file": "FAQ_售后物流.md"},
        ),
    ]


def test_tokenize_dict_terms_and_english():
    """词典整词命中 + 英文整词。"""
    tokens = tokenize("玻尿酸精华适合油皮吗", with_aliases=False)
    # 词典最长匹配：可能是“玻尿酸”或更长的“玻尿酸精华”
    assert any("玻尿酸" in t for t in tokens)
    assert any("精华" in t for t in tokens)
    assert "油皮" in tokens

    tokens2 = tokenize("白色T恤多少钱", with_aliases=False)
    assert "t恤" in tokens2
    assert "多少钱" in tokens2


def test_tokenize_alias_expansion():
    """别名扩展：查询“玻尿酸”会带上标准全称词。"""
    tokens = tokenize("玻尿酸", with_aliases=True)
    assert "玻尿酸" in tokens
    # PG alias_rules 里 玻尿酸 -> 玻尿酸保湿精华液 / 玻尿酸补水面膜
    assert any("保湿精华" in t or "精华" in t for t in tokens)


def test_keyword_signal_ranks_relevant_doc_first():
    """升级版关键词分：相关文档得分显著高于无关文档。"""
    docs = _docs()
    q = "玻尿酸精华适合油皮吗"
    s0 = keyword_signal(q, docs[0])
    s1 = keyword_signal(q, docs[1])
    s2 = keyword_signal(q, docs[2])
    assert s0 > s1
    assert s0 > s2
    assert 0.0 <= s0 <= 1.0


def test_keyword_signal_t_shirt():
    """T恤问题命中服饰文档。"""
    docs = _docs()
    q = "白色T恤多少钱"
    s_clothes = keyword_signal(q, docs[1])
    s_skincare = keyword_signal(q, docs[0])
    assert s_clothes > s_skincare


def test_keyword_signal_price_synonym():
    """同义词扩展：多少钱/贵不贵 能命中含“价格”的文档。"""
    price_doc = Document(
        page_content="产品规格与价格：规格 25ml × 10 片/盒，价格 59 元/盒。",
        metadata={"section": "产品规格与价格"},
    )
    clothes_doc = Document(
        page_content="白色纯棉T恤，260g 重磅面料，通勤百搭。",
        metadata={"section": "商品参数"},
    )
    for q in ("多少钱", "贵不贵"):
        assert keyword_signal(q, price_doc) > keyword_signal(q, clothes_doc)


def test_keyword_signal_how_to_use_synonym():
    """同义词扩展：怎么用 能命中含“使用方法/用量”的文档。"""
    usage_doc = Document(
        page_content="使用方法：早晚洁面后取适量涂抹于面部，用量约一颗黄豆大小。",
        metadata={"section": "使用方法"},
    )
    clothes_doc = Document(
        page_content="白色纯棉T恤，260g 重磅面料，通勤百搭。",
        metadata={"section": "商品参数"},
    )
    assert keyword_signal("怎么用", usage_doc) > keyword_signal("怎么用", clothes_doc)


def test_reranker_keyword_score_delegates():
    """reranker.keyword_score 委托升级实现，签名兼容。"""
    docs = _docs()
    s = keyword_score("玻尿酸精华适合油皮吗", docs[0])
    assert 0.0 <= s <= 1.0
    assert s > keyword_score("玻尿酸精华适合油皮吗", docs[2])


def test_hybrid_fallback_keeps_vector_order_when_keyword_weak():
    """融合兜底：关键词弱匹配时以向量语义序为主，不整体打乱。"""
    d_low_vec = Document(page_content="花颜集玻尿酸精华液 保湿补水", metadata={"section": "商品参数", "vector_score": 0.20})
    d_high_vec = Document(page_content="白色纯棉T恤 260g 重磅面料", metadata={"section": "商品参数", "vector_score": 0.85})
    d_mid_vec = Document(page_content="退换货政策 七天无理由", metadata={"section": "售后FAQ", "vector_score": 0.60})
    out = hybrid_fallback_rank("白色T恤多少钱", [d_low_vec, d_high_vec, d_mid_vec], top_k=3)
    # 语义相关最高的服饰文档应保持第一
    assert out[0] is d_high_vec
    # 元数据兼容：兜底仍标记 keyword
    assert all(d.metadata.get("rerank_strategy") == "keyword" for d in out)
    assert out[0].metadata.get("rerank_score") is not None


def test_hybrid_fallback_without_vector_score_falls_back_pure_keyword():
    """无 vector_score 时融合退化为纯关键词排序。"""
    docs = _docs()
    out = hybrid_fallback_rank("玻尿酸精华适合油皮吗", docs, top_k=2)
    assert out[0] is docs[0]
    assert out[0].metadata.get("rerank_strategy") == "keyword"


def test_model_rerank_falls_back_to_keyword_v2():
    """TEI 不可用时 model 降级走融合兜底，链路不中断（兼容旧契约）。"""
    docs = _docs()
    out = rerank_documents(
        "玻尿酸精华适合油皮吗",
        docs,
        strategy="model",
        top_k=2,
        model_provider="local_tei",
        model_endpoint="http://127.0.0.1:9/rerank",  # 必然失败的端口
        errors=[],
    )
    assert len(out) == 2
    assert all(d.metadata.get("rerank_strategy") == "keyword" for d in out)
    assert out[0].metadata.get("rerank_score") is not None
