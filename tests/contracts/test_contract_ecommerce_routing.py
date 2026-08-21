"""契约（电商版）：意图路由 / 检索策略 / catalog 商品约束解析 / 合规评估。"""


from agent_base.embeddings import build_embeddings
from agent_base.graphs import build_rag_graph
from agent_base.vectorstore import build_vector_store
from agent_base.chains.safety_chain import assess_safety
from agent_base.domain import load_domain
from agent_base.indexing.metadata_index import resolve_query_constraints
from agent_base.retrieval.intent_router import route_question
from agent_base.retrieval.retrieval_policy import build_retrieval_decision


def _domain():
    return load_domain("ecommerce")


def test_ecommerce_intent_routing():
    domain = _domain()
    assert route_question("玻尿酸精华适合敏感肌吗", domain=domain).intent == "product_query"
    assert route_question("白T恤怎么搭配通勤", domain=domain).intent == "fashion_query"
    assert route_question("这件多少钱", domain=domain).intent == "price_query"
    assert route_question("怎么退货", domain=domain).intent == "aftersale"
    assert route_question("适合油皮推荐一下", domain=domain).intent == "recommendation"
    assert route_question("今天天气怎么样", domain=domain).intent == "general_qa"


def test_ecommerce_domain_default():
    # 不传 domain 时自动加载电商
    assert route_question("怎么退货").intent == "aftersale"


def test_retrieval_policy_strategies():
    _, decision = build_retrieval_decision("怎么退货", category="服饰-上衣")
    assert decision.strategy == "metadata_first"

    _, decision2 = build_retrieval_decision("玻尿酸精华适合敏感肌吗", category="精华")
    assert decision2.strategy == "hybrid"
    assert decision2.metadata_filter == {"$and": [{"section": {"$in": ["商品参数", "卖点"]}}, {"category": "精华"}]}

    # P12: "衣服" keyword now correctly routes to fashion_query → hybrid
    _, decision3 = build_retrieval_decision("这件衣服怎么样")
    assert decision3.strategy == "hybrid"
    assert decision3.need_clarification is False

    # True general_qa (no keyword match) still gets summary_guided_hybrid
    _, decision4 = build_retrieval_decision("今天天气怎么样")
    assert decision4.strategy == "summary_guided_hybrid"


def test_catalog_resolution_ecommerce():
    from agent_base.api.main import get_catalog

    catalog = get_catalog()
    resolution = resolve_query_constraints(catalog, "玻尿酸保湿精华液适合油皮吗")
    assert resolution.product_name == "玻尿酸保湿精华液"
    assert resolution.category == "精华"
    assert resolution.ambiguous is False


def test_safety_ecommerce_rules():
    assessment = assess_safety("用了会不会过敏", "注意事项：敏感肌使用前建议局部测试")
    assert assessment.risk_level == "high"
    assert assessment.must_consult is True
    assert any("过敏" in w for w in assessment.warnings)

    low = assess_safety("这款面霜保湿效果怎么样")
    assert low.risk_level == "low"


def test_ecommerce_catalog_structure():
    from agent_base.api.main import get_catalog

    catalog = get_catalog()
    assert "products" in catalog and "product_count" in catalog
    assert isinstance(catalog["products"], dict)
    assert catalog["product_count"] >= 20


def test_graph_smoke_ecommerce(tmp_path):
    """LangGraph 全链路冒烟：电商意图 → 检索 → 安全 → 模板生成（无外部依赖）。"""
    emb = build_embeddings(provider="hash", dimensions=512)
    vs = build_vector_store(
        provider="chroma",
        persist_dir=str(tmp_path),
        collection="ecommerce_chunks",
        embedding_function=emb,
    )
    graph = build_rag_graph(
        vector_store=vs,
        summary_store=None,
        rerank_cfg={"provider": "none"},
        llm_cfg={"provider": "none"},
    )
    result = graph.invoke(
        {"question": "怎么退货", "category": "服饰-上衣", "errors": []},
        {"configurable": {"thread_id": "smoke"}},
    )
    assert result.get("route", {}).get("intent") == "aftersale"
    assert isinstance(result.get("answer", ""), str)
    assert "商品资料" in result.get("answer", "") or "客服" in result.get("answer", "")
