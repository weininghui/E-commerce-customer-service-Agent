"""检索策略决策（电商域）。

根据意图路由结果选择自动检索策略：
- 售后 FAQ 类问题：metadata_first（filter 精确、量少，按插入顺序即可）。
- 商品/穿搭/价格/推荐类问题：hybrid（metadata + vector，兼顾精确过滤和语义召回）。
- 泛问答：summary_guided_hybrid（先用摘要定位章节，再回原文取证）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from agent_base.retrieval.intent_router import build_metadata_filter
from agent_base.retrieval.query_rewriter import QueryRewrite, build_query_rewrite


AUTO_RERANK = "auto"

# 固定章节意图：答案通常位于固定 FAQ 章节，优先 metadata 精确过滤
FIXED_SECTION_INTENTS = {"aftersale"}
# 需要多路召回（metadata + summary + vector）的意图
HYBRID_INTENTS = {"product_query", "fashion_query", "price_query", "recommendation", "comparison", "size_recommendation", "promotion"}
# 需商品澄清的意图子集：缺少具体商品时不应盲目检索（comparison 天然多商品、promotion 无单商品依赖，不触发）
CLARIFICATION_INTENTS = {"product_query", "fashion_query", "price_query", "recommendation", "size_recommendation"}

# BUG-27: 意图中文名（决策理由模板用，避免英文意图名泄漏到前端 Trace）
INTENT_LABELS: dict[str, str] = {
    "product_query": "商品咨询",
    "fashion_query": "穿搭咨询",
    "price_query": "价格咨询",
    "aftersale": "售后咨询",
    "recommendation": "推荐咨询",
    "comparison": "对比咨询",
    "size_recommendation": "尺码推荐",
    "promotion": "促销咨询",
    "general_qa": "通用问答",
    "emotion_handoff": "情绪转人工",
}


def _intent_label(intent: str) -> str:
    """意图英文 key → 中文标签（未收录时原样返回）。"""
    return INTENT_LABELS.get(intent, intent)


@dataclass(slots=True)
class RetrievalDecision:
    """一次检索的完整决策：意图、策略、过滤条件与各通道开关。

    由 build_retrieval_decision 根据意图路由结果生成，
    检索器按此决策决定走 metadata / summary / vector 的哪些组合。
    """

    intent: str
    strategy: str
    metadata_filter: dict[str, Any]
    rewritten_query: str
    candidate_k: int
    final_k: int
    use_metadata: bool
    use_summary: bool
    use_vector: bool
    rerank: str
    reason: str
    need_clarification: bool = False
    clarification_question: str = ""
    route_type: str = "answerable"

    def to_dict(self) -> dict[str, Any]:
        """转为 dict，便于写入 trace 或日志。

        Returns:
            与字段同名的字典。
        """
        return asdict(self)


def build_retrieval_decision(
    question: str,
    product_name: str | None = None,
    product_spec: str | None = None,
    category: str | None = None,
    current_product: str | None = None,
    top_k: int = 6,
    candidate_k: int | None = None,
    rerank: str = AUTO_RERANK,
    use_rewrite: bool = True,
    intent_classifier: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
) -> tuple[QueryRewrite, RetrievalDecision]:
    """根据意图选择自动检索策略（电商版）。

    - 售后 FAQ 问题（aftersale）走 metadata_first。
    - 商品/穿搭/价格/推荐类问题走 hybrid：带类目等 metadata 过滤 + 向量语义召回。
    - 泛问答走 summary_guided_hybrid：摘要定位章节 → 回原文取证 → 向量兜底。

    Args:
        question: 用户问题。
        product_name: 商品名约束（catalog 解析或显式传入）。
        product_spec: 商品规格/别名约束。
        category: 商品类目约束。
        top_k: 最终返回证据数。
        candidate_k: 候选数（不传则按策略取默认）。
        rerank: 重排策略（auto/keyword/model）。
        use_rewrite: 是否使用改写后查询。
        intent_classifier: LLM 意图增强配置（可选）。
        profile: 用户画像字典（意图层消解缺失需求用）。
        current_product: 会话当前商品名（用于指代补全与完整性分级）。

    Returns:
        (QueryRewrite, RetrievalDecision)。
    """
    rewrite = build_query_rewrite(
        question,
        product_name=product_name,
        product_spec=product_spec,
        intent_classifier=intent_classifier,
        profile=profile,
    )
    metadata_filter = build_metadata_filter(
        rewrite.route,
        product_name=product_name,
        product_spec=product_spec,
        category=category,
    )
    intent = rewrite.route.intent
    final_k = max(1, top_k)
    chosen_rerank = "keyword" if rerank in {AUTO_RERANK, "", None} else rerank
    rewritten_query = rewrite.rewritten_question if use_rewrite else question

    # P32a + P32c：商品意图 + 无商品约束 → 按问题完整性分级
    # - 必须澄清：含指示代词或极短无匹配 → strategy=clarification
    # - 可降级推荐：无指示代词、未指定商品但语义明确（如"敏感肌用什么"）
    #   → 正常 hybrid 检索 + route_type="recommendation"（多商品推荐式回答）
    completeness = _classify_question_completeness(
        question, intent, current_product, category_dim=rewrite.route.category_dim
    )
    if (
        intent in CLARIFICATION_INTENTS
        and not product_name
        and not product_spec
        and completeness != "answerable"
    ):
        if completeness == "clarify":
            clarification_q = _build_clarification_candidates(rewrite.route)
            return rewrite, RetrievalDecision(
                intent=intent,
                strategy="clarification",
                metadata_filter={},
                rewritten_query=question,
                candidate_k=0,
                final_k=0,
                use_metadata=False,
                use_summary=False,
                use_vector=False,
                rerank="keyword",
                reason="缺少具体商品约束，需用户澄清目标商品后再检索",
                need_clarification=True,
                clarification_question=clarification_q,
                route_type="clarification_required",
            )
        # completeness == "recommend"：可降级推荐
        strategy = "hybrid"
        default_candidate_k = max(final_k * 2, 10)
        return rewrite, RetrievalDecision(
            intent=intent,
            strategy=strategy,
            metadata_filter=metadata_filter,
            rewritten_query=rewritten_query,
            candidate_k=candidate_k or default_candidate_k,
            final_k=final_k,
            use_metadata=bool(metadata_filter),
            use_summary=True,
            use_vector=True,
            rerank=chosen_rerank,
            reason=(
                f"问题命中{_intent_label(intent)}意图但未指定具体商品，"
                "走多商品召回 + 推荐式回答，不编造单商品结论。"
            ),
            route_type="recommendation",
        )

    if intent in FIXED_SECTION_INTENTS:
        strategy = "metadata_first"
        default_candidate_k = max(final_k + 2, 8)
        use_metadata = True
        use_summary = False
        use_vector = True
        reason = (
            f"问题命中{_intent_label(intent)}意图，答案通常位于固定售后 FAQ 章节；"
            "优先用章节 + 商品/类目元数据精确过滤，不足时向量兜底。"
        )
    elif intent in HYBRID_INTENTS:
        strategy = "hybrid"
        default_candidate_k = max(final_k * 2, 10)
        use_metadata = bool(metadata_filter)
        use_summary = True
        use_vector = True
        reason = (
            f"问题命中{_intent_label(intent)}意图，用类目等元数据过滤 + 摘要定位 + 向量语义召回，"
            "兼顾精确性和覆盖面。"
        )
    else:
        strategy = "summary_guided_hybrid"
        default_candidate_k = max(final_k * 2, 10)
        use_metadata = False
        use_summary = True
        use_vector = True
        reason = "问题未命中明确章节，先用摘要索引定位相关商品/FAQ 章节，再回原文取证并用向量检索兜底。"

    return rewrite, RetrievalDecision(
        intent=intent,
        strategy=strategy,
        metadata_filter=metadata_filter,
        rewritten_query=rewritten_query,
        candidate_k=candidate_k or default_candidate_k,
        final_k=final_k,
        use_metadata=use_metadata,
        use_summary=use_summary,
        use_vector=use_vector,
        rerank=chosen_rerank,
        reason=reason,
        route_type="answerable",
    )


def _load_catalog() -> dict[str, Any] | None:
    """加载商品 catalog（纯 PG 运行时数据源，JSON 文件已淘汰）。"""
    try:
        from agent_base.storage.pg import _conn

        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, name, brand, category, price_band, metadata FROM catalog")
            rows = cur.fetchall()
        if rows:
            products: dict[str, Any] = {}
            for rid, name, brand, category, price_band, md in rows:
                base: dict[str, Any] = {
                    "name": name,
                    "brand": brand or "",
                    "category": category or "",
                    "price_band": price_band or "",
                }
                if md and isinstance(md, dict):
                    for k, v in md.items():
                        if k not in base:
                            base[k] = v
                products[str(rid)] = base
            return {"product_count": len(products), "products": products}
    except Exception:
        pass
    return None


# 指示代词——问题含这些词但未跟具体商品名时很可能缺少约束
_VAGUE_REFERENCES = ["这款", "这个商品", "这个产品", "那个商品", "那个产品", "哪个好", "哪款好"]


def _classify_question_completeness(
    question: str,
    intent: str,
    current_product: str | None = None,
    category_dim: str = "",
) -> str:
    """分类问题完整性，区分"必须澄清"和"可降级推荐"。

    P32c：不是所有缺商品名的问题都澄清。语义明确但未指定具体商品时，
    走多商品推荐式回答比追问"您想了解哪款"体验更好。

    Returns:
        "clarify"    — 必须澄清（含指示代词或极短无匹配）
        "recommend"  — 可降级推荐（无指示代词、未指定商品但语义明确）
        "answerable" — 可直接回答（已含商品名或其他）

    触发"clarify"（满足任一）：
    1. 问题含指示代词（"这款/这个/那个"）但未说明具体商品
    2. 问题很短（≤5 字）且 catalog 匹配不到

    触发"recommend"：
    - 意图在 CLARIFICATION_INTENTS 中，无指示代词，无 product 约束，
      但问题语义完整（如"敏感肌用什么"、"有什么精华推荐"）
    """
    # BUG-12：会话历史已锚定当前商品（current_product）时，
    # 指示代词（"这款/它"）或极短问题视为有明确指代对象，直接可答，不再澄清；
    # 属性类追问（"有什么注意事项"）同样视为对当前商品的补充提问
    if current_product and (
        any(ref in question for ref in _VAGUE_REFERENCES)
        or len(question) <= 5
        or (
            any(p in question for p in _PRODUCT_ATTRIBUTE_PATTERNS)
            and not _looks_like_recommendation(question)
        )
    ):
        return "answerable"
    # 条件 1：明确的模糊引用 → 必须澄清
    if any(ref in question for ref in _VAGUE_REFERENCES):
        return "clarify"
    # 条件 2：极短问题（≤5 字），catalog 解析不出 → 必须澄清
    if len(question) <= 5:
        catalog = _load_catalog()
        if catalog is not None:
            try:
                from agent_base.indexing.metadata_index import resolve_query_constraints
                resolution = resolve_query_constraints(catalog, question)
                if not resolution.product_name and not resolution.product_spec:
                    return "clarify"
            except Exception:
                return "clarify"
        else:
            return "clarify"
    # 条件 3：意图为商品类但无指示代词、无 catalog 精确匹配
    # 区分"推荐式"（"用什么/推荐/有哪些"）和"指定商品"（"玻尿酸精华适合..."）
    # catalog 无法模糊匹配"玻尿酸精华"→"玻尿酸保湿精华液"，用推荐模式词兜底
    if intent in CLARIFICATION_INTENTS:
        catalog = _load_catalog()
        catalog_matched = False
        if catalog is not None:
            try:
                from agent_base.indexing.metadata_index import resolve_query_constraints
                resolution = resolve_query_constraints(catalog, question)
                catalog_matched = bool(resolution.product_name or resolution.product_spec)
            except Exception:
                pass
        if not catalog_matched:
            if _looks_like_recommendation(question):
                # 未指明品类（美妆/服饰）→ 先问品类，而不是直接列商品
                if not category_dim:
                    return "clarify"
                return "recommend"
            # 属性类问题未指名商品（"要注意什么事项"）→ 必须澄清目标商品；
            # 但问题已含商品名词（"玻尿酸精华有什么注意事项"）→ 视为已指名，交给检索
            if any(p in question for p in _PRODUCT_ATTRIBUTE_PATTERNS):
                if any(n in question for n in _PRODUCT_NOUNS):
                    return "answerable"
                return "clarify"
            # 问题可能包含 catalog 无法精确匹配的商品名（如简称），
            # 但有非推荐模式的完整语义 → 让检索层按正常 hybrid 处理
            return "answerable"
    return "answerable"


# 商品属性类问法——问了"某商品的某属性"但没指名商品（注意事项/成分/价格/尺码...）
# 这类问题必须澄清是哪个商品，否则检索会拿任意商品的内容瞎答
_PRODUCT_ATTRIBUTE_PATTERNS = [
    "注意", "禁忌", "成分", "功效", "价格", "多少钱", "价位", "尺码",
    "面料", "版型", "怎么洗", "清洗", "洗涤", "保养", "用法", "用量",
    "副作用", "慎用", "褪色", "缩水", "变形", "起球", "透光",
]

# 常见商品名词——问题含这些词说明大概率已指名商品类别，不应澄清
_PRODUCT_NOUNS = [
    "精华", "面霜", "眼霜", "洁面", "面膜", "防晒", "润肤油", "乳液", "水乳",
    "精华液", "爽肤水", "身体乳", "洗发水", "沐浴露", "口红", "粉底",
    "T恤", "裤子", "裙子", "衬衫", "外套", "卫衣", "连衣裙", "阔腿裤",
    "卫衣", "针织衫", "毛衣", "帽子", "鞋", "背心", "短裤",
]


# 推荐模式词——问题含这些模式更可能是"求推荐"而非"指定商品问详情"
_RECOMMENDATION_PATTERNS = ["用什么", "推荐", "有哪些", "哪种好", "什么.*适合", "怎么选", "买哪个"]


def _looks_like_recommendation(question: str) -> bool:
    """判断问题是否是推荐类问法（非指定商品）。"""
    import re
    for pattern in _RECOMMENDATION_PATTERNS:
        if re.search(pattern, question):
            return True
    return False


_BEAUTY_NAME_WORDS = (
    "精华", "面霜", "洁面", "洗面", "防晒", "眼霜", "面膜", "润肤油",
    "乳液", "水乳", "爽肤水", "身体乳", "精华液", "口红", "粉底", "水",
    "乳",
)

_FASHION_NAME_WORDS = (
    "T恤", "衬衫", "裤", "裙", "针织", "毛衣", "外套", "卫衣", "帽",
    "鞋", "背心", "防晒衣", "套装",
)


def _build_clarification_candidates(route: Any) -> str:
    """构造澄清追问：未指明品类先问美妆/服饰；已指明品类只列该品类候选。

    Args:
        route: 意图路由（含 category_dim）。

    Returns:
        澄清追问文本。
    """
    category_dim = getattr(route, "category_dim", "") or ""
    # 未指明品类：先问方向，不直接列商品
    if not category_dim:
        return (
            "您想了解哪款产品呢？先看美妆还是服饰——美妆比如精华、面霜、防晒，"
            "服饰比如 T恤、连衣裙、阔腿裤。告诉我方向，我帮您挑～"
        )
    catalog = _load_catalog()
    if catalog is None:
        return "您想了解哪款产品呢？先看美妆还是服饰，告诉我方向，我帮您挑～"
    products = catalog.get("products", {})
    words = _BEAUTY_NAME_WORDS if category_dim == "beauty" else _FASHION_NAME_WORDS
    product_names: list[str] = []
    for pid, info in products.items():
        name = ""
        if isinstance(info, dict):
            name = str(info.get("name") or "").strip()
        elif isinstance(info, str):
            name = info.strip()
        if not name:
            name = str(pid)
        if not any(w in name for w in words):
            continue
        product_names.append(name)
        if len(product_names) >= 5:
            break
    if product_names:
        names = "、".join(product_names)
        label = "美妆" if category_dim == "beauty" else "服饰"
        return (
            f"您想看哪款{label}呢？目前可咨询：{names}。"
            f"您可以说出具体名称，我为您详细介绍。"
        )
    return "您想了解哪款产品呢？先看美妆还是服饰，告诉我方向，我帮您挑～"
