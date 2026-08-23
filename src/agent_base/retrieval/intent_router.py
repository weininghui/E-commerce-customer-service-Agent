"""Intent router (ecommerce domain, P12 three-layer version).

Three layers:
  1. Rule layer (keyword scoring): fast, explainable, baseline.
  2. LLM enhancement layer: activates when rule misses or confidence < threshold.
  3. Semantic fallback layer: few-shot example overlap matching.

Each layer marks route.source as "rule" / "llm" / "semantic" for traceability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

KNOWN_INTENTS = {
    "product_query", "fashion_query", "price_query", "aftersale",
    "recommendation", "comparison", "size_recommendation", "promotion",
    "general_qa",
}

INTENT_SECTIONS: dict[str, list[str]] = {
    "product_query": ["商品参数", "卖点"],
    "fashion_query": ["商品参数", "卖点", "搭配建议", "使用要点", "服饰洗涤护理常见问题"],
    "price_query": ["价格"],
    "aftersale": ["售后FAQ"],
    "recommendation": ["商品参数", "卖点", "评价"],
    "comparison": ["商品参数", "卖点", "评价"],
    "size_recommendation": ["商品参数", "卖点", "搭配建议"],
    "promotion": ["价格", "售后FAQ"],
    "general_qa": [],
}

RULE_CONFIDENCE_THRESHOLD = 0.5


@dataclass(slots=True)
class IntentRule:
    """一条意图规则：关键词 + 章节 + few-shot 示例 + 优先级。"""

    intent: str
    keywords: list[str]
    sections: list[str]
    examples: list[str] = field(default_factory=list)
    priority: float = 1.0


@dataclass(slots=True)
class QueryRoute:
    """意图路由结果：命中的意图、章节、过滤条件与置信度。

    source 字段标记识别来源（rule / llm / semantic），用于 trace 可观测性。
    """

    intent: str
    sections: list[str]
    metadata_filter: dict[str, Any]
    matched_keywords: list[str]
    confidence: float
    scores: dict[str, float]
    source: str = "rule"
    fallback_reason: str = ""
    # ── 会话级意图 v2 扩展字段 ──
    sub_intent: str = ""
    """子意图（chat/price_negotiation/price_inquiry/return_exchange/logistics/...）。"""

    buying_signal: str = "normal"
    """购买信号：normal / buying / objection。"""

    objection_type: str = "none"
    """异议类型：price / hesitant / risk / none。"""

    missing_info: list[str] = field(default_factory=list)
    """缺失需求信息（skin_type/budget/scene/product/size），供导购挖需使用。"""

    category_dim: str = ""
    """品类维度：beauty（美妆）/ fashion（服饰）/ ""（未指明），用于品类优先澄清。"""

    def to_dict(self) -> dict[str, Any]:
        """转为 dict，便于 trace / 调试面板展示。

        Returns:
            与字段同名的字典。
        """
        return asdict(self)


def route_question(
    question: str,
    domain: Any | None = None,
    intent_classifier_cfg: dict[str, Any] | None = None,
    embeddings: Any | None = None,
    profile: dict[str, Any] | None = None,
) -> QueryRoute:
    """三层意图识别：规则 → LLM 增强 → 语义兜底。

    Args:
        question: 用户问题。
        domain: DomainAdapter 实例；None 时自动加载 ecommerce。
        intent_classifier_cfg: intent_classifier 配置段（enabled/model 等）。
        embeddings: 语义兜底编码器（生产 bge-m3；None 且未开启语义层时跳过）。
        profile: 用户画像字典（skin_type/price_band/size 等），用于消解缺失需求。

    Returns:
        带 source 标记识别层级的 QueryRoute。
    """
    normalized = question.strip().lower()
    _cfg = intent_classifier_cfg or {}
    _semantic_cfg = _cfg.get("semantic") or {}
    semantic_enabled = bool(_semantic_cfg.get("enabled", False))
    semantic_threshold = float(_semantic_cfg.get("threshold", 0.55))
    semantic_embeddings = embeddings
    if semantic_embeddings is None and semantic_enabled:
        semantic_embeddings = _default_embeddings()

# 未提供时自动加载 ecommerce 域
    if domain is None:
        try:
            from agent_base.domain import load_domain
            domain = load_domain("ecommerce")
        except Exception:
            pass

    if domain is not None:
        domain_intents = getattr(domain, "intents", {}) or {}
        if domain_intents:
            def _rule_priority(name: str) -> float:
                if (
                    name == "recommendation"
                    and any(p in normalized for p in _RECOMMEND_PATTERNS)
                ):
                    return 1.5
                if (
                    name == "size_recommendation"
                    and any(p in normalized for p in _SIZE_BOOST_PATTERNS)
                ):
                    return 1.5
                if (
                    name == "comparison"
                    and ("和" in normalized or "与" in normalized)
                    and any(p in normalized for p in _COMPARE_PATTERNS)
                ):
                    return 1.5
                return 1.0

            domain_rules = [
                IntentRule(
                    intent=name,
                    keywords=cfg.get("keywords", []),
                    sections=cfg.get("sections", []),
                    examples=cfg.get("examples", []),
                    priority=_rule_priority(name),
                )
                for name, cfg in domain_intents.items()
            ]
            scored_domain = [_score_rule(normalized, rule) for rule in domain_rules]
            scored_domain = [s for s in scored_domain if s[0] > 0]
            if scored_domain:
                scored_domain.sort(key=lambda item: item[0], reverse=True)
                best_score, best_rule, best_keywords = scored_domain[0]
                total_score = sum(s for s, _, _ in scored_domain)
                confidence = round(best_score / total_score, 3) if total_score else 0.0
                scores = {rule.intent: round(s, 3) for s, rule, _ in scored_domain}
                rule_route = QueryRoute(
                    intent=best_rule.intent,
                    sections=list(best_rule.sections),
                    metadata_filter=_section_filter(best_rule.sections),
                    matched_keywords=best_keywords,
                    confidence=confidence,
                    scores=scores,
                    source="rule",
                )
                _enrich_route(rule_route, question, profile=profile)
# 低置信度 → 尝试 LLM 层
                cfg = intent_classifier_cfg or {}
                if confidence < RULE_CONFIDENCE_THRESHOLD and cfg.get("enabled"):
                    llm_route = _try_llm_layer(question, rule_route, best_rule, cfg)
                    if llm_route is not None:
                        return llm_route
# LLM 不可用 → 尝试语义兜底
                    semantic_route = _try_semantic_layer(
                        normalized, rule_route, domain_rules,
                        embeddings=semantic_embeddings,
                        threshold=semantic_threshold,
                        profile=profile,
                    )
                    if semantic_route is not None:
                        return semantic_route
                return rule_route

# 规则层零命中 → 尝试 LLM 层
            cfg = intent_classifier_cfg or {}
            if cfg.get("enabled"):
                empty_route = QueryRoute(
                    intent="general_qa", sections=[], metadata_filter={},
                    matched_keywords=[], confidence=0.0, scores={},
                    source="rule", fallback_reason="no_keyword_match",
                )
                llm_route = _try_llm_layer(question, empty_route,
                                          IntentRule("general_qa", [], []), cfg)
                if llm_route is not None and llm_route.intent != "general_qa":
                    return llm_route
# 语义兜底
            semantic_route = _try_semantic_layer(normalized,
                QueryRoute(intent="general_qa", sections=[], metadata_filter={},
                          matched_keywords=[], confidence=0.0, scores={}, source="rule"),
                domain_rules, embeddings=semantic_embeddings, threshold=semantic_threshold,
                profile=profile)
            if semantic_route is not None:
                return semantic_route

# 三层全部未命中：回退 general_qa
    fallback = QueryRoute(
        intent="general_qa",
        sections=[],
        metadata_filter={},
        matched_keywords=[],
        confidence=0.0,
        scores={},
        source="rule",
        fallback_reason="no_domain_rules",
    )
    _enrich_route(fallback, question, profile=profile)
    return fallback


def _try_llm_layer(
    question: str,
    rule_route: QueryRoute,
    rule: IntentRule,
    cfg: dict[str, Any],
) -> QueryRoute | None:
    """LLM 增强层：用 DeepSeek-flash 分类意图。

    用 KNOWN_INTENTS 校验 LLM 输出，非法输出直接拒绝。

    Args:
        question: 用户问题。
        rule_route: 规则路由结果（供 LLM prompt 上下文）。
        rule: 最佳匹配的 IntentRule（含示例）。
        cfg: intent_classifier 配置段。

    Returns:
        New QueryRoute (source="llm"), or None if LLM unavailable/invalid.
    """
    try:
        from agent_base.retrieval.llm_intent_classifier import route_question_with_llm
        llm_route = route_question_with_llm(
            question,
            rule_route,
            provider=cfg.get("provider", "langchain"),
            model=cfg.get("model", "deepseek-v4-flash"),
            base_url=cfg.get("base_url"),
            api_key_env=cfg.get("api_key_env", "ANTHROPIC_AUTH_TOKEN"),
            temperature=float(cfg.get("temperature", 0.0)),
        )
# 校验：LLM 输出必须在已知意图集合内
        if llm_route.intent not in KNOWN_INTENTS:
            return None
        if llm_route.source == "llm":
            return llm_route
    except Exception:
        pass
    return None


def _try_semantic_layer(
    normalized_question: str,
    fallback_route: QueryRoute,
    domain_rules: list[IntentRule],
    embeddings: Any | None = None,
    threshold: float = 0.55,
    profile: dict[str, Any] | None = None,
) -> QueryRoute | None:
    """语义兜底：查询与各意图 few-shot 示例做向量相似度匹配。

    复用现有 embedding（生产 bge-m3，测试可注入确定性桩），
    取与示例的最大余弦相似度，超过阈值才接受匹配；embedding 不可用
    或全部低于阈值时返回 None（交给调用方回退 general_qa）。

    Args:
        normalized_question: 小写并去除首尾空白的问题。
        fallback_route: 当前兜底路由。
        domain_rules: Domain intent rules (with examples).
        embeddings: 实现了 embed_documents / embed_query 的编码器；None 时跳过。
        threshold: 接受匹配的最低余弦相似度。
        profile: 用户画像字典（消解缺失需求用）。

    Returns:
        QueryRoute (source="semantic"), or None if no sufficient match.
    """
    if embeddings is None or not domain_rules:
        return None
    try:
        examples: list[tuple[str, IntentRule]] = [
            (example, rule)
            for rule in domain_rules
            for example in rule.examples
            if example.strip()
        ]
        if not examples:
            return None
        query_vec = embeddings.embed_query(normalized_question)
        doc_vecs = embeddings.embed_documents([text for text, _ in examples])
        best_sim = 0.0
        best_rule: IntentRule | None = None
        best_example = ""
        for (example, rule), vec in zip(examples, doc_vecs):
            sim = _cosine(query_vec, vec)
            if sim > best_sim:
                best_sim = sim
                best_rule = rule
                best_example = example
    except Exception:
        return None

    if best_sim >= threshold and best_rule is not None and best_rule.intent in KNOWN_INTENTS:
        route = QueryRoute(
            intent=best_rule.intent,
            sections=list(best_rule.sections),
            metadata_filter=_section_filter(best_rule.sections),
            matched_keywords=[],
            confidence=min(0.5, best_sim),
            scores={"semantic_max_sim": round(best_sim, 3), "semantic_example": best_example[:40]},
            source="semantic",
        )
        _enrich_route(route, normalized_question, profile=profile)
        return route

# 语义匹配不足：交给调用方回退 general_qa
    return None


def _cosine(a: list[float], b: list[float]) -> float:
    """向量余弦相似度（维度不一致/空向量返回 0）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(y * y for y in b) ** 0.5 or 1.0
    return dot / (na * nb)


def _default_embeddings() -> Any | None:
    """按 app.yaml embedding 配置构建编码器（生产 bge-m3）；失败返回 None。"""
    try:
        from agent_base.config import load_yaml
        from agent_base.embeddings import build_embeddings

        cfg = load_yaml("configs/app.yaml") or {}
        emb = cfg.get("embedding") or {}
        return build_embeddings(
            provider=emb.get("provider", "hash"),
            model=emb.get("model"),
            dimensions=emb.get("dimensions"),
            base_url=emb.get("base_url"),
            api_key_env=emb.get("api_key_env", "DASHSCOPE_API_KEY"),
            keep_alive=emb.get("keep_alive"),
        )
    except Exception:
        return None


# ── 会话级意图 v2：子意图 / 购买信号 / 缺失信息 ──

_CHAT_PATTERNS = (
    "你好", "您好", "hello", "hi", "在吗", "谢谢", "感谢", "再见", "拜拜",
    "辛苦了", "哈哈", "加油", "晚安", "早安", "没事",
)

MEDIA_REQUEST_PATTERNS = (
    "图片", "照片", "实拍", "实物图", "实物", "上身图", "上身效果", "效果图",
    "视频", "看看图", "看图", "发图", "发个图", "有图吗", "有视频", "发视频",
    "发个视频", "展示一下", "细节图", "高清图", "大图", "看下效果", "看看效果",
    "看实物", "看看实物", "有没有视频", "有图片",
)

_NEGOTIATION_PATTERNS = (
    "太贵", "贵了", "有点贵", "能便宜", "便宜点", "优惠点", "砍价",
    "降价", "划不划算", "值不值", "性价比",
)

_PRICE_INQUIRY_PATTERNS = ("多少钱", "什么价", "价位", "价格", "贵吗")

_RETURN_PATTERNS = ("退货", "退款", "换货", "退换", "七天无理由", "退掉")

_LOGISTICS_PATTERNS = ("物流", "发货", "快递", "到货", "签收", "运单", "催单")

_ALLERGY_PATTERNS = ("过敏", "敏感", "刺激", "慎用", "副作用", "孕妇")
_ALLERGY_COMPAT_MARKERS = ("能用吗", "能用", "可以用吗", "能用不", "可以用", "适用", "慎用", "副作用", "刺激吗")
_ALLERGY_SUBJECT_MARKERS = ("敏感", "刺激", "酒精", "水杨酸", "烟酰胺", "视黄醇", "孕妇", "A醇", "酸类")

_RECOMMEND_PATTERNS = ("推荐", "帮我选", "买哪个", "哪款好", "适合什么", "适合我的", "用什么")

_COMPARE_PATTERNS = ("对比", "区别", "哪个好", "差别", "比较")

_USAGE_PATTERNS = ("怎么用", "用法", "用量", "使用方法", "怎么洗", "怎么穿")

_REVIEW_PATTERNS = ("评价", "口碑", "好用吗", "用过", "怎么样")

_SIZE_BOOST_PATTERNS = (
    "尺码", "穿什么码", "什么码", "身高", "体重", "偏大", "偏小",
    "选几码", "什么号", "腰围", "胸围", "肩宽", "衣长", "裤长",
)

_BEAUTY_WORDS = (
    "精华", "面霜", "洁面", "洗面", "防晒", "眼霜", "面膜", "护肤油", "乳液",
    "水乳", "爽肤水", "身体乳", "精华液", "口红", "粉底", "护肤", "肤质",
    "敏感肌", "油皮", "干皮", "痘痘", "美白", "保湿", "控油", "水杨酸",
    "烟酰胺", "玻尿酸", "神经酰胺",
)

_FASHION_WORDS = (
    "衣服", "穿搭", "T恤", "衬衫", "裙", "裤", "针织", "毛衣", "外套",
    "卫衣", "帽子", "鞋", "背心", "短裤", "连衣裙", "阔腿裤", "防晒衣",
    "版型", "面料", "通勤", "约会", "显瘦", "套装",
)

_SKIN_TYPE_KEYWORDS = ("油皮", "干皮", "敏感肌", "混合皮", "中性皮", "痘痘肌", "混油", "混干")
_BUDGET_KEYWORDS = ("预算", "价位", "多少钱", "价格", "入门", "中端", "高端", "百", "千元")
_SCENE_KEYWORDS = ("通勤", "约会", "日常", "秋冬", "夏天", "送礼", "送人", "场合", "场景", "上班")
_PRODUCT_NOUNS = (
    "精华", "面霜", "眼霜", "洁面", "防晒", "面膜", "水乳", "润肤油", "身体乳",
    "T恤", "裤子", "裙子", "衬衫", "外套", "卫衣", "针织衫", "毛衣", "帽子", "鞋",
    "玻尿酸", "烟酰胺", "水杨酸", "胜肽", "视黄醇", "氨基酸",
)

_PROFILE_MISSING_MAP: dict[str, tuple[str, ...]] = {
    "skin_type": ("skin_type",),
    "budget": ("price_band", "budget"),
    "size": ("size",),
    "scene": ("style", "scene"),
}


def _resolve_missing_with_profile(
    missing: list[str],
    profile: dict[str, Any],
) -> list[str]:
    """画像已覆盖的需求字段从缺失列表中移除（避免重复追问）。"""
    if not missing or not profile:
        return missing
    return [
        key
        for key in missing
        if not any(
            profile.get(candidate) not in (None, "", [], {})
            for candidate in _PROFILE_MISSING_MAP.get(key, ())
        )
    ]


def detect_sub_intent(question: str, intent: str = "") -> str:
    """按规则检测子意图（确定性，不依赖 LLM）。"""
    q = (question or "").strip().lower()
    if not q:
        return ""
    if any(p in q for p in _CHAT_PATTERNS):
        return "chat"
    if any(p in q for p in MEDIA_REQUEST_PATTERNS):
        return "media_request"
    if any(p in q for p in _NEGOTIATION_PATTERNS):
        return "price_negotiation"
    if intent == "aftersale" or any(p in q for p in _RETURN_PATTERNS):
        if any(p in q for p in _LOGISTICS_PATTERNS):
            return "logistics"
        return "return_exchange"
    if any(p in q for p in _LOGISTICS_PATTERNS):
        return "logistics"
    if _looks_like_allergy(q):
        return "allergy"
    if "年龄段" in q or "年龄" in q:
        return ""
    if intent == "comparison" or any(p in q for p in _COMPARE_PATTERNS):
        return "compare"
    if intent == "size_recommendation":
        if any(p in q for p in ("怎么选尺码", "穿什么码", "什么码", "选几码", "什么号")):
            return "recommend_request"
        return ""
    if intent == "recommendation" or any(p in q for p in _RECOMMEND_PATTERNS):
        return "recommend_request"
    if intent == "price_query" or any(p in q for p in _PRICE_INQUIRY_PATTERNS):
        return "price_inquiry"
    if any(p in q for p in _USAGE_PATTERNS):
        return "usage"
    if any(p in q for p in _REVIEW_PATTERNS):
        return "review"
    return ""


def detect_category_dim(question: str, intent: str = "") -> str:
    """检测品类维度：beauty / fashion / ""（服饰词优先，避免"防晒衣"误判美妆）。"""
    q = (question or "").strip()
    if not q:
        return ""
    if any(w in q for w in _FASHION_WORDS):
        return "fashion"
    if any(w in q for w in _BEAUTY_WORDS):
        return "beauty"
    return ""


def _looks_like_allergy(q: str) -> bool:
    """过敏/兼容性咨询：强词（过敏/副作用/慎用）或「敏感成分 + 可用性问法」组合。"""
    if any(p in q for p in ("过敏", "副作用", "慎用")):
        return True
    if any(p in q for p in _ALLERGY_COMPAT_MARKERS) and any(
        p in q for p in _ALLERGY_SUBJECT_MARKERS
    ):
        return True
    return False


def detect_missing_info(
    question: str,
    intent: str,
    sub_intent: str = "",
    profile: dict[str, Any] | None = None,
) -> list[str]:
    """检测缺失需求信息（供导购挖需；不覆盖检索层澄清逻辑）。"""
    q = (question or "").strip()
    missing: list[str] = []
    if sub_intent == "recommend_request" or intent == "recommendation":
        if not any(k in q for k in _SKIN_TYPE_KEYWORDS):
            missing.append("skin_type")
        if not any(k in q for k in _BUDGET_KEYWORDS):
            missing.append("budget")
    elif intent == "size_recommendation":
        if sub_intent == "recommend_request" and not any(
            k in q for k in ("体重", "腰围", "胸围", "肩宽")
        ):
            missing.append("size")
    elif intent in {"product_query", "price_query", "fashion_query"}:
        # 适配性咨询（适合我吗/合适吗）需要用户肤质信息
        if (
            intent in {"product_query", "fashion_query"}
            and any(k in q for k in ("适合我吗", "适合我", "合适吗", "适合吗"))
            and not any(k in q for k in _SKIN_TYPE_KEYWORDS)
            and "skin_type" not in missing
        ):
            missing.append("skin_type")
        if not any(k in q for k in _PRODUCT_NOUNS) and not any(
            k in q for k in ("这款", "这个", "那款", "它")
        ):
            missing.append("product")
    return _resolve_missing_with_profile(missing, profile or {})


def _enrich_route(
    route: QueryRoute,
    question: str,
    profile: dict[str, Any] | None = None,
) -> None:
    """填充扩展字段：子意图 / 购买信号 / 异议类型 / 缺失信息（不覆盖非空值）。"""
    from agent_base.agents.sales import detect_sales_signal

    if not route.sub_intent:
        route.sub_intent = detect_sub_intent(question, route.intent)
    # 过敏/兼容性咨询归一为商品咨询（成分/适用性属于商品参数范畴）
    if route.sub_intent == "allergy" and route.intent in {"recommendation", "general_qa", "fashion_query"}:
        route.intent = "product_query"
        route.sections = list(INTENT_SECTIONS["product_query"])
        route.metadata_filter = _section_filter(route.sections)
    signal = detect_sales_signal(question)
    route.buying_signal = signal["mode"]
    if signal["mode"] == "objection":
        route.objection_type = signal["objection_type"]
    elif not route.objection_type or route.objection_type == "none":
        route.objection_type = "none"
    if not route.missing_info:
        route.missing_info = detect_missing_info(
            question, route.intent, route.sub_intent, profile=profile
        )
    if not route.category_dim:
        route.category_dim = detect_category_dim(question, route.intent)


def build_metadata_filter(
    route: QueryRoute,
    product_name: str | None = None,
    product_spec: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """由意图章节 + 商品/类目约束构建 metadata 过滤条件。

    Args:
        route: 意图路由结果。
        product_name: 商品名约束（catalog 解析或显式传入）。
        product_spec: 商品规格/别名约束。
        category: 类目约束。

    Returns:
        Chroma/Qdrant-compatible metadata filter.
    """
    filters: list[dict[str, Any]] = []
    if route.metadata_filter:
        filters.append(route.metadata_filter)
    if product_name:
        filters.append({"product_name": product_name})
    if product_spec:
        filters.append({"product_spec": product_spec})
    if category:
        filters.append({"category": category})
    return combine_metadata_filters(filters)


def combine_metadata_filters(filters: list[dict[str, Any]]) -> dict[str, Any]:
    """用 $and 合并多个过滤条件（单条件直接返回）。"""
    non_empty = [item for item in filters if item]
    if not non_empty:
        return {}
    if len(non_empty) == 1:
        return non_empty[0]
    return {"$and": non_empty}


def _score_rule(question: str, rule: IntentRule) -> tuple[float, IntentRule, list[str]]:
    matched = [keyword for keyword in rule.keywords if keyword.lower() in question]
    if not matched:
        return 0.0, rule, []
    keyword_score = sum(1.0 + min(len(keyword), 6) * 0.1 for keyword in matched)
    return keyword_score * rule.priority, rule, matched


def _section_filter(sections: list[str]) -> dict[str, Any]:
    if not sections:
        return {}
    if len(sections) == 1:
        return {"section": sections[0]}
    return {"section": {"$in": sections}}
