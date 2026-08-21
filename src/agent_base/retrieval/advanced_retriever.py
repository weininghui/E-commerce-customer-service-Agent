"""高级检索编排：意图路由 → 策略决策 → 多路召回 → 重排 → trace。

本模块把检索策略决策（retrieval_policy）、查询改写（query_rewriter）、
意图识别（intent_router）、摘要/向量召回与重排（reranker）串成一条
可观测的检索链路，产出 AdvancedRetrievalTrace 供调试面板展示。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, TypedDict

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnableParallel

from agent_base.retrieval.metadata_retriever import (
    RetrievalItem,
    _similarity_search,
    document_to_retrieval_item,
)
from agent_base.retrieval.query_rewriter import QueryRewrite, build_query_rewrite
from agent_base.retrieval.retrieval_config import RetrievalConfig
from agent_base.retrieval.retrieval_policy import RetrievalDecision, build_retrieval_decision
from agent_base.retrieval.reranker import rerank_documents

# none：不重排；keyword：规则重排；model：外部重排模型；auto：由业务策略决定。
RERANK_STRATEGIES = ("auto", "none", "keyword", "model")


@dataclass(slots=True)
class AdvancedRetrievalTrace:
    """高级检索的可观察 Trace。

    这个对象有两个用途：
    1. 给 QA 链路继续使用：docs 字段保存最终入选的 LangChain Document。
    2. 给前端/评测使用：to_dict() 会把路由、改写、过滤器、各阶段命中数、最终来源都暴露出来。

    注意：docs 里是原始 Document 对象，不适合直接 JSON 序列化，所以 repr=False，
    API 返回时主要使用 results 里的结构化字段。
    """

    question: str
    mode: str
    rerank: str
    rewrite: QueryRewrite
    metadata_filter: dict[str, Any]
    search_query: str
    candidate_k: int
    final_k: int
    stage_counts: dict[str, int]
    fallback_used: bool
    results: list[RetrievalItem]
    decision: RetrievalDecision | None = None
    enhancement: dict[str, Any] = field(default_factory=dict)
    docs: list[Any] = field(default_factory=list, repr=False)
    errors: list[str] = field(default_factory=list)

    def to_dict(self, include_preview: bool = True) -> dict[str, Any]:
        """转换为可返回给前端的字典。

        include_preview=False 时会移除 preview，适合评测日志或更小的接口响应。
        """
        payload = {
            "question": self.question,
            "mode": self.mode,
            "rerank": self.rerank,
            "search_query": self.search_query,
            "route": self.rewrite.route.to_dict(),
            "rewrite": self.rewrite.to_dict(),
            "metadata_filter": self.metadata_filter,
            "candidate_k": self.candidate_k,
            "final_k": self.final_k,
            "stage_counts": self.stage_counts,
            "fallback_used": self.fallback_used,
            "enhancement": self.enhancement,
            "decision": self.decision.to_dict() if self.decision else None,
            "errors": self.errors,
            "results": [asdict(item) for item in self.results],
        }
        if not include_preview:
            for item in payload["results"]:
                item.pop("preview", None)
        return payload


def retrieve_advanced(
    vector_store: Any,
    question: str,
    cfg: RetrievalConfig | None = None,
    *,
    summary_store: Any | None = None,
    decision: RetrievalDecision | None = None,
    domain: Any | None = None,
    sparse_store: Any | None = None,
    search_query: str | None = None,
    current_product: str | None = None,
) -> AdvancedRetrievalTrace:
    """执行可观察的高级检索流程。

    入口会先根据 mode 决定是否使用自动策略。auto 模式会产生 RetrievalDecision，
    再按策略执行 metadata、summary、summary_guided、vector 等阶段。最后统一
    去重、rerank，并把每个阶段数量和最终结果写入 Trace，供前端展示和评测。

    整体执行顺序：
    1. 校验 mode/rerank，避免非法参数进入检索链路。
    2. 如果传入了 decision（来自上游 route 节点），跳过内部 rebuild，直接复用。
    3. 否则调用 build_retrieval_decision 做"问题重写 + 意图路由 + 策略选择"。
    4. 按策略执行一个或多个检索阶段，每个阶段都会给 Document 标记 retrieval_stage。
    5. 多路召回结果去重，避免同一个 chunk 被重复用于答案生成。
    6. 对候选文档 rerank，得到最终 Top K。
    7. 把最终结果和中间过程封装为 AdvancedRetrievalTrace。

    P1 新增 decision 参数（契约 P1-03）：传入时跳过 build_retrieval_decision，
    直接使用传入的策略决策（route 节点已产出）。不传时行为与现状完全一致。
    """
    cfg = cfg or RetrievalConfig()
    rerank = cfg.rerank
    candidate_k = cfg.candidate_k
    final_k = cfg.top_k
    product_name = cfg.product_name
    product_spec = cfg.product_spec
    category = cfg.category
    use_rewrite = cfg.use_rewrite
    fallback_without_filter = cfg.fallback_without_filter
    rerank_model_provider = cfg.rerank_model.provider
    rerank_model_name = cfg.rerank_model.model
    rerank_model_endpoint = cfg.rerank_model.endpoint
    rerank_model_api_key_env = cfg.rerank_model.api_key_env
    rerank_model_timeout = cfg.rerank_model.timeout
    rerank_model_strategies = cfg.rerank_model.strategies
    preserve_preferred_sections = cfg.preserve_preferred_sections
    intent_classifier = cfg.intent_classifier
    profile = cfg.profile
    rerank = _validate(rerank, RERANK_STRATEGIES, "rerank strategy")

    if decision is not None:
        # 复用上游决策，跳过内部路由/改写
        rewrite = build_query_rewrite(
            question,
            product_name=product_name,
            product_spec=product_spec,
            intent_classifier=intent_classifier,
            domain=domain,
            profile=profile,
        )
    else:
        rewrite, decision = build_retrieval_decision(
            question=question,
            product_name=product_name,
            product_spec=product_spec,
            category=category,
            current_product=current_product,
            top_k=final_k,
            candidate_k=candidate_k,
            rerank=rerank,
            use_rewrite=use_rewrite,
            intent_classifier=intent_classifier,
            profile=profile,
        )
    metadata_filter = decision.metadata_filter
    # P15：search_query 可由调用方传入（如 enrich 后的查询）；
    # 意图路由（rewrite/decision）仍基于原始 question，避免 enrich 干扰 intent。
    search_query = search_query or decision.rewritten_query
    # T4: enrich_alias 启用时扩展别名（"玻尿酸精华"→"玻尿酸保湿精华液"），
    # 提升商品定位准确率与召回；只影响检索查询，不影响意图路由。
    try:
        from agent_base.config import deep_get, load_yaml

        _cfg = load_yaml("configs/app.yaml") or {}
        if deep_get(_cfg, "retrieval.enrich_alias.enabled", False):
            from agent_base.retrieval.enrich import expand_aliases

            search_query = expand_aliases(search_query)
        # T4b: enrich_reference 启用且会话有 current_product 时，
        # 指代补全（"它多少钱"→"玻尿酸保湿精华液 它多少钱"）
        if current_product and deep_get(_cfg, "retrieval.enrich_reference.enabled", False):
            from agent_base.retrieval.enrich import resolve_question

            search_query = resolve_question(search_query, current_product)
    except Exception:
        pass
    rerank = _resolve_auto_rerank_strategy(
        requested_rerank=rerank,
        decision=decision,
        model_provider=rerank_model_provider,
        model_strategies=rerank_model_strategies,
    )
    candidate_k = decision.candidate_k
    final_k = decision.final_k
    stage_docs: list[Any] = []
    stage_counts: dict[str, int] = {}
    errors: list[str] = []
    fallback_used = False

    stage_docs, stage_counts, fallback_used = _retrieve_by_decision(
        decision=decision,
        vector_store=vector_store,
        summary_store=summary_store,
        sparse_store=sparse_store,
        query=search_query,
        metadata_filter=metadata_filter,
        candidate_k=candidate_k,
        final_k=final_k,
        errors=errors,
        fallback_without_filter=fallback_without_filter,
    )

    # P32d/e：检索增强按需触发（Decomposition / Multi-Query）
    # 两者互斥，Decomposition 优先；均关闭时零开销跳过
    enhancement: dict[str, Any] = {"triggered": False, "type": "none"}
    if decision.strategy not in ("clarification",):
        try:
            from agent_base.retrieval.enhancement import (
                assess_enhancement,
                run_decomposition_enhancement,
                run_multi_query_enhancement,
            )

            enhancement = assess_enhancement(
                question=search_query,
                stage_docs=stage_docs,
                intent=decision.intent,
            )
            if enhancement["triggered"]:
                if enhancement["type"] == "decomposition":
                    enhanced_docs = run_decomposition_enhancement(
                        question, vector_store,
                        k=max(4, candidate_k), metadata_filter=metadata_filter,
                    )
                    stage_docs = list(enhanced_docs)
                    stage_counts["enhancement_decomposition"] = len(stage_docs)
                elif enhancement["type"] == "multi_query":
                    llm = None
                    try:
                        from agent_base.llms import build_chat_model
                        from agent_base.config import deep_get, load_yaml
                        _cfg = load_yaml("configs/app.yaml") or {}
                        _mq_cfg = deep_get(_cfg, "retrieval.multi_query", {}) or {}
                        _llm_cfg = deep_get(_cfg, "llm", {}) or {}
                        llm = build_chat_model(
                            provider=_mq_cfg.get("provider") or _llm_cfg.get("provider", "none"),
                            model=_mq_cfg.get("model") or _llm_cfg.get("model"),
                            base_url=_mq_cfg.get("base_url") or _llm_cfg.get("base_url"),
                            api_key_env=(
                                _mq_cfg.get("api_key_env")
                                or _llm_cfg.get("api_key_env")
                                or "DASHSCOPE_API_KEY"
                            ),
                            temperature=0.1,
                        )
                    except Exception:
                        pass
                    enhanced_docs = run_multi_query_enhancement(
                        question, vector_store, llm=llm,
                        k=max(4, candidate_k), metadata_filter=metadata_filter,
                    )
                    if enhanced_docs:
                        stage_docs = list(enhanced_docs)
                        stage_counts["enhancement_multi_query"] = len(stage_docs)
                    else:
                        # 增强空结果不得清掉基线（如 filter 命中 0 时保持原候选）
                        stage_counts["enhancement_multi_query"] = 0
        except Exception:
            pass

    candidates = _dedupe(stage_docs)

    # keyword rerank 会根据问题词、章节偏好等给候选文档重新打分。
    # preferred_sections 来自意图路由，能让"用法用量"问题优先保留用法用量章节。
    selected_docs = rerank_documents(
        search_query,
        candidates,
        strategy=rerank,
        top_k=final_k,
        preferred_sections=rewrite.route.sections,
        model_provider=rerank_model_provider,
        model_name=rerank_model_name,
        model_endpoint=rerank_model_endpoint,
        model_api_key_env=rerank_model_api_key_env,
        model_timeout=rerank_model_timeout,
        preserve_preferred_sections=preserve_preferred_sections,
        errors=errors,
    )

    # 把最终 Document 转成前端可展示的结果项：
    # rank、score、retrieval_stage、chunk_id、section、source_file、page 等。
    results = [
        document_to_retrieval_item(
            rank=rank,
            doc=doc,
            score=(getattr(doc, "metadata", {}) or {}).get("vector_score"),
        )
        for rank, doc in enumerate(selected_docs, start=1)
    ]

    # Trace 是这个函数最重要的产物：既包含最终 docs，也包含检索过程。
    # QA 层用 trace.docs 生成回答；前端用 trace.to_dict() 展示检索路径。
    return AdvancedRetrievalTrace(
        question=question,
        mode="auto",
        rerank=rerank,
        rewrite=rewrite,
        metadata_filter=metadata_filter,
        search_query=search_query,
        candidate_k=candidate_k,
        final_k=final_k,
        stage_counts=stage_counts,
        fallback_used=fallback_used,
        results=results,
        decision=decision,
        enhancement=enhancement,
        docs=selected_docs,
        errors=errors,
    )


def _resolve_auto_rerank_strategy(
    requested_rerank: str,
    decision: RetrievalDecision,
    model_provider: str,
    model_strategies: list[str] | None,
) -> str:
    """解析 auto rerank 的实际执行方式。

    项目默认重排方式为模型（本地 TEI bge-reranker-v2-m3）：只要模型提供方
    已配置，auto 一律走模型重排，不再按策略区分；模型不可用时由
    rerank_documents 内部自动降级 keyword，保证链路不中断。
    """
    requested = (requested_rerank or "auto").lower()
    if requested != "auto":
        return requested
    if (model_provider or "none").lower() in {"none", "off", "false"}:
        return decision.rerank
    return "model"


class _RetrievalCtx(TypedDict, total=False):
    """检索链输入上下文（策略分支/召回通道共享）。"""
    strategy: str
    vector_store: Any
    summary_store: Any | None
    sparse_store: Any | None
    query: str
    metadata_filter: dict[str, Any]
    candidate_k: int
    final_k: int
    errors: list[str]
    fallback_without_filter: bool
    _fallback_used: bool  # 无过滤兜底标记（通道内写入，分支读取）


def _k(ctx: _RetrievalCtx, full: bool) -> int:
    """按分支取召回量：full=candidate_k；half=max(2, candidate_k//2)。"""
    return ctx["candidate_k"] if full else max(2, ctx["candidate_k"] // 2)


def _channel_metadata(ctx: _RetrievalCtx) -> list[Any]:
    """metadata 精确召回通道（含无结果时的 vector 检索与无过滤兜底）。"""
    k = _k(ctx, full=True)
    mf = ctx["metadata_filter"]
    docs = _metadata_get_stage(ctx["vector_store"], k, mf, "metadata", ctx["errors"])
    if not docs:
        docs = _search_stage(ctx["vector_store"], ctx["query"], k, mf, "metadata_vector", ctx["errors"])
    if not docs and mf and ctx["fallback_without_filter"]:
        ctx["_fallback_used"] = True
        docs = _search_stage(ctx["vector_store"], ctx["query"], k, {}, "metadata_fallback", ctx["errors"])
    return docs


def _channel_catalog_vector(ctx: _RetrievalCtx) -> list[Any]:
    """catalog 分支的 vector 通道（含无结果时的无过滤兜底）。"""
    k = _k(ctx, full=True)
    mf = ctx["metadata_filter"]
    docs = _search_stage(ctx["vector_store"], ctx["query"], k, mf, "catalog_vector", ctx["errors"])
    if not docs and mf and ctx["fallback_without_filter"]:
        ctx["_fallback_used"] = True
        docs = _search_stage(ctx["vector_store"], ctx["query"], k, {}, "catalog_vector_fallback", ctx["errors"])
    return docs


def _channel_summary(ctx: _RetrievalCtx, full: bool) -> list[Any]:
    """摘要检索通道（供 catalog 分支单独使用）。"""
    return _search_summary_stage(
        ctx["summary_store"], ctx["query"], _k(ctx, full), ctx["metadata_filter"], ctx["errors"],
    )


def _channel_summary_guided(ctx: _RetrievalCtx, full: bool) -> dict[str, Any]:
    """摘要定位 → 原文取证两跳通道（summary 只查一次，供计数与 guided 共用）。"""
    k = _k(ctx, full)
    summary_docs = _search_summary_stage(
        ctx["summary_store"], ctx["query"], k, ctx["metadata_filter"], ctx["errors"],
    )
    guided_docs = _search_summary_guided_chunks(
        vector_store=ctx["vector_store"],
        query=ctx["query"],
        summary_docs=summary_docs,
        metadata_filter=ctx["metadata_filter"],
        candidate_k=k,
        errors=ctx["errors"],
    )
    return {"summary": summary_docs, "guided": guided_docs}


def _channel_vector(ctx: _RetrievalCtx, full: bool) -> list[Any]:
    """无过滤向量召回通道。"""
    return _search_stage(ctx["vector_store"], ctx["query"], _k(ctx, full), {}, "vector", ctx["errors"])


def _channel_sparse(ctx: _RetrievalCtx) -> list[Any]:
    """BM25 稀疏向量补充通道（可选，异常不阻断主链）。"""
    if ctx.get("sparse_store") is None:
        return []
    try:
        return _search_sparse_stage(
            ctx["sparse_store"], ctx["query"], _k(ctx, full=False),
            ctx["metadata_filter"], "sparse", ctx["errors"],
        )
    except Exception as exc:
        ctx["errors"].append(f"sparse: {type(exc).__name__}: {exc}")
        return []


# ── dense+sparse 融合（RRF 加权，配置门控）───────────────────────────────

_FUSION_DEFAULTS = {"enabled": True, "k": 60, "dense_weight": 0.7, "sparse_weight": 0.3}


def _fusion_config() -> dict[str, Any]:
    """读取 configs/app.yaml 的 RRF 融合配置（retrieval.fusion 开关 + retrieval.hybrid 权重）。"""
    try:
        from agent_base.config import deep_get, load_yaml

        cfg = load_yaml("configs/app.yaml") or {}
        fusion = dict(deep_get(cfg, "retrieval.fusion", {}) or {})
        hybrid = dict(deep_get(cfg, "retrieval.hybrid", {}) or {})
        out = dict(_FUSION_DEFAULTS)
        if "enabled" in fusion:
            out["enabled"] = bool(fusion["enabled"])
        if "k" in fusion:
            out["k"] = int(fusion["k"])
        if "dense_weight" in hybrid:
            out["dense_weight"] = float(hybrid["dense_weight"])
        if "sparse_weight" in hybrid:
            out["sparse_weight"] = float(hybrid["sparse_weight"])
        return out
    except Exception:
        return dict(_FUSION_DEFAULTS)


def _append_semantic(
    stage_docs: list[Any],
    stage_counts: dict[str, int],
    vector_docs: list[Any],
    sparse_docs: list[Any],
    errors: list[str],
) -> None:
    """语义通道并入主召回：dense 向量 + sparse BM25 按配置加权 RRF 融合。

    metadata 精确过滤与摘要两跳通道保持前置优先级（高精度证据不被 RRF 打散）；
    只有同为"打分排序"的稠密向量与 BM25 稀疏两路做排名融合。融合关闭或失败时
    退化为旧行为（向量在前、稀疏追加），链路不中断。
    """
    if not vector_docs and not sparse_docs:
        return
    if sparse_docs:
        stage_counts["sparse"] = len(sparse_docs)
    if not vector_docs:
        stage_docs.extend(sparse_docs)
        return
    if not sparse_docs:
        stage_docs.extend(vector_docs)
        return
    cfg = _fusion_config()
    if not cfg.get("enabled", False):
        stage_docs.extend(vector_docs)
        stage_docs.extend(sparse_docs)
        return
    try:
        from agent_base.retrieval.fusion import rrf_fusion

        fused = rrf_fusion(
            [vector_docs, sparse_docs],
            k=int(cfg.get("k", 60)),
            weights=[float(cfg.get("dense_weight", 0.7)), float(cfg.get("sparse_weight", 0.3))],
        )
        for doc in fused:
            md = dict(getattr(doc, "metadata", {}) or {})
            md["fusion"] = "rrf"
            try:
                doc.metadata = md
            except Exception:
                pass
        stage_docs.extend(fused)
        stage_counts["rrf_fusion"] = len(fused)
    except Exception as exc:
        errors.append(f"rrf_fusion: {type(exc).__name__}: {exc}")
        stage_docs.extend(vector_docs)
        stage_docs.extend(sparse_docs)


def _clarify_branch(ctx: _RetrievalCtx) -> tuple[list[Any], dict[str, int], bool]:
    """clarification：信息不足，不做盲目检索，由答案层追问。"""
    return [], {"clarification": 0}, False


def _catalog_branch(ctx: _RetrievalCtx) -> tuple[list[Any], dict[str, int], bool]:
    """catalog_search：轻量召回 + 目录导向（summary ∥ vector ∥ sparse 并行）。"""
    recall = RunnableParallel(
        summary=RunnableLambda(lambda c: _channel_summary(c, full=False)),
        vector=RunnableLambda(_channel_catalog_vector),
        sparse=RunnableLambda(_channel_sparse),
    ).invoke(ctx)
    stage_docs: list[Any] = [*recall["summary"]]
    stage_counts = {"summary": len(recall["summary"]), "catalog_vector": len(recall["vector"])}
    _append_semantic(stage_docs, stage_counts, recall["vector"], recall["sparse"], ctx["errors"])
    return stage_docs, stage_counts, bool(ctx.get("_fallback_used"))


def _metadata_first_branch(ctx: _RetrievalCtx) -> tuple[list[Any], dict[str, int], bool]:
    """metadata_first：固定章节问题优先精确过滤（metadata ∥ sparse 并行 + 不足时 vector 兜底）。"""
    recall = RunnableParallel(
        metadata=RunnableLambda(_channel_metadata),
        sparse=RunnableLambda(_channel_sparse),
    ).invoke(ctx)
    stage_docs: list[Any] = [*recall["metadata"]]
    stage_counts = {"metadata": len(recall["metadata"])}
    # 精准过滤后证据不足 → 无过滤向量检索兜底（保留高精度，避免章节识别错误时无答案）
    vector_docs: list[Any] = []
    if len(_dedupe(stage_docs)) < ctx["final_k"]:
        vector_docs = _search_stage(ctx["vector_store"], ctx["query"], _k(ctx, full=False), {}, "vector_fallback", ctx["errors"])
        stage_counts["vector_fallback"] = len(vector_docs)
    _append_semantic(stage_docs, stage_counts, vector_docs, recall["sparse"], ctx["errors"])
    return stage_docs, stage_counts, bool(ctx.get("_fallback_used"))


def _safety_branch(ctx: _RetrievalCtx) -> tuple[list[Any], dict[str, int], bool]:
    """safety_hybrid：合规/健康风险多章节分散（metadata ∥ 摘要两跳 ∥ vector ∥ sparse 并行）。"""
    recall = RunnableParallel(
        metadata=RunnableLambda(_channel_metadata),
        guided=RunnableLambda(lambda c: _channel_summary_guided(c, full=False)),
        vector=RunnableLambda(lambda c: _channel_vector(c, full=False)),
        sparse=RunnableLambda(_channel_sparse),
    ).invoke(ctx)
    stage_docs: list[Any] = [
        *recall["metadata"], *recall["guided"]["guided"], *recall["guided"]["summary"],
    ]
    stage_counts = {
        "metadata": len(recall["metadata"]),
        "summary_guided": len(recall["guided"]["guided"]),
        "summary": len(recall["guided"]["summary"]),
        "vector": len(recall["vector"]),
    }
    _append_semantic(stage_docs, stage_counts, recall["vector"], recall["sparse"], ctx["errors"])
    return stage_docs, stage_counts, bool(ctx.get("_fallback_used"))


def _summary_guided_branch(ctx: _RetrievalCtx) -> tuple[list[Any], dict[str, int], bool]:
    """summary_guided_hybrid：泛问先摘要定位章节再回原文取证（摘要两跳 ∥ vector ∥ sparse 并行）。"""
    recall = RunnableParallel(
        guided=RunnableLambda(lambda c: _channel_summary_guided(c, full=True)),
        vector=RunnableLambda(lambda c: _channel_vector(c, full=True)),
        sparse=RunnableLambda(_channel_sparse),
    ).invoke(ctx)
    stage_docs: list[Any] = [
        *recall["guided"]["guided"], *recall["guided"]["summary"],
    ]
    stage_counts = {
        "summary_guided": len(recall["guided"]["guided"]),
        "summary": len(recall["guided"]["summary"]),
        "vector": len(recall["vector"]),
    }
    _append_semantic(stage_docs, stage_counts, recall["vector"], recall["sparse"], ctx["errors"])
    return stage_docs, stage_counts, bool(ctx.get("_fallback_used"))


def _hybrid_branch(ctx: _RetrievalCtx) -> tuple[list[Any], dict[str, int], bool]:
    """默认 hybrid：有约束但意图不明（metadata ∥ 摘要两跳 ∥ vector ∥ sparse 并行）。"""
    recall = RunnableParallel(
        metadata=RunnableLambda(_channel_metadata),
        guided=RunnableLambda(lambda c: _channel_summary_guided(c, full=False)),
        vector=RunnableLambda(lambda c: _channel_vector(c, full=False)),
        sparse=RunnableLambda(_channel_sparse),
    ).invoke(ctx)
    stage_docs: list[Any] = [
        *recall["metadata"], *recall["guided"]["guided"], *recall["guided"]["summary"],
    ]
    stage_counts = {
        "metadata": len(recall["metadata"]),
        "summary_guided": len(recall["guided"]["guided"]),
        "summary": len(recall["guided"]["summary"]),
        "vector": len(recall["vector"]),
    }
    _append_semantic(stage_docs, stage_counts, recall["vector"], recall["sparse"], ctx["errors"])
    return stage_docs, stage_counts, bool(ctx.get("_fallback_used"))


# 官方 LCEL 链：RunnableBranch 按策略分发（每分支内部 RunnableParallel 并行召回）
_RETRIEVE_CHAIN: Runnable = RunnableBranch(
    (lambda ctx: ctx["strategy"] == "clarification", RunnableLambda(_clarify_branch)),
    (lambda ctx: ctx["strategy"] == "catalog_search", RunnableLambda(_catalog_branch)),
    (lambda ctx: ctx["strategy"] == "metadata_first", RunnableLambda(_metadata_first_branch)),
    (lambda ctx: ctx["strategy"] == "safety_hybrid", RunnableLambda(_safety_branch)),
    (lambda ctx: ctx["strategy"] == "summary_guided_hybrid", RunnableLambda(_summary_guided_branch)),
    RunnableLambda(_hybrid_branch),  # 默认 hybrid
)


def _retrieve_by_decision(
    decision: RetrievalDecision | None,
    vector_store: Any,
    summary_store: Any | None,
    sparse_store: Any | None,
    query: str,
    metadata_filter: dict[str, Any],
    candidate_k: int,
    final_k: int,
    errors: list[str],
    fallback_without_filter: bool,
) -> tuple[list[Any], dict[str, int], bool]:
    """按 RetrievalDecision 执行召回阶段——官方 LCEL 链入口。

    策略分发用 ``RunnableBranch``，多路召回用 ``RunnableParallel``（见 ``_RETRIEVE_CHAIN``）。
    策略不是用户选择的，而是由 retrieval_policy 根据问题意图产生：
    - metadata_first：固定章节问题，先用 metadata filter 精准召回。
    - safety_hybrid：合规/健康风险类问题，多路召回降低漏掉风险信息的概率。
    - summary_guided_hybrid：泛问或章节不明确，先用摘要定位章节，再回原文取证。
    - hybrid：商品/类目约束但意图不清晰，综合多路召回。
    - clarification：信息不足，不做盲目检索，由答案层追问。
    - catalog_search：商品/类别咨询，做轻量召回和目录导向提示。

    每类策略的 dense 向量与 sparse BM25 语义通道经 _append_semantic 合并：
    retrieval.fusion.enabled 开启时做加权 RRF 排名融合（metadata/摘要两跳
    前置通道保持优先级），关闭或异常时退化为“向量在前、稀疏追加”。
    """
    if decision is None:
        return [], {}, False
    ctx: _RetrievalCtx = {
        "strategy": decision.strategy,
        "vector_store": vector_store,
        "summary_store": summary_store,
        "sparse_store": sparse_store,
        "query": query,
        "metadata_filter": metadata_filter,
        "candidate_k": candidate_k,
        "final_k": final_k,
        "errors": errors,
        "fallback_without_filter": fallback_without_filter,
    }
    return _RETRIEVE_CHAIN.invoke(ctx)


def _search_stage(
    store: Any,
    query: str,
    k: int,
    metadata_filter: dict[str, Any],
    stage: str,
    errors: list[str],
) -> list[Any]:
    """执行一次 Chroma similarity_search，并给返回文档标记 retrieval_stage。

    参数说明：
    - store：Chroma collection 包装对象，可以是原文 chunk store，也可以是摘要 store。
    - query：最终用于向量检索的查询，一般是重写后的问题。
    - k：本阶段召回数量。
    - metadata_filter：Chroma filter，例如 {"section": "用法用量"}。
    - stage：当前阶段名称，会写入 doc.metadata["retrieval_stage"]，供 Trace 和来源展示使用。
    - errors：异常不抛出到上层，而是记录在这里，保证其他检索阶段还能继续执行。
    """
    try:
        docs_with_scores = _similarity_search(store, query, k=k, metadata_filter=metadata_filter)
    except Exception as exc:
        errors.append(f"{stage}: {type(exc).__name__}: {exc}")
        return []
    return [_tag_doc(doc, score=score, stage=stage) for doc, score in docs_with_scores]


def _metadata_get_stage(
    store: Any,
    k: int,
    metadata_filter: dict[str, Any],
    stage: str,
    errors: list[str],
) -> list[Any]:
    """执行确定性的 metadata-only 查询。

    固定章节问题，例如"这件衣服怎么退换"，本质上已经有非常强的结构化条件：
    section=售后FAQ + product_name/product_spec=商品名/规格。此时不应该完全依赖
    similarity_search_with_score，因为它仍然要先做 query embedding 和向量相似度查询。

    这里直接调用 Chroma collection.get(where=...)，只按 metadata 取文档。
    取到后再交给后续 keyword rerank 排序，保证固定章节问题不会因为向量检索阶段异常或相似度
    行为差异而返回 0 条。
    """
    if not metadata_filter:
        return []
    collection = getattr(store, "_collection", None)
    if collection is None or not hasattr(collection, "get"):
        return []
    try:
        payload = collection.get(where=metadata_filter, limit=k, include=["documents", "metadatas"])
    except Exception as exc:
        errors.append(f"{stage}_get: {type(exc).__name__}: {exc}")
        return []

    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    ids = payload.get("ids") or []
    docs: list[Any] = []
    for idx, text in enumerate(documents):
        metadata = dict(metadatas[idx] or {}) if idx < len(metadatas) else {}
        if idx < len(ids) and ids[idx] and "chunk_id" not in metadata:
            metadata["chunk_id"] = ids[idx]
        docs.append(_tag_doc(Document(page_content=text or "", metadata=metadata), score=None, stage=stage))
    return docs


def _search_summary_stage(
    summary_store: Any | None,
    query: str,
    k: int,
    metadata_filter: dict[str, Any],
    errors: list[str],
) -> list[Any]:
    """检索章节摘要 collection；metadata filter 会裁剪为摘要索引支持的字段。

    摘要索引不是完整原文，而是按章节生成的短摘要。它适合解决：
    - 用户问题很泛，难以直接定位原文章节。
    - 原文 chunk 很长或表述分散，直接向量检索不稳定。

    注意：summary_store 可能没有配置，所以这里会把错误写入 Trace，而不是中断问答。
    """
    if summary_store is None:
        errors.append("summary: summary_store is not configured")
        return []
    return _search_stage(summary_store, query, k, _summary_metadata_filter(metadata_filter), "summary", errors)


def _search_summary_guided_chunks(
    vector_store: Any,
    query: str,
    summary_docs: list[Any],
    metadata_filter: dict[str, Any],
    candidate_k: int,
    errors: list[str],
) -> list[Any]:
    """根据摘要命中的 section 回到原文 chunk collection 取证。

    summary_guided 的思想是"两跳检索"：
    1. 第一跳查 ecommerce_summaries，找到可能相关的章节，例如"商品参数""成分"。
    2. 第二跳查 ecommerce_chunks，并把第一跳命中的 section 作为 metadata filter。

    这样既利用摘要帮助定位章节，又保证最终证据仍来自商品资料原文 chunk。
    """
    sections = _sections_from_docs(summary_docs)
    if not sections:
        return []
    if len(sections) == 1:
        section_filter: dict[str, Any] = {"section": sections[0]}
    else:
        section_filter = {"section": {"$in": sections}}
    guided_filter = _combine_stage_filters(section_filter, _identity_metadata_filter(metadata_filter))
    return _search_stage(vector_store, query, candidate_k, guided_filter, "summary_guided", errors)


def _tag_doc(doc: Any, score: float | None, stage: str) -> Any:
    """给召回到的 Document 补充检索过程元数据。

    LangChain/Chroma 返回的 Document metadata 来自入库时的 chunk metadata。
    这里额外加：
    - vector_score：当前检索阶段返回的相似度分数。
    - retrieval_stage：来自哪个检索阶段，前端 Trace 和来源展示会用到。
    """
    metadata = dict(getattr(doc, "metadata", {}) or {})
    metadata["vector_score"] = score
    metadata["retrieval_stage"] = stage
    try:
        doc.metadata = metadata
    except Exception:
        pass
    return doc


def _dedupe(docs: Iterable[Any]) -> list[Any]:
    """多路召回会产生重复结果，这里按 chunk_id 或摘要键去重并保留首次出现。"""
    seen: set[str] = set()
    unique: list[Any] = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        # 原文 chunk 优先用 chunk_id 去重；摘要文档没有 chunk_id 时，
        # 用 doc_id + summary_type + section 作为摘要级别的唯一键。
        key = (
            metadata.get("chunk_id")
            or f"{metadata.get('doc_id')}:{metadata.get('summary_type')}:{metadata.get('section')}"
            or getattr(doc, "page_content", str(doc))[:120]
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _sections_from_docs(docs: list[Any]) -> list[str]:
    """从摘要检索结果中提取 section，并保持首次出现顺序。"""
    sections = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        section = metadata.get("section")
        if section:
            sections.append(str(section))
    return list(dict.fromkeys(sections))


def _summary_metadata_filter(metadata_filter: dict[str, Any]) -> dict[str, Any]:
    """把通用 metadata_filter 裁剪成摘要索引支持的字段。

    原文 chunk 和摘要 collection 的 metadata 字段不完全一样。
    如果把摘要索引不支持的字段传给 Chroma filter，可能导致查询失败或 0 命中。
    """
    return _filter_metadata_keys(
        metadata_filter,
        supported_keys={"doc_id", "section", "product_name", "product_spec", "source_file", "summary_type"},
    )


def _identity_metadata_filter(metadata_filter: dict[str, Any]) -> dict[str, Any]:
    """保留商品身份相关过滤条件，供 summary_guided 回原文取证时使用。

    summary_guided 已经会用摘要命中的 section 做章节过滤；
    这里再叠加 doc_id/product_name/product_spec/category 等身份过滤，避免跨商品取证。
    """
    return _filter_metadata_keys(
        metadata_filter,
        supported_keys={"doc_id", "product_name", "product_spec", "category", "source_file"},
    )


def _filter_metadata_keys(metadata_filter: dict[str, Any], supported_keys: set[str]) -> dict[str, Any]:
    """递归裁剪 metadata filter，只保留目标 collection 支持的字段。

    支持 Chroma 常见的组合过滤结构：
    - {"$and": [filter1, filter2]}
    - {"$or": [filter1, filter2]}

    如果某个子过滤器被裁剪为空，就丢弃它；如果组合条件只剩一个子条件，则拍平成普通 filter。
    """
    if not metadata_filter:
        return {}

    filtered: dict[str, Any] = {}
    for key, value in metadata_filter.items():
        if key in {"$and", "$or"} and isinstance(value, list):
            children = [_filter_metadata_keys(item, supported_keys) for item in value if isinstance(item, dict)]
            children = [item for item in children if item]
            if not children:
                continue
            if len(children) == 1:
                filtered.update(children[0])
            else:
                filtered[key] = children
        elif key in supported_keys:
            filtered[key] = value
    return filtered


def _combine_stage_filters(*filters: dict[str, Any]) -> dict[str, Any]:
    """合并多个 Chroma metadata filter。

    空 filter 会被忽略；多个非空 filter 用 $and 组合。
    例如 summary_guided 中会合并：
    - {"section": {"$in": ["商品参数", "卖点"]}}
    - {"product_spec": "保湿精华"}
    """
    non_empty = [item for item in filters if item]
    if not non_empty:
        return {}
    if len(non_empty) == 1:
        return non_empty[0]
    return {"$and": non_empty}


def _validate(value: str, allowed: tuple[str, ...], label: str) -> str:
    """校验枚举型参数，并统一转小写。"""
    normalized = (value or "").lower()
    if normalized not in allowed:
        raise ValueError(f"Unsupported {label}: {value}. Expected one of: {', '.join(allowed)}")
    return normalized


def _search_sparse_stage(
    sparse_store: Any,
    query: str,
    k: int,
    metadata_filter: dict[str, Any],
    stage: str,
    errors: list[str],
) -> list[Any]:
    """执行一次 Qdrant sparse（BM25 稀疏向量）检索，返回带 retrieval_stage 的 Document。

    P15：作为策略链路的补充召回路，与 metadata/summary/vector 一起参与融合。

    Args:
        sparse_store: Qdrant sparse collection 的 VectorStore 实例。
        query: 检索查询文本。
        k: 返回数量。
        metadata_filter: Chroma 风格 metadata filter（自动转换 Qdrant 语法）。
        stage: 阶段名（写入 Document metadata.retrieval_stage）。
        errors: 错误收集列表。

    Returns:
        带检索分数的 Document 列表；失败时返回空列表。
    """
    from langchain_core.documents import Document
    from agent_base.retrieval.sparse_encoder import encode_query_sparse
    from agent_base.retrieval.filter_adapter import chroma_to_qdrant_filter

    client = getattr(sparse_store, "client", None)
    if client is None:
        return []

    sparse_vec = encode_query_sparse(query)
    if not sparse_vec.indices:
        return []

    query_filter = None
    if metadata_filter:
        qf = chroma_to_qdrant_filter(metadata_filter)
        if qf:
            query_filter = qf

    try:
        hits = client.query_points(
            collection_name=sparse_store.collection_name,
            # BUG-4 修复：稀疏向量直接作 query（NearestQuery 包装会触发
            # Qdrant 400「Not existing vector name」，稀疏通道被静默降级）
            query=sparse_vec,
            using="text",
            limit=k,
            with_payload=True,
            query_filter=query_filter,
        )
    except Exception as exc:
        errors.append(f"{stage}: {type(exc).__name__}: {exc}")
        return []

    docs: list[Document] = []
    for hit in hits.points:
        payload = hit.payload or {}
        meta = dict(payload.get("metadata", {}) or {})
        meta["retrieval_stage"] = stage
        meta["sparse_score"] = hit.score
        docs.append(Document(page_content=payload.get("text", ""), metadata=meta))
    return docs
