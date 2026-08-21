"""经典问答编排（v0.30.0 从 api/main.py 抽出）。

职责：classic 模式完整问答（检索 + 生成 + 缓存），request 参数为
RagRequest（类型 Any 避免与 api 层循环依赖）。
"""

from __future__ import annotations

from typing import Any

from agent_base.chains import answer_question_with_trace
from agent_base.retrieval.retrieval_config import RetrievalConfig


def run_classic_ask(
    request: Any,
    constraints: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """classic 模式非流式回答（P11: 缓存接线）。

    Args:
        request: RagRequest（question/top_k/candidate_k/rerank/session_id）。
        constraints: 商品约束（product_name/product_spec/category 等）。
        runtime: 运行期对象（向量库/LLM/rerank 配置）。

    Returns:
        问答结果 dict。
    """
    # P33a：Supervisor 编排开关（默认关闭；开启时走主 Agent 调度）
    try:
        from agent_base.config import deep_get, load_yaml

        _cfg = load_yaml("configs/app.yaml") or {}
        _supervisor_enabled = bool(deep_get(_cfg, "framework.supervisor.enabled", False))
    except Exception:
        _supervisor_enabled = False

    # P11-03: Redis 缓存（classic 路径原本未接，导致缓存 0%）
    from agent_base.storage.cache import cache_key, get_cached, set_cache, DATA_VERSION
    ck = cache_key(
        request.question,
        {"v": DATA_VERSION, "session": request.session_id or "", "fw": "supervisor" if _supervisor_enabled else "classic"},
    )
    cached = get_cached(ck)
    if cached:
        return cached

    # P33a：Supervisor 编排开关（默认关闭；开启时走主 Agent 调度，缓存已共用）
    if _supervisor_enabled:
        from agent_base.agents.graph_supervisor import run_supervisor_graph as run_supervisor

        result = run_supervisor(
            request.question,
            runtime,
            constraints,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        if result.get("answer"):
            set_cache(
                ck,
                {
                    "answer": result["answer"],
                    "safety": {},
                    "catalog_resolution": constraints.get("catalog_resolution"),
                    "trace": {"results": []},
                },
            )
        return result

    cfg = RetrievalConfig.from_runtime(runtime)
    cfg.top_k = request.top_k
    cfg.candidate_k = request.candidate_k
    cfg.rerank = request.rerank
    cfg.product_name = constraints["product_name"]
    cfg.product_spec = constraints["product_spec"]
    cfg.category = constraints["category"]
    result = answer_question_with_trace(
        request.question,
        runtime["vector_store"],
        cfg,
        summary_store=runtime["summary_store"],
        sparse_store=runtime["sparse_store"],
    )
    payload = result.to_dict()
    payload["catalog_resolution"] = constraints["catalog_resolution"]
    # P11 v0.19.1：轻量缓存——只存 answer + safety + 来源摘要（≤500 chars），不阻塞请求
    if payload.get("answer"):
        light = {
            "answer": payload["answer"],
            "safety": payload.get("safety", {}),
            "catalog_resolution": payload.get("catalog_resolution"),
            "trace": {"results": light_results(payload.get("trace", {}).get("results", []))},
        }
        set_cache(ck, light)
    return payload


def light_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """轻量缓存用：只保留结果核心字段（去 preview 大文本）。"""
    light: list[dict[str, Any]] = []
    for item in results:
        light.append({
            "rank": item.get("rank"),
            "section": item.get("section"),
            "source_file": item.get("source_file"),
            "rerank_score": item.get("rerank_score"),
        })
    return light
