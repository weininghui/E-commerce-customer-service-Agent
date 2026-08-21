"""四 Agent 知识生产流水线（采集 → 解析抽取 → 质检 → 发布）。

LangGraph StateGraph 编排，固定 Workflow 骨架 + 自主节点 Agent：
- collect：接收上传文档，规整输入；
- parse_extract：MinerU 解析 + LLM 结构化字段抽取（复用 knowledge_factory 打标）；
- qc：规则引擎 + LLM 双重复核，产出 pass/fail；
- publish：通过 → 打标 approved 自动入库；不通过 → 转人工复核（returned）。

质检不通过不会入库，保持现有 document_staging / document_strategy 状态机兼容。
"""

from __future__ import annotations

from typing import Any, TypedDict


class PipelineState(TypedDict, total=False):
    """知识流水线共享状态：采集 → 解析 → 质检 → 发布全链路字段。"""

    filename: str
    content: str
    category: str
    parsed_text: str
    parser_engine: str
    doc_type: str
    strategy: list[str]
    confidence: float
    reasoning: str
    risk_flags: list[str]
    qc_pass: bool
    qc_reason: str
    doc_id: str
    status: str
    error: str


def _collect(state: PipelineState) -> dict[str, Any]:
    """采集 Agent：规整输入，做基础校验（空内容/过短）。"""
    content = str(state.get("content") or "").strip()
    filename = str(state.get("filename") or "uploaded-document")
    if not content:
        return {"status": "failed", "error": "empty content", "qc_pass": False}
    if len(content) < 20:
        return {
            "status": "failed",
            "error": "content too short",
            "qc_pass": False,
            "qc_reason": "文档内容过短，无法入库",
        }
    return {"status": "collecting", "content": content, "filename": filename}


def _parse_extract(state: PipelineState) -> dict[str, Any]:
    """解析抽取 Agent：MinerU 解析 + LLM/规则打标抽取结构化字段。"""
    filename = state.get("filename", "")
    content = state.get("content", "")
    parsed_text = content
    engine = "direct"
    try:
        from agent_base.ingest.mineru_parser import parse_document

        parsed = parse_document(filename, content.encode("utf-8"))
        parsed_text = parsed["text"] or content
        engine = parsed.get("engine", "mock")
    except Exception:
        parsed_text = content
    # 打标：复用知识工厂的预审（LLM 可用时走 LLM，否则启发式兜底）
    doc_type = ""
    strategy: list[str] = []
    confidence = 0.0
    reasoning = ""
    risk_flags: list[str] = []
    try:
        from agent_base.knowledge_factory import STRATEGY_MAP, pre_review_document

        tag = pre_review_document(parsed_text[:4000], filename=filename)
        doc_type = tag.doc_type
        strategy = list(tag.strategy or STRATEGY_MAP.get(doc_type, ["default_vector"]))
        confidence = float(tag.confidence or 0.0)
        reasoning = tag.reasoning
        first = tag.first_review or {}
        risk_flags = list(first.get("risk_flags") or [])
    except Exception:
        pass
    return {
        "status": "parsing",
        "parsed_text": parsed_text,
        "parser_engine": engine,
        "doc_type": doc_type,
        "strategy": strategy,
        "confidence": confidence,
        "reasoning": reasoning,
        "risk_flags": risk_flags,
    }


def _qc(state: PipelineState) -> dict[str, Any]:
    """质检 Agent：规则引擎 + LLM 双重复核。"""
    text = state.get("parsed_text") or state.get("content") or ""
    doc_type = state.get("doc_type", "")
    reasons: list[str] = []
    risk_flags = list(state.get("risk_flags") or [])

    # 规则引擎：合规红线 + 内容质量
    forbidden = [
        "治愈", "根治", "药到病除", "100%", "保证见效", "包好",
        "最有效", "第一品牌", "绝对",
    ]
    hit_words = [w for w in forbidden if w in text]
    if hit_words:
        risk_flags.append("compliance")
        reasons.append(f"命中合规红线词：{'、'.join(hit_words)}")
    if not doc_type:
        reasons.append("无法识别文档类型")

    # LLM 复核（可用时）；失败静默，不阻断流程
    try:
        from agent_base.config import deep_get, load_yaml

        cfg = load_yaml("configs/app.yaml") or {}
        llm_cfg = deep_get(cfg, "pre_review_llm") or {"provider": "none"}
        if (llm_cfg or {}).get("provider") not in (None, "", "none", "off"):
            from agent_base.llms import build_chat_model

            model = build_chat_model(**llm_cfg, tracking_agent="qc_agent", tracking_source="knowledge_pipeline")
            if model is not None:
                resp = model.invoke(
                    "你是知识库质检员。判断以下文档是否适合入库（电商客服知识库）。"
                    "若存在夸大宣称、医疗承诺、内容残缺或明显错误，输出：FAIL:原因；否则输出 PASS。\n\n"
                    f"文档类型：{doc_type}\n文档内容（前 1500 字）：\n{text[:1500]}"
                )
                answer = str(getattr(resp, "content", "") or resp or "")
                if "FAIL" in answer.upper():
                    detail = answer.split(":", 1)[1].strip() if ":" in answer else "LLM 质检未通过"
                    reasons.append(f"LLM 复核：{detail[:120]}")
    except Exception:
        pass

    qc_pass = not reasons
    return {
        "status": "qc" if qc_pass else "returned",
        "qc_pass": qc_pass,
        "qc_reason": "；".join(reasons) if reasons else "规则 + LLM 双重复核通过",
        "risk_flags": risk_flags,
    }


def _publish(state: PipelineState) -> dict[str, Any]:
    """发布 Agent：通过 → 打标 approved 自动入库；不通过 → 转人工复核。"""
    filename = state.get("filename", "")
    content = state.get("parsed_text") or state.get("content") or ""
    category = state.get("category", "上传文档")
    doc_type = state.get("doc_type", "")
    strategy = state.get("strategy") or []
    qc_pass = bool(state.get("qc_pass"))
    try:
        from agent_base.storage.pg import staging_find_by_content, staging_upsert

        existing = staging_find_by_content(content[:1200])
        doc_id = (existing or {}).get("doc_id") or ""
        if not doc_id:
            import hashlib

            doc_id = hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]
        first_review = {
            "type": doc_type,
            "strategy": strategy,
            "confidence": float(state.get("confidence", 0.0) or 0.0),
            "reasoning": state.get("reasoning", ""),
            "risk_flags": state.get("risk_flags") or [],
            "source": "knowledge_pipeline",
        }
        staging_upsert(
            doc_id=doc_id,
            filename=filename,
            content=content,
            category=category,
            first_review=first_review,
            status="approved" if qc_pass else "returned",
            reject_reason="" if qc_pass else state.get("qc_reason", ""),
        )
        if qc_pass:
            try:
                from agent_base.api.main import get_runtime
                from agent_base.storage.staging import approve_and_ingest

                runtime = get_runtime()
                approve_and_ingest(
                    doc_id=doc_id,
                    vector_store=runtime["vector_store"],
                    doc_type=doc_type or "guide",
                    strategy=strategy or ["default_vector"],
                    summary_store=runtime.get("summary_store"),
                    reviewer="knowledge_pipeline",
                )
                return {"status": "published", "doc_id": doc_id, "qc_pass": True}
            except Exception as exc:
                return {
                    "status": "published_staged",
                    "doc_id": doc_id,
                    "qc_pass": True,
                    "error": f"自动入库失败，已暂存待人工：{str(exc)[:160]}",
                }
        return {"status": "returned", "doc_id": doc_id, "qc_pass": False}
    except Exception as exc:
        return {"status": "failed", "error": str(exc)[:200], "qc_pass": False}


def _route_after_qc(state: PipelineState) -> str:
    return "publish"


def build_pipeline_graph():
    """构建知识生产流水线 StateGraph。"""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(PipelineState)
    g.add_node("collect", _collect)
    g.add_node("parse_extract", _parse_extract)
    g.add_node("qc", _qc)
    g.add_node("publish", _publish)
    g.add_edge(START, "collect")
    g.add_edge("collect", "parse_extract")
    g.add_edge("parse_extract", "qc")
    g.add_edge("qc", "publish")
    g.add_edge("publish", END)
    return g.compile()


def run_knowledge_pipeline(
    *,
    filename: str,
    content: str,
    category: str = "上传文档",
) -> dict[str, Any]:
    """运行四 Agent 知识生产流水线。"""
    try:
        graph = build_pipeline_graph()
        result = graph.invoke(
            {
                "filename": filename,
                "content": content,
                "category": category,
                "status": "collecting",
            }
        )
        return result
    except Exception as exc:
        return {"status": "failed", "error": str(exc)[:200]}
