"""P33a 主 Agent（Supervisor）编排入口。

项目定位：基于 Agent 的 RAG——每个环节是独立子 Agent（一能力一专精），
主 Agent 负责任务分配，子 Agent 之间不直接通信，只与主 Agent 交互，
返回统一契约 AgentResult。

落地路径（渐进，不推翻现有代码）：
- 子 Agent 化：把现有函数（意图/完善/澄清/检索/工具/记忆/生成/反思）
  包装成统一契约的子 Agent；
- 主 Agent 调度：直通（单意图高置信）/ 澄清 / ReAct 工具循环 / TASR 反思；
- 模式补全（P33b）：PER / PAE / Self-Ask 按触发条件接入路由。

红线：工具/转人工/记忆全部下沉对话链路；supervisor 开关默认关闭，
关闭时行为与经典链路完全一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def compute_generation_temperature(intent: str = "general_qa", emotion: str = "neutral") -> float:
    """生成温度 = 意图基底 + 情绪调节（0.05~0.5 区间）。

    动态温度（生产级）：决策类温度恒 0.0（可复现），生成类按任务类型 + 情绪动态调——
    用户越激动温度越低（稳定安抚），越积极温度越高（有感染力）。

    Args:
        intent: 检索意图（product_query/comparison/recommendation/aftersale...）。
        emotion: 情绪标签（anger/anxiety/positive/neutral）。

    Returns:
        0.05~0.5 的生成温度。
    """
    base_map = {
        "product_query": 0.1,
        "price_query": 0.1,
        "comparison": 0.2,
        "recommendation": 0.4,
        "aftersale": 0.15,
        "general_qa": 0.3,
    }
    try:
        from agent_base.config import deep_get, load_yaml

        llm_cfg = (load_yaml("configs/app.yaml") or {}).get("llm", {}) or {}
        overrides = deep_get(llm_cfg, "temperature_by_intent", {}) or {}
        base_map = {**base_map, **overrides}
    except Exception:
        pass
    base = float(base_map.get(intent, 0.2))

    emotion_delta = {
        "anger": -0.1,
        "anxiety": -0.05,
        "positive": 0.1,
        "neutral": 0.0,
    }.get(emotion, 0.0)

    return round(min(0.5, max(0.05, base + emotion_delta)), 2)


# ── 统一契约 ─────────────────────────────────────────────────────────────


@dataclass
class AgentResult:
    """子 Agent 统一返回契约。

    - status: ok / clarify / handoff / retry / fallback
    - data: 子 Agent 特有产出（intent/rewritten/trace/answer...）
    - sources: 供主 Agent 汇总溯源
    - confidence: 置信度 0-1
    - meta: 调度元数据（耗时/来源/警告）
    """

    status: str = "ok"
    data: dict[str, Any] = field(default_factory=dict)
    sources: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转为字典（status/data/sources/confidence/meta 五字段契约）。"""
        return {
            "status": self.status,
            "data": self.data,
            "sources": self.sources,
            "confidence": self.confidence,
            "meta": self.meta,
        }


# ── 子 Agent：一能力一专精 ───────────────────────────────────────────────


def intent_agent(question: str, domain: Any = None) -> AgentResult:
    """意图 Agent：三层意图识别（规则 → LLM → 语义兜底）。"""
    try:
        from agent_base.retrieval.intent_router import route_question

        route = route_question(question, domain=domain)
        return AgentResult(
            data={
                "intent": route.intent,
                "sections": route.sections,
                "matched_keywords": route.matched_keywords,
                "source": route.source,
            },
            confidence=float(route.confidence),
            meta={"agent": "intent", "source": route.source},
        )
    except Exception as exc:  # noqa: BLE001
        return AgentResult(status="fallback", data={"intent": "general_qa"}, meta={"error": str(exc)[:120]})


def enrich_agent(question: str, current_product: str | None = None) -> AgentResult:
    """完善 Agent：别名扩展 + 多轮指代补全（镜像主路径配置门控）。"""
    try:
        from agent_base.config import deep_get, load_yaml
        from agent_base.retrieval.enrich import expand_aliases, resolve_question

        _cfg = load_yaml("configs/app.yaml") or {}
        rewritten = question
        if deep_get(_cfg, "retrieval.enrich_alias.enabled", False):
            rewritten = expand_aliases(rewritten)
        if current_product and deep_get(_cfg, "retrieval.enrich_reference.enabled", False):
            rewritten = resolve_question(rewritten, current_product)
        return AgentResult(
            data={"rewritten": rewritten, "changed": rewritten != question},
            confidence=1.0,
            meta={"agent": "enrich"},
        )
    except Exception as exc:  # noqa: BLE001
        # Enrich 失败不阻断：原问题直通
        return AgentResult(data={"rewritten": question, "changed": False}, meta={"error": str(exc)[:120]})


def clarify_agent(
    question: str,
    product_name: str | None = None,
    product_spec: str | None = None,
) -> AgentResult:
    """澄清 Agent：完整性检查——必须澄清 / 可降级推荐 / 可直接回答。"""
    try:
        from agent_base.retrieval.retrieval_policy import build_retrieval_decision

        _, decision = build_retrieval_decision(
            question,
            product_name=product_name,
            product_spec=product_spec,
        )
        return AgentResult(
            data={
                "need_clarification": decision.need_clarification,
                "route_type": decision.route_type,
                "clarification_question": decision.clarification_question,
                "candidates": getattr(decision, "clarification_question", "") or "",
                "strategy": decision.strategy,
            },
            confidence=1.0,
            status="clarify" if decision.need_clarification else "ok",
            meta={"agent": "clarify"},
        )
    except Exception as exc:  # noqa: BLE001
        return AgentResult(data={"need_clarification": False, "route_type": "answerable"}, meta={"error": str(exc)[:120]})


def memory_agent(session_id: str | None, user_id: str | None) -> AgentResult:
    """记忆 Agent：会话历史 + 用户画像（供生成注入，检索仍按当前问题）。"""
    try:
        from agent_base.storage.chat_memory import get_chat_history
        from agent_base.storage.memory import build_profile_context

        history = get_chat_history(session_id, limit=32) if session_id else []
        profile = build_profile_context(user_id or "", intent="", max_chars=1000) if user_id else ""
        return AgentResult(
            data={"history": history, "profile": profile, "history_count": len(history)},
            confidence=1.0,
            meta={"agent": "memory"},
        )
    except Exception as exc:  # noqa: BLE001
        return AgentResult(data={"history": [], "profile": ""}, meta={"error": str(exc)[:120]})


def tool_agent(question: str, intent: str) -> AgentResult:
    """工具 Agent：订单/物流/库存意图 → 查真实 PG（工具下沉对话链路）。"""
    tool_name, tool_fn, arg = None, None, None
    if intent == "aftersale":
        import re

        from agent_base.agents.tools_ecommerce import get_logistics, get_order

        order = re.search(r"(?:订单|单号)[:：#\s]*([A-Za-z0-9]+)", question)
        if order:
            arg = order.group(1)
            tool_name, tool_fn = ("order_query", get_order) if "物流" not in question else ("logistics_query", get_logistics)
    if tool_fn is None:
        return AgentResult(status="ok", data={"used": False}, confidence=0.0, meta={"agent": "tool"})
    try:
        # B1 修复：@tool 包装的是 StructuredTool（不可直接调用），
        # 用 .func 取底层函数调用（等价于 invoke 单参工具）
        result = tool_fn.func(arg)
        return AgentResult(
            data={"used": True, "tool": tool_name, "result": str(result)[:800]},
            confidence=1.0,
            meta={"agent": "tool"},
        )
    except Exception as exc:  # noqa: BLE001
        return AgentResult(status="fallback", data={"used": False}, meta={"error": str(exc)[:120]})


def retrieve_agent(
    question: str,
    vector_store: Any,
    summary_store: Any | None,
    constraints: dict[str, Any],
    rerank_cfg: dict[str, Any] | None = None,
    intent_classifier: dict[str, Any] | None = None,
    sparse_store: Any | None = None,
) -> AgentResult:
    """检索 Agent：可观察高级检索（复用 retrieve_advanced，含 T6 增强钩子）。"""
    try:
        from agent_base.retrieval import retrieve_advanced
        from agent_base.retrieval.retrieval_config import RerankConfig, RetrievalConfig

        rerank_cfg = rerank_cfg or {}
        cfg = RetrievalConfig(
            top_k=constraints.get("top_k", 6),
            candidate_k=constraints.get("candidate_k"),
            rerank="auto",
            product_name=constraints.get("product_name"),
            product_spec=constraints.get("product_spec"),
            category=constraints.get("category"),
            intent_classifier=intent_classifier,
            rerank_model=RerankConfig(
                provider=rerank_cfg.get("provider", "none"),
                model=rerank_cfg.get("model", "bge-reranker-v2-m3"),
                endpoint=rerank_cfg.get("endpoint"),
                api_key_env=rerank_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
                timeout=int(rerank_cfg.get("timeout", 30)),
                strategies=rerank_cfg.get("use_for_strategies"),
                preserve_preferred_sections=bool(rerank_cfg.get("preserve_preferred_sections", True)),
            ),
        )
        trace = retrieve_advanced(
            vector_store,
            question,
            cfg,
            summary_store=summary_store,
            sparse_store=sparse_store,
        )
        return AgentResult(
            data={"trace": trace.to_dict(include_preview=True)},
            sources=[
                {
                    "doc_name": (_m := (getattr(d, "metadata", {}) or {})).get("doc_name", ""),
                    "section": _m.get("section", ""),
                    "score": _m.get("rerank_score") or _m.get("vector_score"),
                    "content": getattr(d, "page_content", "")[:240],
                }
                for d in (trace.docs or [])[:6]
            ],
            confidence=1.0,
            meta={"agent": "retrieve"},
        )
    except Exception as exc:  # noqa: BLE001
        return AgentResult(status="fallback", data={"trace": {}}, meta={"error": str(exc)[:120]})


def self_ask_agent(
    question: str,
    vector_store: Any,
    k: int = 6,
    metadata_filter: dict[str, Any] | None = None,
) -> AgentResult:
    """Self-Ask 检索 Agent：拆子问题 → 各检索 → RRF 融合（P33b）。"""
    try:
        from agent_base.retrieval.decomposition import decompose_question, self_ask_retrieve

        subs = decompose_question(question) or []
        docs = self_ask_retrieve(question, vector_store, k=k, metadata_filter=metadata_filter)
        return AgentResult(
            data={"sub_questions": subs, "docs_count": len(docs)},
            sources=[
                {
                    "doc_name": (_m := (getattr(d, "metadata", {}) or {})).get("doc_name", ""),
                    "section": _m.get("section", ""),
                    "score": _m.get("rerank_score") or _m.get("vector_score"),
                    "content": getattr(d, "page_content", "")[:400],
                }
                for d in docs[:8]
            ],
            confidence=1.0,
            meta={"agent": "self_ask", "sub_questions": subs},
        )
    except Exception as exc:  # noqa: BLE001
        return AgentResult(status="fallback", data={"docs_count": 0, "sub_questions": []}, meta={"error": str(exc)[:120]})


def generate_agent(
    question: str,
    evidence: str,
    history: list[dict[str, Any]] | None = None,
    profile: str = "",
    tool_result: str = "",
    llm_cfg: dict[str, Any] | None = None,
) -> AgentResult:
    """生成 Agent：受控生成（证据优先，LLM 不可用回退模板摘要）。"""
    llm_cfg = llm_cfg or {}
    provider = llm_cfg.get("provider", "none")
    if provider in {"none", "off", "false"}:
        return AgentResult(
            data={"answer": _template_answer(question, evidence, tool_result)},
            confidence=1.0,
            meta={"agent": "generate", "mode": "template"},
        )
    try:
        from agent_base.llms import build_chat_model
        from agent_base.prompts import get_prompt

        model = build_chat_model(
            provider=provider,
            model=llm_cfg.get("model"),
            base_url=llm_cfg.get("base_url"),
            api_key_env=llm_cfg.get("api_key_env", "ANTHROPIC_AUTH_TOKEN"),
            temperature=float(llm_cfg.get("temperature", 0.1)),
        )
        if model is None:
            raise RuntimeError("model unavailable")
        # LCEL 官方链：ChatPromptTemplate | model | StrOutputParser（P26 单一口径）
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", get_prompt("qa", "system")),
                    ("user", "{context}"),
                ]
            )
            | model
            | StrOutputParser()
        )
        answer = chain.invoke({
            "context": build_generate_context(question, evidence, history, profile, tool_result),
        })
        return AgentResult(
            data={"answer": answer},
            confidence=1.0,
            meta={"agent": "generate", "mode": "llm"},
        )
    except Exception as exc:  # noqa: BLE001
        return AgentResult(
            data={"answer": _template_answer(question, evidence, tool_result)},
            meta={"agent": "generate", "mode": "template_fallback", "error": str(exc)[:120]},
        )


def build_generate_context(
    question: str,
    evidence: str,
    history: list[dict[str, Any]] | None = None,
    profile: str = "",
    tool_result: str = "",
) -> str:
    """构建生成节点的完整用户上下文（历史/画像/工具结果 + 问题 + 证据）。

    P28-2：统一注入函数——supervisor 轻量上下文（max_msgs=6）走四级降级，
    保证口径与 streaming.py 主链路一致（档 1 全量 / 档 2 窗口 / 档 3 规则压缩）。
    generate_agent（LCEL）与官方 create_agent 生成路径共用，双路提示词保持一致。
    """
    history_block = ""
    if history:
        try:
            from agent_base.storage.chat_memory import build_injectable_history, get_context_config

            injectable, _ = build_injectable_history(history, get_context_config(), max_msgs=6)
        except Exception:
            injectable = history[-6:]
        lines = [f"{'用户' if m.get('role') == 'user' else '助手'}：{str(m.get('content', ''))[:400]}" for m in injectable]
        history_block = "对话历史：\n" + "\n".join(lines) + "\n\n"
    tool_block = f"系统查询结果（权威数据，直接采用）：{tool_result}\n\n" if tool_result else ""
    profile_block = f"用户画像：{profile}\n\n" if profile else ""
    evidence_block = f"参考资料：\n{evidence[:6000]}\n\n" if evidence else ""
    return (
        f"{history_block}{profile_block}{tool_block}"
        f"用户问题：{question}\n\n"
        f"{evidence_block}"
        "请像小满一样用自然聊天的方式回答，把资料细节融进话里，不要使用任何章节标题。"
    )



def _template_answer(question: str, evidence: str, tool_result: str = "") -> str:
    """无 LLM 兜底：证据摘要模板回答。"""
    if tool_result:
        return tool_result
    lines = [ln.strip() for ln in evidence.splitlines() if ln.strip()]
    preview = "\n".join(lines[:8])[:600]
    if not preview:
        return "抱歉，未检索到相关资料，请换个问法或提供更多信息。"
    return f"根据商品资料，相关信息如下：\n{preview}"


# ── 主 Agent（Supervisor）：调度 ──────────────────────────────────────────



def _extract_current_product(history: list[dict[str, Any]] | None) -> str | None:
    """从会话历史提取最近提及的商品名（供指代补全）。"""
    if not history:
        return None
    try:
        from agent_base.storage.pg import _conn

        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM catalog WHERE name IS NOT NULL LIMIT 200")
            names = [str(r[0]) for r in cur.fetchall()]
    except Exception:
        return None
    if not names:
        return None
    for m in reversed(history):
        text = str(m.get("content", ""))
        if not text:
            continue
        for name in names:
            if name and name in text:
                return name
    return None


def _detect_mode(question: str, intent: str) -> str:
    """模式路由（P33b）：Self-Ask / PAE / PER / 直通。

    - self_ask：比较/复合句式或多问号 → 拆子问题分别检索
    - per：多步任务 + 需自查（对比+推荐、下单引导）→ 规划→执行→反思
    - pae：确定性流程引导 → 规划→执行（省自查）
    - 其余：直通（工具意图由 tool_agent 处理）
    """
    import re as _re

    if _re.search(r"哪个好|哪个更|和.*比|与.*比|对比|有什么区别|还是.*好|区别", question) or question.count("？") + question.count("?") >= 2:
        return "self_ask"
    if intent in {"comparison", "recommendation"} and any(
        k in question for k in ["怎么选", "推荐", "哪个", "适合"]
    ):
        return "per"
    if any(k in question for k in ["怎么下单", "如何下单", "购买流程", "下单步骤", "怎么买", "如何购买", "怎么退货", "退款流程"]):
        return "pae"
    return "direct"
