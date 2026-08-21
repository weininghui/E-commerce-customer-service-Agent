"""问答链路：检索 → 安全评估 → 生成（经典模式）。

对外入口 answer_question：先 retrieve_advanced 取证据，
再 assess_safety 做合规门禁，最后可选用 LLM 生成受控回答。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent_base.chains.safety_chain import SafetyAssessment, assess_safety
from agent_base.config import load_yaml
from agent_base.llms import build_chat_model
from agent_base.retrieval.advanced_retriever import AdvancedRetrievalTrace, retrieve_advanced
from agent_base.retrieval.retrieval_config import RetrievalConfig


DEFAULT_QA_SYSTEM_PROMPT = (
    "你是美妆服饰电商客服问答系统的受控生成层。只能依据给定的商品资料和 FAQ 证据回答，"
    "不能补充证据外的信息，不做功效承诺。"
    "答案用简洁的标准 Markdown 组织：结论先行，要点用 - 分条，对比/方案可用表格，"
    "每块不超过 3 行，总长控制在 250 字以内。"
    "禁止出现“结论/依据/购买建议/来源/安全等级/风险标签”等内部字段名；"
    "禁止客套废话与空话，资料缺失时只写一句“资料未标注该数值”。"
    "安全提示和来源保留在答案末尾（用引用块）。"
)

DEFAULT_QA_USER_TEMPLATE = """用户问题：
{question}

系统初步结论：
{conclusion}

商品/FAQ 资料：
{evidence}

购买/使用建议：
{guidance}

安全等级：{risk_level}
风险标签：{risk_tags}
安全提示：
{safety_warnings}

来源：
{sources}
"""


@dataclass(slots=True)
class AnswerResult:
    """一次问答的完整结果：回答文本 + 检索 trace + 安全评估。"""

    answer: str
    trace: AdvancedRetrievalTrace
    safety: SafetyAssessment

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典（trace/safety 嵌套展开）。"""
        return {
            "answer": self.answer,
            "trace": self.trace.to_dict(),
            "safety": self.safety.to_dict(),
        }


def answer_question(
    question: str,
    vector_store: Any,
    cfg: RetrievalConfig | None = None,
    *,
    summary_store: Any | None = None,
    sparse_store: Any | None = None,
    current_product: str | None = None,
) -> str:
    """只返回答案文本的简化入口，内部复用 answer_question_with_trace。"""
    return answer_question_with_trace(
        question,
        vector_store,
        cfg,
        summary_store=summary_store,
        sparse_store=sparse_store,
        current_product=current_product,
    ).answer


def answer_question_with_trace(
    question: str,
    vector_store: Any,
    cfg: RetrievalConfig | None = None,
    *,
    summary_store: Any | None = None,
    sparse_store: Any | None = None,
    current_product: str | None = None,
) -> AnswerResult:
    """带 Trace 的完整问答主流程。

    这个函数是后端 RAG 问答链路的核心入口。它不只返回答案文本，还会返回：
    - trace：检索过程，包括意图路由、问题重写、metadata filter、召回来源等。
    - safety：安全评估结果，包括风险等级和风险标签。

    可以把它理解成 5 个连续阶段：
    1. 检索：retrieve_advanced 负责意图路由、问题重写、多路召回、rerank。
    2. 证据筛选：_answer_docs 从召回结果中挑选真正用于回答的原文证据。
    3. 多商品保护：如果未指定商品且召回跨多个商品，则按商品分组回答。
    4. 安全评估：assess_safety 根据问题和证据判断风险等级。
    5. 答案生成：根据 use_llm 选择受控 LLM 生成或本地模板生成。
    """
    # 第一步：执行高级检索。
    # retrieve_advanced 返回 AdvancedRetrievalTrace，而不是单纯的 docs。
    # Trace 中既有最终入选的 trace.docs，也有检索中间过程：
    # - trace.rewrite：问题改写与意图路由结果
    # - trace.metadata_filter：实际使用的 Chroma metadata 过滤条件
    # - trace.decision：auto 模式下系统选择的检索策略
    # - trace.results：前端“数据来源与检索 Trace”里展示的结构化结果
    cfg = cfg or RetrievalConfig()
    # 配置对象 → 局部变量（保持函数体其余部分不变）
    use_llm = cfg.llm.use_llm
    llm_provider = cfg.llm.provider
    llm_model = cfg.llm.model
    llm_base_url = cfg.llm.base_url
    llm_api_key_env = cfg.llm.api_key_env
    llm_temperature = cfg.llm.temperature
    llm_evidence_max_chars_per_doc = cfg.llm.evidence_max_chars_per_doc
    llm_evidence_max_total_chars = cfg.llm.evidence_max_total_chars
    llm_evidence_max_chars_per_product = cfg.llm.evidence_max_chars_per_product
    llm_evidence_max_chars_per_product_doc = cfg.llm.evidence_max_chars_per_product_doc
    prompts_path = cfg.prompts_path
    product_name = cfg.product_name
    product_spec = cfg.product_spec

    trace = retrieve_advanced(
        vector_store,
        question,
        cfg,
        summary_store=summary_store,
        sparse_store=sparse_store,
        current_product=current_product,
    )

    # 第二步：从 trace.docs 中挑选“真正用于回答”的证据。
    # retrieve_advanced 可能返回 metadata/vector/summary/summary_guided 多种阶段的结果。
    # _answer_docs 会优先选择命中目标章节的“原文 chunk”，尽量不直接依赖摘要生成答案。
    docs = _answer_docs(trace.docs, trace.rewrite.route.sections)

    if _needs_clarification_response(trace):
        assessment = assess_safety(question)
        clarification_msg = _build_clarification_guidance(trace)
        # 澄清场景返回自然语言追问，不套用结构化模板（结论/依据/建议/安全等级）
        answer = clarification_msg
        return AnswerResult(answer=answer, trace=trace, safety=assessment)

    # 如果检索没有拿到任何可用证据，不能让模型自由发挥。
    # 这里直接返回一个安全兜底答案，并把空检索的 trace 一起返回，方便前端排查。
    if not docs:
        # 没有证据时，只能基于用户问题本身做最低限度的安全评估。
        assessment = assess_safety(question)
        # 使用本地模板生成“证据不足”的回答。
        answer = _format_answer(
            conclusion="当前商品资料中未检索到足够信息，不能据此给出明确建议。",
            evidence="未找到相关商品/FAQ 片段。",
            guidance="请核对商品名称、规格和使用场景后重新提问，或直接联系在线客服。",
            safety=assessment,
            sources="无",
        )
        # AnswerResult 是接口层最终返回的数据结构：答案 + 检索 trace + 安全评估。
        return AnswerResult(answer=answer, trace=trace, safety=assessment)

    # 第三步：为“多商品检索结果”准备证据。
    # trace.docs 里可能包含多个商品的结果，尤其是用户只问“推荐”“适合敏感肌吗”
    # 但没有限定具体商品时。这个函数会为每个商品保留最合适的证据。
    multi_product_docs = _multi_product_answer_docs(trace.docs, trace.rewrite.route.sections)

    # 判断是否需要进入“多商品分组回答”。
    # 触发条件大致是：
    # - 用户问题是商品相关意图，如 product_query/recommendation 等。
    # - 用户没有明确指定 product_name/product_spec。
    # - 检索结果跨多个 doc_id 或多个商品 identity。
    if _needs_multi_product_answer(trace, multi_product_docs, product_name=product_name, product_spec=product_spec):
        # 用户没有限定具体商品，但检索命中多个商品：不能任选一个结论，必须按商品分组列出。
        # 安全评估使用的上下文优先取路由章节相关内容。
        # 如果相关章节内容为空，则退回使用所有多商品证据文本。
        safety_context = _safety_context(multi_product_docs, trace.rewrite.route.sections) or "\n\n".join(
            _doc_text(doc) for doc in multi_product_docs
        )
        # 根据用户问题和召回证据判断风险等级。
        # 例如过敏、敏感肌、孕妇/儿童使用、功效承诺等会提升风险等级。
        assessment = assess_safety(question, safety_context)
        # 按商品维度分组，避免把 A 商品结论套用到 B 商品。
        grouped_docs = _group_docs_by_product(multi_product_docs)

        # 如果配置启用了 LLM，则走“受控 LLM 生成”。
        # 注意这里不是让 LLM 自由检索，而是把已经整理好的结论、证据、建议、来源填入 prompt。
        if use_llm:
            from agent_base.agents.sales import build_sales_strategy

            sales_strategy = build_sales_strategy(question, trace.rewrite.route.intent)
            answer = _controlled_llm_answer(
                # 原始用户问题。
                question=question,
                # 多商品场景的总说明：提示用户本次检索跨多个商品，不能混用结论。
                conclusion=_build_multi_product_conclusion(trace.rewrite.route.intent, grouped_docs),
                # 给 LLM 使用更长、更完整的多商品证据，避免只根据压缩展示片段生成答案。
                evidence=_build_multi_product_llm_evidence(
                    question=question,
                    grouped_docs=grouped_docs,
                    route_sections=trace.rewrite.route.sections,
                    max_chars_per_product=llm_evidence_max_chars_per_product,
                    max_chars_per_doc=llm_evidence_max_chars_per_product_doc,
                ),
                # 多商品场景下的购买/使用建议。
                guidance=_build_multi_product_guidance(assessment),
                # 安全评估结果会进入 prompt，但最终正文会去掉“安全等级/风险标签”重复展示。
                safety=assessment,
                # 按商品整理的来源信息。
                sources=_format_multi_product_sources(grouped_docs),
                # LLM 运行参数，来自 configs/app.yaml 和环境变量。
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                llm_api_key_env=llm_api_key_env,
                llm_temperature=llm_temperature,
                prompts_path=prompts_path,
                sales_strategy=sales_strategy,
            )
            # 多商品 + LLM 分支直接返回结果。
            return AnswerResult(answer=answer, trace=trace, safety=assessment)

        # 如果没有启用 LLM，则使用本地模板生成多商品答案。
        answer = _format_answer(
            conclusion=_build_multi_product_conclusion(trace.rewrite.route.intent, grouped_docs),
            evidence=_compact_multi_product_evidence(grouped_docs),
            guidance=_build_multi_product_guidance(assessment),
            safety=assessment,
            sources=_format_multi_product_sources(grouped_docs),
        )
        # 多商品 + 模板分支直接返回结果。
        return AnswerResult(answer=answer, trace=trace, safety=assessment)

    # 第四步：单商品或已明确商品约束的正常问答流程。

    # 把最终用于回答的文档内容拼成上下文。
    # context 主要用于安全评估兜底，不直接等同于最终回答。
    context = "\n\n".join(_doc_text(doc) for doc in docs)

    # 安全评估优先使用路由命中的目标章节上下文。
    # 例如敏感肌/过敏问题优先看“成分/注意事项/售后FAQ”，
    # 避免无关章节里的词触发错误风险判断。
    safety_context = _safety_context(docs, trace.rewrite.route.sections) or context

    # 根据用户问题和商品资料证据做安全评估。
    # 返回 SafetyAssessment，包含 risk_level、findings、warnings、must_consult 等。
    assessment = assess_safety(question, safety_context)

    # 构造“结论”。
    conclusion = _build_conclusion(question, docs, assessment)

    # 构造“依据商品/FAQ 资料”部分，使用 _compact_evidence 压缩原文 chunk。
    evidence = _compact_evidence(docs)

    # 第五步：生成最终答案。
    # 如果启用了 LLM，走受控生成层；否则走本地模板答案。
    if use_llm:
        # LLM 输入和页面展示使用不同证据版本：
        # - 页面/模板展示：evidence，短一些，方便用户阅读。
        # - LLM 生成：llm_evidence，长一些，尽量保留完整关键句、条款和命中章节。
        llm_evidence = _build_llm_evidence(
            question=question,
            docs=docs,
            route_sections=trace.rewrite.route.sections,
            max_chars_per_doc=llm_evidence_max_chars_per_doc,
            max_total_chars=llm_evidence_max_total_chars,
        )
        from agent_base.agents.sales import build_sales_strategy

        sales_strategy = build_sales_strategy(question, trace.rewrite.route.intent)
        answer = _controlled_llm_answer(
            # 用户原始问题。
            question=question,
            # 上一步已经确定好的结论，LLM 只能基于它和证据组织语言。
            conclusion=conclusion,
            # 给 LLM 的证据包，不再使用 _compact_evidence 的 320 字展示截断。
            evidence=llm_evidence,
            # 根据风险等级生成的购买/使用建议。
            guidance=_build_guidance(question, docs, assessment),
            # 安全评估对象。
            safety=assessment,
            # 来源信息，用于让答案保留可追溯性。
            sources=_format_sources(docs),
            # LLM 参数。
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key_env=llm_api_key_env,
            llm_temperature=llm_temperature,
            prompts_path=prompts_path,
            sales_strategy=sales_strategy,
        )
        # 单商品 + LLM 分支返回。
        return AnswerResult(answer=answer, trace=trace, safety=assessment)

    # 未启用 LLM 时，使用稳定的模板答案。
    # 这个分支适合课堂演示“纯检索 + 规则模板”的可控效果，也能作为 LLM 失败时的兜底。
    answer = _format_answer(
        conclusion=conclusion,
        evidence=evidence,
        guidance=_build_guidance(question, docs, assessment),
        safety=assessment,
        sources=_format_sources(docs),
    )
    # 单商品 + 模板分支返回。
    return AnswerResult(answer=answer, trace=trace, safety=assessment)


def _structured_adverse_frequency_parts(question: str, docs: list[Any]) -> dict[str, str] | None:
    """结构化提取「不良反应/发生率」要点（药品/健康类目专用）。

    从检索命中的片段里筛出含不良反应频率关键词的文本，组装成结构化
    结论 + 证据，供生成节点优先引用；未命中返回 None（走通用结论）。

    Args:
        question: 用户问题。
        docs: 检索命中文档列表。

    Returns:
        {"conclusion": ..., "evidence": ...}；未命中返回 None。
    """
    import re as _re

    pattern = _re.compile(r"(不良反应|发生率|常见|偶见|罕见|十分常见|禁忌|慎用)")
    hits: list[str] = []
    for d in docs or []:
        text = str(getattr(d, "page_content", "") or "")
        if pattern.search(text):
            hits.append(text.strip())
    if not hits:
        return None
    evidence = "\n".join(h[:400] for h in hits[:3])
    return {
        "conclusion": "已按说明书/资料整理不良反应与注意事项要点，请以引用片段为准。",
        "evidence": evidence,
    }


def _build_conclusion(question: str, docs: list[Any], assessment: SafetyAssessment) -> str:
    """基于问题类型和检索证据生成抽取式结论，用作兜底答案或 LLM 初稿。"""
    if any(keyword in question for keyword in ["过敏", "泛红", "刺痛", "发痒", "红肿", "皮疹"]):
        risk_section = _first_section_text(docs, "注意事项") or _first_section_text(docs, "成分")
        if risk_section:
            return _first_sentence(risk_section)
    if any(keyword in question for keyword in ["尺码", "面料", "版型", "怎么选尺码"]):
        fit_section = _first_section_text(docs, "商品参数") or _first_section_text(docs, "搭配建议")
        if fit_section:
            return _first_sentence(fit_section)
    if any(keyword in question for keyword in ["成分", "功效", "适合", "肤质"]):
        spec_section = _first_section_text(docs, "商品参数") or _first_section_text(docs, "卖点")
        if spec_section:
            return _first_sentence(spec_section)
    if any(keyword in question for keyword in ["退货", "退款", "换货", "物流", "发票", "发货", "快递"]):
        faq_section = _first_section_text(docs, "售后FAQ")
        if faq_section:
            return _first_sentence(faq_section)
    if assessment.risk_level == "high":
        return "该问题涉及过敏或高风险使用场景，请优先按商品资料的注意事项和成分信息处理，必要时咨询专业人士。"
    return "以下为根据商品资料检索到的相关信息，供参考。"


def _needs_multi_product_answer(
    trace: AdvancedRetrievalTrace,
    docs: list[Any] | None = None,
    product_name: str | None = None,
    product_spec: str | None = None,
) -> bool:
    """判断是否需要多商品分组回答。

    如果问题是商品相关意图、没有显式商品约束，并且证据跨多个商品 identity，
    则进入多商品回答，避免把 A 商品结论误套到 B 商品。
    """
    product_specific_intents = {
        "product_query",
        "fashion_query",
        "recommendation",
    }
    if trace.rewrite.route.intent not in product_specific_intents:
        return False
    if product_name or product_spec:
        return False
    answer_docs = docs or trace.docs
    if _question_mentions_product_identity(trace.question, answer_docs):
        return False
    return _has_multiple_products(answer_docs)


def _question_mentions_product_identity(question: str, docs: list[Any]) -> bool:
    normalized_question = _normalize_identity_text(question)
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        names = [
            metadata.get("product_name"),
            metadata.get("product_spec"),
            metadata.get("source_file"),
        ]
        for name in names:
            normalized_name = _normalize_identity_text(str(name or ""))
            if normalized_name and normalized_name != "unknown" and normalized_name in normalized_question:
                return True
    return False


def _has_multiple_products(docs: list[Any]) -> bool:
    identities = set()
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        identity = (
            metadata.get("doc_id"),
            metadata.get("product_spec"),
            metadata.get("product_name"),
        )
        if any(identity):
            identities.add(identity)
    return len(identities) > 1


def _group_docs_by_product(docs: list[Any]) -> list[dict[str, Any]]:
    """按 doc_id + product_spec + product_name 对召回证据分组。"""
    groups: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        key = (
            metadata.get("doc_id"),
            metadata.get("product_spec"),
            metadata.get("product_name"),
        )
        if not any(key):
            key = (metadata.get("source_file"), None, None)
        if key not in groups:
            groups[key] = {
                "doc_id": metadata.get("doc_id"),
                "product_name": metadata.get("product_name") or "未知商品名",
                "product_spec": metadata.get("product_spec") or "未知通用名",
                "source_file": metadata.get("source_file") or "未知文件",
                "docs": [],
            }
        groups[key]["docs"].append(doc)
    return list(groups.values())


def _build_multi_product_conclusion(intent: str, grouped_docs: list[dict[str, Any]]) -> str:
    intent_names = {
        "product_query": "商品参数/功效",
        "fashion_query": "穿搭/尺码/面料",
        "price_query": "价格/优惠",
        "aftersale": "售后政策",
        "recommendation": "推荐理由",
    }
    topic = intent_names.get(intent, "相关问题")
    count = len(grouped_docs)
    return (
        f"本次问题未限定具体商品，系统在商品知识库中检索到 {count} 个相关商品。"
        f"不同商品的{topic}可能不同，下面按商品分别列出资料依据，不能把其中一个商品的结论套用到其他商品。"
    )


def _compact_multi_product_evidence(grouped_docs: list[dict[str, Any]], max_chars_per_product: int = 360) -> str:
    """为多商品回答压缩证据，每个商品输出若干条章节片段。"""
    lines = []
    for idx, group in enumerate(grouped_docs, start=1):
        display_name = _product_display_name(group)
        snippets = []
        seen = set()
        for doc in group["docs"]:
            metadata = getattr(doc, "metadata", {}) or {}
            section = metadata.get("section", "未知章节")
            text = _doc_text(doc).replace("\n", " ").strip()
            if not text:
                continue
            snippet = f"[{section}] {_first_sentence(text)}"
            if snippet not in seen:
                snippets.append(snippet)
                seen.add(snippet)
            if sum(len(item) for item in snippets) >= max_chars_per_product:
                break
        evidence = "；".join(snippets)
        if len(evidence) > max_chars_per_product:
            evidence = evidence[:max_chars_per_product].rstrip() + "..."
        lines.append(f"{idx}. {display_name}：{evidence or '检索到相关章节，但片段内容为空。'}")
    return "\n".join(lines)


def _build_multi_product_guidance(assessment: SafetyAssessment) -> str:
    if assessment.risk_level == "high":
        return "已按商品分别列出结果；过敏、敏感肌、孕妇/儿童使用等高风险场景请谨慎购买，必要时咨询专业医生或在线客服。"
    return "请先确认要购买或使用的是哪一个商品，再按对应商品资料核对；不同商品即使类目相近，也不能共用功效、成分或使用结论。"


def _format_multi_product_sources(grouped_docs: list[dict[str, Any]]) -> str:
    lines = []
    for idx, group in enumerate(grouped_docs, start=1):
        source_lines = _format_sources(group["docs"]).replace("\n", "；")
        lines.append(f"{idx}. {_product_display_name(group)}：{source_lines}")
    return "\n".join(lines)


def _product_display_name(group: dict[str, Any]) -> str:
    product_name = group.get("product_name") or "未知商品名"
    product_spec = group.get("product_spec") or "未知规格"
    if product_name == product_spec:
        return product_name
    return f"{product_name}（{product_spec}）"


def _normalize_identity_text(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("®", "")
        .replace("庐", "")
    )


def _build_guidance(question: str, docs: list[Any], assessment: SafetyAssessment) -> str:
    if assessment.risk_level == "high":
        return "不建议仅凭自动问答下单或使用；请核对商品资料中的注意事项和成分，过敏或不适时及时停止使用并咨询专业医生。"
    if assessment.risk_level == "medium":
        return "购买/使用前请重点核对商品参数、成分、注意事项和售后政策，敏感人群先咨询专业人士。"
    return "以上信息仅供参考，具体效果因人而异；如有皮肤不适请及时停止使用并咨询专业人士。"


def _needs_clarification_response(trace: AdvancedRetrievalTrace) -> bool:
    """判断是否需要澄清追问。

    触发条件：retrieval_policy 设置了 need_clarification=True
    且 strategy 为 clarification 或 catalog_search。
    """
    decision = trace.decision
    if decision is None:
        return False
    return bool(decision.need_clarification) and decision.strategy in {"clarification", "catalog_search"}


def _build_clarification_guidance(trace: AdvancedRetrievalTrace) -> str:
    """从决策中的 clarification_question 取澄清追问文本。

    优先使用决策中由 catalog 生成的候选商品提示，
    兜底使用通用追问提示。
    """
    decision = trace.decision
    if decision and decision.clarification_question:
        return decision.clarification_question
    return "您想了解哪款产品呢？请说出具体的商品名称，我为您查询详情。"


def _compact_evidence(docs: list[Any], max_chars_per_doc: int = 320) -> str:
    """生成页面展示/模板兜底用的短证据。

    这个函数只负责“展示层压缩”，不再作为 LLM 的主要证据输入。
    企业落地中，如果直接把每条证据截断到 320 字再交给 LLM，长成分表、长参数、注意事项、售后政策等内容可能被截掉，进而影响最终答案完整性。
    """
    lines = []
    for idx, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        section = metadata.get("section", "未知章节")
        text = _doc_text(doc).replace("\n", " ")
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc].rstrip() + "..."
        lines.append(f"{idx}. [{section}] {text}")
    return "\n".join(lines)


def _build_llm_evidence(
    question: str,
    docs: list[Any],
    route_sections: list[str] | None = None,
    max_chars_per_doc: int = 1200,
    max_total_chars: int = 6000,
) -> str:
    """生成专门给受控 LLM 使用的证据包。

    和 _compact_evidence 的区别：
    - _compact_evidence 面向页面展示，每条默认 320 字，强调可读性。
    - _build_llm_evidence 面向 LLM 生成，每条默认 1200 字，强调证据完整性。

    设计原则：
    1. 优先保留意图路由命中的章节，例如“商品参数”“成分”“售后FAQ”。
    2. 对长文本不做粗暴后半截删除，而是抽取包含问题关键词的完整句/条款。
    3. 如果没有匹配到关键词，则保留 chunk 开头的完整片段作为兜底。
    4. 长表格类问题（成分表/尺码表）在 evidence 中保留完整表格片段，
       不走粗暴的普通文本截断逻辑。
    """
    route_sections = route_sections or []
    ordered_docs = _prioritize_docs_for_llm(docs, route_sections)
    lines: list[str] = []
    total_chars = 0

    for idx, doc in enumerate(ordered_docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        section = metadata.get("section", "未知章节")
        source_file = metadata.get("source_file", "未知文件")
        page_start = metadata.get("page_start", "?")
        page_end = metadata.get("page_end", page_start)
        text = _normalize_evidence_text(_doc_text(doc))
        if not text:
            continue

        packed_text = _pack_doc_text_for_llm(
            question=question,
            text=text,
            max_chars=max_chars_per_doc,
        )
        line = f"{idx}. [{section}]（{source_file}，第 {page_start}-{page_end} 页）{packed_text}"
        if total_chars + len(line) > max_total_chars:
            remaining = max_total_chars - total_chars
            if remaining > 200:
                lines.append(line[:remaining].rstrip() + "...")
            break
        lines.append(line)
        total_chars += len(line)

    return "\n".join(lines)


def _build_multi_product_llm_evidence(
    question: str,
    grouped_docs: list[dict[str, Any]],
    route_sections: list[str] | None = None,
    max_chars_per_product: int = 1800,
    max_chars_per_doc: int = 900,
) -> str:
    """为多商品 LLM 回答构造更完整的分组证据。

    页面展示仍使用 _compact_multi_product_evidence；LLM 生成时则需要每个商品保留更长证据，
    否则容易只回答某一个商品或漏掉后续商品的重要资料。
    """
    lines: list[str] = []
    for idx, group in enumerate(grouped_docs, start=1):
        display_name = _product_display_name(group)
        evidence = _build_llm_evidence(
            question=question,
            docs=group["docs"],
            route_sections=route_sections,
            max_chars_per_doc=max_chars_per_doc,
            max_total_chars=max_chars_per_product,
        )
        lines.append(f"{idx}. {display_name}\n{evidence or '检索到相关章节，但片段内容为空。'}")
    return "\n".join(lines)


def _prioritize_docs_for_llm(docs: list[Any], route_sections: list[str]) -> list[Any]:
    """按章节相关性给 LLM 证据排序，命中路由章节的证据优先。"""
    if not route_sections:
        return docs
    route_set = set(route_sections)
    return sorted(
        docs,
        key=lambda doc: 0 if (getattr(doc, "metadata", {}) or {}).get("section") in route_set else 1,
    )


def _pack_doc_text_for_llm(question: str, text: str, max_chars: int) -> str:
    """在单个 chunk 内为 LLM 选择较完整的证据文本。"""
    if len(text) <= max_chars:
        return text

    focused = _focused_evidence_excerpt(question, text, max_chars=max_chars)
    if focused:
        return focused
    return text[:max_chars].rstrip() + "..."


def _focused_evidence_excerpt(question: str, text: str, max_chars: int) -> str:
    """优先抽取包含问题关键词的完整句/条款，避免按固定字符数切断关键语义。"""
    keywords = _question_keywords(question)
    if not keywords:
        return ""

    units = _split_evidence_units(text)
    selected: list[str] = []
    total = 0
    for unit in units:
        if not any(keyword in unit for keyword in keywords):
            continue
        addition = unit.strip()
        if not addition:
            continue
        if total + len(addition) > max_chars:
            break
        selected.append(addition)
        total += len(addition)

    if not selected:
        return ""
    excerpt = " ".join(selected)
    return excerpt if len(excerpt) <= max_chars else excerpt[:max_chars].rstrip() + "..."


def _split_evidence_units(text: str) -> list[str]:
    """把证据文本切成尽量完整的句子或条款。"""
    units = re.split(r"(?<=[。；;])\s*|\n+", text)
    return [unit.strip() for unit in units if unit.strip()]


def _question_keywords(question: str) -> list[str]:
    """从用户问题中提取用于定位证据句的关键词。"""
    stopwords = {
        "这个",
        "那个",
        "哪些",
        "什么",
        "怎么",
        "是否",
        "可以",
        "能够",
        "有没有",
        "有哪些",
        "一下",
    }
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9.%＜<>~～-]{2,}", question)
    keywords = [token for token in tokens if token not in stopwords]
    return list(dict.fromkeys(keywords))


def _normalize_evidence_text(text: str) -> str:
    """清理证据中的多余空白，保留句子和条款顺序。"""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _answer_docs(docs: list[Any], route_sections: list[str]) -> list[Any]:
    """选择最终单商品回答使用的证据。

    优先选择命中目标章节的原文 chunk；如果没有，再退回所有原文 chunk；
    最后才使用摘要结果，避免最终答案过度依赖摘要。
    """
    if route_sections:
        section_docs = []
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            if metadata.get("retrieval_stage") != "summary" and metadata.get("section") in route_sections:
                section_docs.append(doc)
        if section_docs:
            return section_docs

    source_docs = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        if metadata.get("retrieval_stage") != "summary":
            source_docs.append(doc)
    return source_docs or docs


def _multi_product_answer_docs(docs: list[Any], route_sections: list[str]) -> list[Any]:
    """为多商品回答保留每个商品的最佳证据。

    每个商品优先使用目标章节原文，其次使用任意原文，再其次使用摘要。
    这样可以避免 Trace 中命中了多个商品，但最终回答只剩一个商品。
    """
    selected = []
    for group in _group_docs_by_product(docs):
        group_docs = group["docs"]
        route_source_docs = _filter_docs(group_docs, route_sections=route_sections, include_summary=False)
        source_docs = _filter_docs(group_docs, route_sections=None, include_summary=False)
        route_docs = _filter_docs(group_docs, route_sections=route_sections, include_summary=True)
        selected.extend(route_source_docs or source_docs or route_docs or group_docs)
    return selected


def _filter_docs(
    docs: list[Any],
    route_sections: list[str] | None = None,
    include_summary: bool = True,
) -> list[Any]:
    filtered = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        if not include_summary and metadata.get("retrieval_stage") == "summary":
            continue
        if route_sections and metadata.get("section") not in route_sections:
            continue
        filtered.append(doc)
    return filtered


def _format_sources(docs: list[Any]) -> str:
    """把证据来源格式化为文件名、页码、章节和检索阶段。"""
    sources = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        source_file = metadata.get("source_file", "未知文件")
        page_start = metadata.get("page_start", "?")
        page_end = metadata.get("page_end", page_start)
        section = metadata.get("section", "未知章节")
        stage = metadata.get("retrieval_stage", "retrieval")
        sources.append(f"{source_file}，第 {page_start}-{page_end} 页，{section}，检索阶段：{stage}")
    return "\n".join(dict.fromkeys(sources))


def _format_answer(
    conclusion: str,
    evidence: str,
    guidance: str,
    safety: SafetyAssessment,
    sources: str,
) -> str:
    safety_lines = "\n".join(f"- {warning}" for warning in safety.warnings)
    return (
        f"{conclusion}\n\n"
        f"**商品资料要点**\n{evidence}\n\n"
        f"**建议**\n{guidance}\n\n"
        + (f"> 安全提示：{safety_lines}\n" if safety_lines else "")
        + f"> 来源：{sources}"
    )


def _first_section_text(docs: list[Any], section: str) -> str:
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        if metadata.get("section") == section:
            return _doc_text(doc)
    return ""


def _safety_context(docs: list[Any], route_sections: list[str]) -> str:
    if not route_sections:
        return "\n\n".join(_doc_text(doc) for doc in docs)
    relevant = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        if metadata.get("section") in route_sections:
            relevant.append(_doc_text(doc))
    return "\n\n".join(relevant)


def _first_sentence(text: str) -> str:
    for sep in ["。", "；", "\n"]:
        if sep in text:
            return text.split(sep, 1)[0].strip() + ("。" if sep == "。" else "")
    return text.strip()



def _doc_text(doc: Any) -> str:
    return getattr(doc, "page_content", str(doc))


def _controlled_llm_answer(
    question: str,
    conclusion: str,
    evidence: str,
    guidance: str,
    safety: SafetyAssessment,
    sources: str,
    llm_provider: str,
    llm_model: str | None,
    llm_base_url: str | None,
    llm_api_key_env: str,
    llm_temperature: float,
    prompts_path: str | None,
    sales_strategy: str = "",
) -> str:
    """受控 LLM 生成层。

    该函数不会直接让模型自由发挥，而是把抽取式结论、证据、安全提示和来源
    填入 prompts.yaml 中的模板。模型输出后还会清理“安全等级/风险标签”，
    因为这两个字段由前端来源区单独展示。
    """
    fallback = _format_answer(
        conclusion=conclusion,
        evidence=evidence,
        guidance=guidance,
        safety=safety,
        sources=sources,
    )
    chat = build_chat_model(
        provider=llm_provider,
        model=llm_model,
        base_url=llm_base_url,
        api_key_env=llm_api_key_env,
        temperature=llm_temperature,
    )
    if chat is None:
        return fallback
    safety_lines = "\n".join(f"- {warning}" for warning in safety.warnings)
    findings = ", ".join(finding.label for finding in safety.findings) or "none"
    prompt_config = _load_qa_prompt_config(prompts_path)
    try:
        # LCEL 官方链：ChatPromptTemplate | model | StrOutputParser
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        system_prompt = prompt_config["system"]
        if sales_strategy:
            system_prompt = system_prompt + "\n\n" + sales_strategy
        chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("user", prompt_config["user_template"]),
                ]
            )
            | chat
            | StrOutputParser()
        )
        raw = chain.invoke(
            {
                "question": question,
                "conclusion": conclusion,
                "evidence": evidence,
                "guidance": guidance,
                "risk_level": safety.risk_level,
                "risk_tags": findings,
                "safety_warnings": safety_lines,
                "sources": sources,
            }
        )
        return _strip_answer_safety_summary(raw)
    except Exception as exc:
        # 生产安全：异常详情只进日志/trace，不回显给买家（避免暴露内部实现）
        try:
            from agent_base.monitoring.logger import log_event

            log_event("ERROR", "qa_chain", "controlled_llm_failed", {
                "error": f"{type(exc).__name__}: {exc}"[:300],
            })
        except Exception:
            pass
        return fallback


def _load_qa_prompt_config(prompts_path: str | None) -> dict[str, str]:
    """读取 qa.system 和 qa.user_template；缺失时回退到代码内默认提示词。"""
    config = _load_prompts_yaml(prompts_path) if prompts_path else {}
    qa_config = config.get("qa", {}) if isinstance(config, dict) else {}
    return {
        "system": qa_config.get("system") or DEFAULT_QA_SYSTEM_PROMPT,
        "user_template": qa_config.get("user_template") or DEFAULT_QA_USER_TEMPLATE,
    }


@lru_cache(maxsize=8)
def _load_prompts_yaml(prompts_path: str) -> dict[str, Any]:
    path = Path(prompts_path)
    if not path.exists():
        return {}
    try:
        return load_yaml(path)
    except Exception:
        return {}


def _strip_answer_safety_summary(answer: str) -> str:
    """从 LLM 正文中移除安全等级和风险标签，避免和 Trace 来源区重复展示。"""
    text = str(answer)
    lines = text.splitlines()
    cleaned: list[str] = []
    skip_next_value = False
    for line in lines:
        stripped = line.strip()
        normalized = stripped.rstrip("：:")
        if normalized in {"安全等级", "风险标签"}:
            skip_next_value = True
            continue
        if stripped.startswith(("安全等级：", "安全等级:", "风险标签：", "风险标签:")):
            continue
        if skip_next_value and stripped and not stripped.endswith(("：", ":")):
            skip_next_value = False
            continue
        skip_next_value = False
        cleaned.append(line)
    return "\n".join(cleaned).strip()
