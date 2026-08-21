"""全链路评测引擎（意图 → 检索 → 生成 → 打分 → 落库）。

取代旧的单点意图测试（P21 intent_eval）：每个用例跑完整问答链路，
从四个维度自动打分：
- 意图命中（intent）：路由意图是否与预期一致；
- 召回命中（recall）：预期来源（商品/文档关键词）是否出现在检索证据中；
- 事实命中（fact）：预期事实点（价格/克重/成分等关键词）是否出现在回答中；
- 合规（compliance）：回答是否出现医疗承诺/绝对化用语等红线。

结果写入 eval_runs / eval_cases（PG），管理端可查看历史批次与明细。
"""

from __future__ import annotations

from typing import Any, Callable


# ── 合规违禁词（红线：医疗承诺 / 绝对化用语） ──

COMPLIANCE_FORBIDDEN = [
    "治疗", "治愈", "治好", "药妆", "药用", "根治", "百分之百",
    "100%", "包好", "保证见效", "最有效", "最好用", "第一品牌",
]


def _check_compliance(answer: str) -> bool:
    """回答合规检查：命中红线违禁词即失败。"""
    if not answer:
        return True
    return not any(w in answer for w in COMPLIANCE_FORBIDDEN)



def classify_failure(entry):
    """Failure attribution: retrieval / generation / knowledge gap / intent / system."""
    if entry.get("error"):
        return "system_error", str(entry["error"])[:200]
    exp_intent = entry.get("expected_intent") or ""
    if exp_intent and not entry.get("intent_hit"):
        return "intent_miss", "intent expected {0}, actual {1}".format(exp_intent, entry.get("actual_intent") or "unknown")
    exp_src = entry.get("expected_source") or ""
    if exp_src and not entry.get("recall_hit"):
        return "retrieval_fail", "expected source not recalled: {0}".format(exp_src)
    facts = entry.get("expected_facts") or []
    fact_total = int(entry.get("fact_total", 0) or 0)
    if facts and fact_total and int(entry.get("fact_hits", 0)) < fact_total:
        return "generation_fail", "evidence recalled but facts {0}/{1}".format(entry.get("fact_hits"), fact_total)
    if float(entry.get("faithfulness", 0) or 0) < 0.5 and entry.get("sources"):
        return "generation_fail", "low faithfulness: {0}".format(entry.get("faithfulness"))
    if not exp_src and not facts and not entry.get("recall_hit"):
        return "knowledge_gap", "knowledge base lacks content"
    return "", ""


def _source_text(sources: list[dict[str, Any]]) -> str:
    """把检索来源拼接成文本，用于召回命中判断。"""
    parts = []
    for s in sources:
        parts.append(str(s.get("doc_name") or s.get("section") or ""))
        parts.append(str(s.get("content") or s.get("preview") or ""))
    return "\n".join(parts)


def generate_chain_cases(
    intents: list[dict[str, Any]],
    per_intent: int = 2,
) -> list[dict[str, Any]]:
    """AI 动态生成全链路评测用例（每意图 N 条，含预期来源与事实点）。

    与单点意图评测不同：每条用例带 expected_source（检索应命中的商品/文档关键词）
    和 expected_facts（回答应包含的事实点），这样意图/召回/事实/合规四维都能打分。

    Returns:
        [{question, expected_intent, expected_source, expected_facts}]；
        生成失败返回空列表（调用方回退 PG 用例定义）。
    """
    try:
        from agent_base.llms import build_chat_model
    except Exception:
        return []

    # 真实商品清单：约束 source/facts 必须来自真实商品，避免 AI 编造不存在的商品/数值
    try:
        from agent_base.storage.pg import _conn

        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM catalog ORDER BY id")
            product_list = "、".join(str(r[0]) for r in cur.fetchall() if r[0])
    except Exception:
        product_list = ""

    intent_desc = "\n".join(
        f"- {it.get('intent')}: 关键词={','.join((it.get('keywords') or [])[:8])}；"
        f"示例={','.join((it.get('examples') or [])[:2])}"
        for it in intents
        if it.get("intent")
    )
    prompt = (
        "你是电商客服全链路评测用例生成器。为下面每个意图生成电商客服测试问题，"
        f"每个意图 {per_intent} 条，输出 JSON 数组。\n"
        "要求：\n"
        "1. 问题必须是真实用户会问的电商问题（价格/成分/肤质/搭配/售后/订单/闲聊），简短自然；\n"
        "2. 每条用例包含：\n"
        "   - question：测试问题；\n"
        "   - intent：所属意图名（必须是下面列表中的 intent）；\n"
        "   - source：必须取自下方真实商品清单（如\"玻尿酸保湿精华液\"\"UPF50+防晒衣\"；闲聊/售后泛问题给空字符串）；\n"
        "   - facts：必须是该商品真实相关的属性事实（成分/肤质/功效/尺码/防晒值等，不编造价格数值；闲聊给空数组）；\n"
        "3. 尽量覆盖真实商品与常见问法，避免重复。\n"
        "只输出 JSON：\n"
        "[{\"question\": \"...\", \"intent\": \"...\", \"source\": \"...\", \"facts\": [...]}]\n\n"
        "真实商品清单（source 只从这里取）：\n" + (product_list or "（无）") + "\n\n"
        "意图定义：\n" + intent_desc
    )
    try:
        model = build_chat_model(
            provider="langchain", model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            api_key_env="ANTHROPIC_AUTH_TOKEN", temperature=0.4,
        )
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        chain = ChatPromptTemplate.from_messages([("user", "{prompt}")]) | model | StrOutputParser()
        text = chain.invoke({"prompt": prompt}).strip()
        import json as _json
        import re

        cleaned = re.sub(r"```(?:json)?", "", text).strip()
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start < 0 or end <= start:
            return []
        data = _json.loads(cleaned[start:end + 1])
        if not isinstance(data, list):
            return []
        cases = []
        valid_intents = {it.get("intent") for it in intents if it.get("intent")}
        for item in data:
            if not isinstance(item, dict):
                continue
            q = str(item.get("question", "")).strip()
            intent = str(item.get("intent", "")).strip()
            if not q or intent not in valid_intents:
                continue
            cases.append({
                "question": q,
                "expected_intent": intent,
                "expected_source": str(item.get("source", "")).strip(),
                "expected_facts": [str(f) for f in (item.get("facts") or []) if str(f).strip()],
                "expect_compliance": False,
            })
        return cases
    except Exception:
        return []


def _lc_evaluator(
    question: str,
    answer: str,
    contexts: str,
) -> tuple[float, float]:
    """LangChain LCEL 评测链：faithfulness（忠实度）+ answer_relevancy（相关性）。

    langchain 1.x 已移除官方 evaluation 模块（评测收敛到 LangSmith），
    这里用官方 LCEL 组件（ChatPromptTemplate | model | StrOutputParser）搭建
    同款评测链：DeepSeek 评判回答是否忠实于检索证据、是否切题，输出 0-1 分。

    Returns:
        (faithfulness, relevancy)；任一失败返回 (0.0, 0.0)。
    """
    try:
        from agent_base.llms import build_chat_model
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        model = build_chat_model(
            provider="langchain", model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            api_key_env="ANTHROPIC_AUTH_TOKEN", temperature=0.0,
        )
        if model is None:
            return 0.0, 0.0

        faithfulness_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "你是 RAG 评测员。判断「回答」中的每个事实点是否都能在「检索证据」中找到依据。\n"
                "规则：回答里出现证据中不存在的具体信息（数字/成分/功效/承诺）即为幻觉，扣分。\n"
                "只输出 JSON：{{'score': 0.0-1.0}}"
            )),
            ("user", "检索证据：\n{contexts}\n\n回答：\n{answer}"),
        ])
        relevancy_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "你是 RAG 评测员。判断「回答」是否切题、是否完整回应了「问题」。\n"
                "只输出 JSON：{{'score': 0.0-1.0}}"
            )),
            ("user", "问题：\n{question}\n\n回答：\n{answer}"),
        ])

        def _parse_score(text: str) -> float:
            import json as _json
            import re

            cleaned = re.sub(r"```(?:json)?", "", str(text)).strip()
            m = re.search(r"\{.*\}", cleaned, re.S)
            if m:
                try:
                    data = _json.loads(m.group(0))
                    return max(0.0, min(1.0, float(data.get("score", 0))))
                except Exception:
                    pass
            m2 = re.search(r"(\d+(?:\.\d+)?)", cleaned)
            if m2:
                try:
                    return max(0.0, min(1.0, float(m2.group(1))))
                except Exception:
                    pass
            return 0.0

        f_chain = faithfulness_prompt | model | StrOutputParser()
        r_chain = relevancy_prompt | model | StrOutputParser()
        ctx = (contexts or "")[:4000]
        f_score = _parse_score(f_chain.invoke({"contexts": ctx, "answer": (answer or "")[:1500]}))
        r_score = _parse_score(r_chain.invoke({"question": question, "answer": (answer or "")[:1500]}))
        return f_score, r_score
    except Exception:
        return 0.0, 0.0


RAGAS_METRIC_KEYS = ("faithfulness", "relevancy", "context_precision", "context_recall")

# 进程级熔断：ragas 在个别环境（如 Windows 事件循环问题、网络黑洞）可能长时间
# 挂起；连续多个用例全部失败后本进程不再尝试 ragas，评测链路永远可完成。
# 单个用例失败只丢弃该例 ragas 分数（网络抖动可能只影响单例），不熔断整轮。
# 超时按"推理型判官模型"校准：真实用例（长上下文、多条语句）下 faithfulness
# 单指标实测 60~120s，240s 给足余量；挂死场景每例最多浪费 240s。
_RAGAS_BROKEN = False
_RAGAS_STRIKE = 0
_RAGAS_STRIKE_LIMIT = 3
_RAGAS_METRIC_TIMEOUT_S = 240.0


class _SyncJudgeLLM:
    """ragas 判官 LLM：同步 HTTP 调用在 executor 线程中执行。

    绕过 LangchainLLMWrapper 的异步 agenerate_prompt 路径——受限网络/代理环境
    下 httpx 异步客户端偶发挂起（同步客户端实测 100% 可用）。生成统一走
    model.invoke（同步），ragas 的 generate_multiple 按 BaseRagasLLM 分支消费
    resp.generations[0][i].text。
    """

    def __init__(self, langchain_model: Any) -> None:
        self._model = langchain_model

    async def generate(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> Any:
        """对齐 ragas BaseRagasLLM.generate 契约：返回 generations=[[Generation, ...]]。"""
        import asyncio

        from langchain_core.outputs import Generation, LLMResult

        texts: list[str] = []
        for _ in range(max(1, int(n or 1))):
            try:
                resp = await asyncio.to_thread(self._model.invoke, prompt.to_string())
                texts.append(str(getattr(resp, "content", resp) or ""))
            except Exception as exc:
                raise RuntimeError(f"sync judge call failed: {type(exc).__name__}: {exc}") from exc
        return LLMResult(generations=[[Generation(text=t) for t in texts]])


def _ragas_score_with_timeout(metric: Any, sample: Any, timeout_s: float) -> float | None:
    """单指标打分（守护线程 + 超时护栏，挂起只丢弃该指标不拖垮评测）。"""
    import threading

    holder: dict[str, float | None] = {}

    def _work() -> None:
        try:
            holder["value"] = float(metric.single_turn_score(sample))
        except Exception:
            holder["value"] = None

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        return None
    return holder.get("value")


def _ragas_scores(
    question: str,
    answer: str,
    contexts: str,
    reference: str = "",
) -> dict[str, float]:
    """RAGAS 0.4.3 四指标打分（faithfulness / answer_relevancy / context_precision / context_recall）。

    与自研 LCEL 判官互补：LCEL 快而省（每例 2 次 LLM），RAGAS 是公开标准口径
    （每例多次 LLM + embedding，成本更高），由 eval.ragas_enabled 配置门控。
    context_recall 需要参考回答（reference），未提供时跳过该指标。

    Args:
        question: 用户问题。
        answer: 生成回答。
        contexts: 检索证据文本（多个片段以空行分隔）。
        reference: 参考回答（评测用例的 expected_facts 拼接；空则跳过 context_recall）。

    Returns:
        {指标名: 0~1 分数}；ragas 不可用或全部失败时返回 {}（调用方回退 LCEL）。
    """
    global _RAGAS_BROKEN, _RAGAS_STRIKE
    if _RAGAS_BROKEN:
        return {}
    if not answer.strip():
        return {}
    try:
        from agent_base.vendor import ragas_compat  # noqa: F401  # 兼容 shim（必须先于 ragas 导入）

        from ragas import SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

        # Windows 下 ragas 默认的 nest_asyncio 补丁会死锁，替换为普通 asyncio.run
        ragas_compat.disable_nest_asyncio_patch()
    except Exception:
        return {}
    try:
        from agent_base.config import deep_get, load_yaml
        from agent_base.embeddings import build_embeddings
        from agent_base.llms import build_chat_model

        _cfg = load_yaml("configs/app.yaml") or {}
        _emb_cfg = deep_get(_cfg, "embedding", {}) or {}
        _llm_cfg = deep_get(_cfg, "eval_ragas_llm", {}) or {}
        model = build_chat_model(
            provider=_llm_cfg.get("provider") or "langchain",
            model=_llm_cfg.get("model") or "deepseek-v4-flash",
            base_url=_llm_cfg.get("base_url"),
            api_key_env=_llm_cfg.get("api_key_env") or "ANTHROPIC_AUTH_TOKEN",
            temperature=0.0,
            timeout=float(_llm_cfg.get("timeout", 60)),
            max_retries=int(_llm_cfg.get("max_retries", 1)),
        )
        if model is None:
            return {}
        embeddings = build_embeddings(
            provider=_emb_cfg.get("provider", "hash"),
            model=_emb_cfg.get("model"),
            dimensions=int(_emb_cfg.get("dimensions", 512)),
            base_url=_emb_cfg.get("base_url"),
            api_key_env=_emb_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
            keep_alive=int(_emb_cfg.get("keep_alive", 1800)),
        )
        ragas_llm = _SyncJudgeLLM(model)
        ragas_emb = LangchainEmbeddingsWrapper(embeddings)
    except Exception:
        return {}

    ctx_items = [c.strip() for c in (contexts or "").split("\n\n") if c.strip()]
    if not ctx_items and contexts.strip():
        ctx_items = [contexts.strip()[:4000]]
    sample = SingleTurnSample(
        user_input=question,
        response=answer[:2000],
        retrieved_contexts=ctx_items[:8],
        reference=reference.strip() or None,
    )

    def _score(metric: Any) -> float | None:
        value = _ragas_score_with_timeout(metric, sample, _RAGAS_METRIC_TIMEOUT_S)
        if value is None:
            return None
        return round(max(0.0, min(1.0, value)), 3)

    scores: dict[str, float] = {}
    faithfulness = _score(Faithfulness(llm=ragas_llm, max_retries=1))
    if faithfulness is None:
        # 首个指标即挂起：本用例放弃 ragas（仅记一次失败，连续多次才熔断整轮）
        _RAGAS_STRIKE += 1
        if _RAGAS_STRIKE >= _RAGAS_STRIKE_LIMIT:
            _RAGAS_BROKEN = True
        return {}
    scores["faithfulness"] = faithfulness
    relevancy = _score(AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb))
    if relevancy is not None:
        scores["relevancy"] = relevancy
    precision = _score(ContextPrecision(llm=ragas_llm, max_retries=1))
    if precision is not None:
        scores["context_precision"] = precision
    if sample.reference:
        recall = _score(ContextRecall(llm=ragas_llm, max_retries=1))
        if recall is not None:
            scores["context_recall"] = recall
    if scores:
        _RAGAS_STRIKE = 0  # 有真实产出即清零失败计数
    else:
        _RAGAS_STRIKE += 1
        if _RAGAS_STRIKE >= _RAGAS_STRIKE_LIMIT:
            _RAGAS_BROKEN = True
    return scores


def run_chain_eval(
    custom_cases: list[dict[str, Any]] | None = None,
    name: str = "",
    progress_cb: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    use_ragas: bool | None = None,
) -> dict[str, Any]:
    """执行全链路评测并落库。

    Args:
        custom_cases: 自定义用例 [{question, expected_intent, expected_source?, expected_facts?}]；
            缺省完全交给 AI 按意图动态生成（失败回退意图 examples）。
        name: 批次名称（前端可传「意图 X 全链路评测」）。
        progress_cb: 进度回调 (done, total, phase)。
        is_cancelled: 取消检查回调（返回 True 停止评测，供前端取消按钮）。
        use_ragas: 是否启用 RAGAS 四指标打分；None 时按配置（默认关闭，走自研 LCEL 判官）。
        is_cancelled: 取消回调，True 时中止。

    Returns:
        {"run_id", "name", "total_cases", 各维度准确率, "overall", "cases": 明细, ...}。
    """
    def _progress(done: int, total: int, phase: str = "评测") -> None:
        if progress_cb is not None:
            try:
                progress_cb(done, total, phase)
            except Exception:
                pass

    def _cancelled() -> bool:
        return bool(is_cancelled is not None and is_cancelled())

    # RAGAS 开关：显式传参优先，否则读 configs/app.yaml eval.ragas_enabled（默认关闭）
    ragas_enabled = bool(use_ragas)
    if use_ragas is None:
        try:
            from agent_base.config import deep_get, load_yaml

            _cfg = load_yaml("configs/app.yaml") or {}
            ragas_enabled = bool(deep_get(_cfg, "eval.ragas_enabled", False))
        except Exception:
            ragas_enabled = False

    from agent_base.retrieval.intent_router import route_question
    from agent_base.api.main import get_runtime
    from agent_base.chains import answer_question_with_trace
    from agent_base.storage.pg import eval_case_insert, eval_run_insert

    if isinstance(custom_cases, list) and custom_cases:
        generation = "custom"
        cases = []
        for c in custom_cases:
            q = str(c.get("question", "")).strip()
            if not q:
                continue
            cases.append({
                "name": str(c.get("name", "") or q[:12]),
                "question": q,
                "expected_intent": str(c.get("expected_intent", "")),
                "expected_source": str(c.get("expected_source", "")),
                "expected_facts": [str(f) for f in (c.get("expected_facts") or [])],
                "expect_compliance": bool(c.get("expect_compliance")),
            })
    else:
        # 完全交给 AI：按意图动态生成用例（含来源/事实点）
        _progress(0, 1, "AI 生成用例")
        from agent_base.storage.pg import intent_list

        intents = intent_list()
        cases = generate_chain_cases(intents, per_intent=2)
        if _cancelled():
            return {"cancelled": True, "cases": []}
        generation = "ai" if cases else ""
        # AI 生成失败/为空 → 回退意图 examples（仅 question + 预期意图，来源/事实维度跳过）
        if not cases:
            for it in intents:
                for ex in (it.get("examples") or [])[:2]:
                    cases.append({
                        "name": str(ex)[:12],
                        "question": str(ex),
                        "expected_intent": str(it.get("intent", "")),
                        "expected_source": "",
                        "expected_facts": [],
                        "expect_compliance": False,
                    })
            generation = "examples"

    runtime = get_runtime()
    vector_store = runtime["vector_store"]
    summary_store = runtime.get("summary_store")
    sparse_store = runtime.get("sparse_store")
    # 用与生产链路一致的配置，并强制 LLM 生成（评测真实回答质量，而非本地模板）
    from agent_base.retrieval.retrieval_config import RetrievalConfig

    cfg = RetrievalConfig.from_runtime(runtime)
    cfg.llm.use_llm = True

    results: list[dict[str, Any]] = []
    total = len(cases)
    for idx, case in enumerate(cases):
        if _cancelled():
            return {"cancelled": True, "cases": results}
        _progress(idx, total, "评测")
        entry: dict[str, Any] = {
            "question": case["question"],
            "expected_intent": case["expected_intent"],
            "expected_source": case["expected_source"],
            "expected_facts": case["expected_facts"],
            "expect_compliance": bool(case.get("expect_compliance")),
        }
        try:
            route = route_question(case["question"])
            entry["actual_intent"] = route.intent
            entry["intent_hit"] = route.intent == case["expected_intent"]

            res = answer_question_with_trace(
                case["question"],
                vector_store,
                cfg,
                summary_store=summary_store,
                sparse_store=sparse_store,
            )
            answer = str(res.answer or "")
            entry["answer"] = answer
            # 召回来源：trace.docs / safety 侧来源
            sources: list[dict[str, Any]] = []
            try:
                trace = res.trace
                for d in (getattr(trace, "docs", None) or []):
                    meta = getattr(d, "metadata", None) or {}
                    if isinstance(meta, dict):
                        sources.append({
                            "doc_name": meta.get("doc_name", ""),
                            "section": meta.get("section", ""),
                            "content": str(getattr(d, "page_content", "") or "")[:200],
                        })
            except Exception:
                pass
            entry["sources"] = sources

            src_text = _source_text(sources)
            exp_src = entry["expected_source"]
            entry["recall_hit"] = (not exp_src) or (exp_src in src_text)

            facts = entry["expected_facts"]
            hit = sum(1 for f in facts if f in answer)
            entry["fact_hits"] = hit
            entry["fact_total"] = len(facts)

            entry["compliance_ok"] = _check_compliance(answer)

            # LangChain LCEL 评测链：faithfulness（忠实度）+ relevancy（相关性）
            f_score, r_score = _lc_evaluator(case["question"], answer, src_text)
            entry["faithfulness"] = round(f_score, 3)
            entry["relevancy"] = round(r_score, 3)

            # RAGAS 四指标（配置门控）：标准公开口径，与 LCEL 判官互补
            if ragas_enabled:
                reference = "；".join(str(f) for f in facts) if facts else ""
                entry["ragas"] = _ragas_scores(case["question"], answer, src_text, reference=reference)
        except Exception as exc:  # noqa: BLE001
            entry.update({
                "actual_intent": "error",
                "intent_hit": False,
                "recall_hit": False,
                "fact_hits": 0,
                "fact_total": len(case["expected_facts"]),
                "compliance_ok": True,
                "faithfulness": 0.0,
                "relevancy": 0.0,
                "error": str(exc)[:160],
            })
            if ragas_enabled:
                entry["ragas"] = {}
        results.append(entry)
        _progress(idx + 1, total, "评测")

    n = max(1, len(results))
    intent_acc = round(sum(1 for r in results if r.get("intent_hit")) / n, 4)
    recall_acc = round(sum(1 for r in results if r.get("recall_hit")) / n, 4)
    fact_acc = round(
        sum(r.get("fact_hits", 0) for r in results)
        / max(1, sum(r.get("fact_total", 0) for r in results)),
        4,
    )
    compliance_acc = round(sum(1 for r in results if r.get("compliance_ok")) / n, 4)
    faithfulness_acc = round(sum(float(r.get("faithfulness", 0)) for r in results) / n, 4)
    relevancy_acc = round(sum(float(r.get("relevancy", 0)) for r in results) / n, 4)
    overall = round(
        intent_acc * 0.25 + recall_acc * 0.15 + fact_acc * 0.15
        + compliance_acc * 0.10 + faithfulness_acc * 0.20 + relevancy_acc * 0.15,
        4,
    )
    # RAGAS 聚合：按实际跑出分数的用例数求均值（个别用例失败不拉低整体）
    ragas_acc: dict[str, float] = {}
    if ragas_enabled:
        for dim in RAGAS_METRIC_KEYS:
            values = [float(r["ragas"][dim]) for r in results if (r.get("ragas") or {}).get(dim) is not None]
            if values:
                ragas_acc[dim] = round(sum(values) / len(values), 4)

    run_id = 0
    try:
        run_id = eval_run_insert(
            name=name or f"全链路评测 {len(results)} 用例",
            total_cases=len(results),
            intent_acc=intent_acc,
            recall_acc=recall_acc,
            fact_acc=fact_acc,
            compliance_acc=compliance_acc,
            faithfulness_acc=faithfulness_acc,
            relevancy_acc=relevancy_acc,
            overall=overall,
            ragas_faithfulness_acc=ragas_acc.get("faithfulness", 0),
            ragas_relevancy_acc=ragas_acc.get("relevancy", 0),
            ragas_context_precision_acc=ragas_acc.get("context_precision", 0),
            ragas_context_recall_acc=ragas_acc.get("context_recall", 0),
        )
        for r in results:
            case_id = eval_case_insert(
                run_id=run_id,
                question=r["question"],
                expected_intent=r.get("expected_intent", ""),
                actual_intent=r.get("actual_intent", ""),
                intent_hit=bool(r.get("intent_hit")),
                expected_source=r.get("expected_source", ""),
                recall_hit=bool(r.get("recall_hit")),
                expected_facts=r.get("expected_facts", []),
                fact_hits=int(r.get("fact_hits", 0)),
                fact_total=int(r.get("fact_total", 0)),
                compliance_ok=bool(r.get("compliance_ok")),
                faithfulness=float(r.get("faithfulness", 0)),
                relevancy=float(r.get("relevancy", 0)),
                ragas_faithfulness=float((r.get("ragas") or {}).get("faithfulness", 0)),
                ragas_relevancy=float((r.get("ragas") or {}).get("relevancy", 0)),
                ragas_context_precision=float((r.get("ragas") or {}).get("context_precision", 0)),
                ragas_context_recall=float((r.get("ragas") or {}).get("context_recall", 0)),
                answer=r.get("answer", ""),
                sources=r.get("sources", []),
                error=r.get("error", ""),
            )
            failure_type, detail = classify_failure(r)
            if failure_type:
                try:
                    from agent_base.storage.pg import upsert_eval_feedback

                    upsert_eval_feedback(
                        case_id=int(case_id or 0),
                        run_id=int(run_id or 0),
                        question=r.get("question", ""),
                        failure_type=failure_type,
                        detail=detail,
                        status="pending",
                    )
                except Exception:
                    pass
    except Exception:
        pass

    payload: dict[str, Any] = {
        "run_id": run_id,
        "name": name or "全链路评测",
        "generation": generation,
        "total_cases": len(results),
        "intent_acc": intent_acc,
        "recall_acc": recall_acc,
        "fact_acc": fact_acc,
        "compliance_acc": compliance_acc,
        "faithfulness_acc": faithfulness_acc,
        "relevancy_acc": relevancy_acc,
        "overall": overall,
        "cases": results,
    }
    if ragas_enabled:
        payload["ragas_enabled"] = True
        payload["ragas"] = {f"{dim}_acc": value for dim, value in ragas_acc.items()}
    return payload
