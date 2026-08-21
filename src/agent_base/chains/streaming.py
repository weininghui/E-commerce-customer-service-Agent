"""P20 SSE 流式问答生成器（v0.30.0 从 api/main.py 抽出）。

事件序列：sources → trace → [thinking → delta]* → done。
检索复用 classic 链路（use_llm=False 只取证据），生成走 LCEL 官方链
（ChatPromptTemplate | ChatDeepSeek，deepseek-reasoner 透传 reasoning_content——
通用 ChatOpenAI 会丢弃该字段，见 langchain-ai/langchain#35059）。
request 参数为 RagRequest（类型 Any 避免循环依赖）。
"""

from __future__ import annotations

import os
import uuid
from functools import lru_cache
from typing import Any

from agent_base.chains import answer_question_with_trace
from agent_base.retrieval.retrieval_config import RetrievalConfig


def _media_for_stream(
    constraints: dict[str, Any],
    current_product: str | None,
    question: str | None,
    history_products: list[str] | None = None,
) -> list[dict[str, Any]]:
    """批量解析预置商品图，供 SSE media 事件下发（Phase 1 多商品推荐带图）。

    候选商品 ID 提取（去重保序）：
    1. catalog_resolution.matched_products——问题直接命中的商品（单/多商品）；
    2. matched_products 为空时按 matched_categories 类目展开
       （「推荐几款精华」→ 精华类全部商品，一次出多图）；
    3. 会话历史中的全部标准商品名（多轮指代候选）；
    4. 兜底：product_name / current_product / 问题里的 P### 编号。

    单商品守卫：matched_products 非空时不展开类目——类目词可能是商品名子串
    （「玻尿酸保湿精华液」含「精华」），展开会破坏单商品出图语义。
    """
    try:
        import re

        from agent_base.retrieval.multi_source import _resolve_product_ids
        from agent_base.retrieval.retrieval_policy import _load_catalog
        from agent_base.storage.pg import media_for_product_cards, media_for_product_ids

        ids: list[str] = []
        resolution = constraints.get("catalog_resolution") or {}
        if hasattr(resolution, "to_dict"):  # 兼容 CatalogResolution 实例
            resolution = resolution.to_dict()
        # 1) 问题直接命中的商品
        for item in resolution.get("matched_products") or []:
            pid = str(item.get("id") or item.get("doc_id") or "").strip()
            if pid and pid not in ids:
                ids.append(pid)
        # 2) 未命中具体商品 → 类目展开
        if not ids:
            categories = [str(c).strip() for c in (resolution.get("matched_categories") or []) if c]
            if not categories and resolution.get("category"):
                categories = [str(resolution["category"]).strip()]
            if categories:
                catalog = _load_catalog()
                if catalog:
                    for pid, item in (catalog.get("products") or {}).items():
                        if str(item.get("category") or "") in categories and pid not in ids:
                            ids.append(str(pid))
        # 3) 会话历史中的商品名
        for name in history_products or []:
            for pid in _resolve_product_ids(name):
                if pid not in ids:
                    ids.append(pid)
        # 4) 兜底：单商品名 / 当前商品 / P### 编号
        if not ids:
            product_name = resolution.get("product_name") or current_product
            ids = _resolve_product_ids(product_name) if product_name else []
        if not ids:
            q = str(question or "")
            if any(k in q for k in ("衣服", "服饰", "穿搭", "裙子", "裤", "T恤", "衬衫", "开衫", "防晒衣")):
                ids = ["P013", "P014", "P015", "P016", "P017", "P018", "P019", "P020"]
            elif any(k in q for k in ("护肤", "精华", "面霜", "洁面", "防晒", "面膜", "眼霜", "水乳", "润肤油")):
                ids = ["P001", "P002", "P003", "P004", "P005", "P006", "P007", "P008", "P009", "P010", "P011", "P012"]
        if not ids:
            ids = re.findall(r"P\d{3}", str(question or "").upper())
        if not ids:
            return []
        if len(ids) == 1:
            items = media_for_product_ids(ids, limit=8)
        else:
            items = media_for_product_cards(ids, limit=6)
        return _enrich_media_titles(items)
    except Exception:
        return []


def _enrich_media_titles(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """给预置素材补商品名标题（预置 title 是「商品主图」，多图时无法区分商品）。"""
    try:
        from agent_base.retrieval.retrieval_policy import _load_catalog

        catalog = _load_catalog()
        names = (
            {pid: str(item.get("name") or "") for pid, item in (catalog.get("products") or {}).items()}
            if catalog
            else {}
        )
        for m in items:
            title = str(m.get("title") or "")
            pid = str(m.get("product_id") or "")
            if title in ("", "商品主图") and names.get(pid):
                m["title"] = names[pid]
    except Exception:
        pass
    return items


def _media_requested(question: str, sub_intent: str = "") -> bool:
    """媒体按需门控：仅用户明确要看图/视频时才返回媒体。

    Args:
        question: 用户问题。
        sub_intent: 意图层子意图（media_request 或空）。

    Returns:
        是否需要下发媒体。
    """
    if sub_intent == "media_request":
        return True
    try:
        from agent_base.retrieval.intent_router import MEDIA_REQUEST_PATTERNS

        q = (question or "").strip().lower()
        return any(p in q for p in MEDIA_REQUEST_PATTERNS)
    except Exception:
        return False


def _doc_images_from_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从检索命中的 sources 提取图片/视频知识库媒体。

    图片来自 metadata.image_urls；视频来自 metadata.video_urls（含封面/时长）。
    返回可直接下发的 media 项（media_type=image/video，product_id 为空）。
    """
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        for s in sources or []:
            doc_name = str(s.get("doc_name") or "")
            # 图片
            urls = s.get("image_urls") or []
            if isinstance(urls, str):
                urls = [urls]
            for u in urls:
                u = str(u).strip()
                if not u or u in seen:
                    continue
                seen.add(u)
                items.append(
                    {
                        "media_type": "image",
                        "url": u,
                        "product_id": "",
                        "title": f"知识库图片：{doc_name}" if doc_name else "知识库图片",
                    }
                )
            # 视频（封面 poster + 时长给前端播放器用）
            vids = s.get("video_urls") or []
            if isinstance(vids, str):
                vids = [vids]
            poster = str(s.get("poster_url") or "")
            duration = int(s.get("duration_sec") or 0)
            for v in vids:
                v = str(v).strip()
                if not v or v in seen:
                    continue
                seen.add(v)
                items.append(
                    {
                        "media_type": "video",
                        "url": v,
                        "poster": poster,
                        "duration_sec": duration,
                        "product_id": "",
                        "title": f"知识库视频：{doc_name}" if doc_name else "知识库视频",
                    }
                )
    except Exception:
        pass
    return items

# 媒体数量护栏：总量 ≤8，其中视频 ≤2（视频卡片高，多了会把聊天窗口撑得很长）
_MAX_COMBINED_MEDIA = 8
_MAX_VIDEOS = 2


def _combined_media_items(
    product_items: list[dict[str, Any]],
    sources: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """合并商品图与检索命中的知识库媒体（URL 去重，商品图优先，总量 ≤8、视频 ≤2）。"""
    combined: list[dict[str, Any]] = list(product_items or [])
    seen = {str(m.get("url") or "") for m in combined}
    video_count = sum(1 for m in combined if m.get("media_type") == "video")
    for item in _doc_images_from_sources(list(sources or [])):
        url = str(item.get("url") or "")
        if url in seen or len(combined) >= _MAX_COMBINED_MEDIA:
            continue
        if item.get("media_type") == "video" and video_count >= _MAX_VIDEOS:
            continue
        if item.get("media_type") == "video":
            video_count += 1
        seen.add(url)
        combined.append(item)
    return combined[:_MAX_COMBINED_MEDIA]


def _extract_current_product(history: list[dict[str, Any]]) -> str | None:
    """从会话历史最后用户消息提取商品名（enrich_reference 指代补全用）。

    匹配 PG alias_rules 里的标准商品名（长名优先），从最近一条用户消息向前找。

    Args:
        history: 会话历史消息列表。

    Returns:
        商品标准名；找不到返回 None。
    """
    try:
        from agent_base.retrieval.enrich import expand_aliases, load_aliases
        from agent_base.retrieval.retrieval_policy import _load_catalog

        aliases = load_aliases() or {}
        # 商品名必须以 catalog 为准（别名表含"适合/退货"等非商品词，
        # 直接当商品名会误匹配）
        catalog = _load_catalog()
        if not catalog:
            return None
        product_names = sorted(
            {
                str(p.get("name"))
                for p in (catalog.get("products") or {}).values()
                if isinstance(p, dict) and p.get("name")
            },
            key=len,
            reverse=True,
        )
        for m in reversed(history or []):
            if m.get("role") != "user":
                continue
            text = str(m.get("content", "") or "")
            # 先扩展别名（历史消息可能是"玻尿酸精华"这类别名，标准名不在原文）
            enriched = expand_aliases(text, aliases)
            for name in product_names:
                if name in enriched:
                    return name
    except Exception:
        pass
    return None


def _extract_all_product_names(history: list[dict[str, Any]]) -> list[str]:
    """从会话历史提取全部标准商品名（多商品推荐场景的媒体候选，按出现顺序去重）。

    与 _extract_current_product 口径一致（catalog 标准名 + 别名扩展），
    但收集全部命中而非只取最近一条。
    """
    try:
        from agent_base.retrieval.enrich import expand_aliases, load_aliases
        from agent_base.retrieval.retrieval_policy import _load_catalog

        aliases = load_aliases() or {}
        catalog = _load_catalog()
        if not catalog:
            return []
        product_names = sorted(
            {
                str(p.get("name"))
                for p in (catalog.get("products") or {}).values()
                if isinstance(p, dict) and p.get("name")
            },
            key=len,
            reverse=True,
        )
        found: list[str] = []
        for m in history or []:
            if m.get("role") != "user":
                continue
            text = str(m.get("content", "") or "")
            enriched = expand_aliases(text, aliases)
            for name in product_names:
                if name in enriched and name not in found:
                    found.append(name)
        return found
    except Exception:
        return []


@lru_cache(maxsize=1024)
def _pg_doc_name(doc_id: str) -> str:
    """从 PG documents 取文档显示名（doc_id 内容哈希稳定，可安全缓存）。"""
    try:
        from agent_base.storage.pg import _conn

        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT metadata FROM documents "
                "WHERE doc_id=%s AND status<>'deleted' ORDER BY version DESC LIMIT 1",
                (doc_id,),
            )
            row = cur.fetchone()
        if row:
            return str((row[0] or {}).get("doc_name") or "")
    except Exception:
        pass
    return ""


def _resolve_doc_name(doc_id: str, metadata: dict[str, Any]) -> str:
    """解析来源文档显示名（主流 RAG 来源卡规范：文档名做标题，不暴露内部 ID）。

    优先级：chunk metadata.doc_name（新入库自带）→ PG documents metadata.doc_name
    （存量兜底）→ 空串（前端回退到章节名）。

    Args:
        doc_id: 文档 ID（可为空）。
        metadata: chunk 元数据。

    Returns:
        文档显示名；找不到时返回空字符串。
    """
    name = (metadata or {}).get("doc_name")
    if name:
        return str(name)
    if not doc_id:
        return ""
    return _pg_doc_name(doc_id)


def stream_ask(
    request: Any,
    constraints: dict[str, Any],
    runtime: dict[str, Any],
    owner: str = "",
):
    r"""生成 P20 SSE 事件流字符串序列。

    Args:
        request: RagRequest（question/top_k/candidate_k/rerank/session_id）。
        constraints: 商品约束。
        runtime: 运行期对象。
        owner: 会话归属用户（写盘时注入消息 owner）。

    Yields:
        SSE 事件字符串（``data: {json}\\n\\n``）。
    """
    import json

    def emit(event_type: str, **kwargs: Any) -> str:
        payload = json.dumps({"type": event_type, **kwargs}, ensure_ascii=False)
        return f"data: {payload}\n\n"

    llm_config = runtime["llm_config"]
    use_llm = llm_config.get("provider", "none") not in {"none", "off", "false"}

    # 日志 MVP：请求入口
    try:
        from agent_base.monitoring.logger import log_event
        log_event("INFO", "streaming", "request_start", {
            "question": request.question[:200],
            "session_id": getattr(request, "session_id", None),
            "user_id": getattr(request, "user_id", None),
            "framework": getattr(request, "framework", "classic"),
        })
    except Exception:
        pass

    # P21: 情绪路由（规则通道，不调 LLM）——影响话术 + 展示 + 转人工
    try:
        from agent_base.agents.emotion import detect_emotion, emotion_tone_guide

        emotion = detect_emotion(request.question)
        emotion["label_cn"] = {
            "anger": "愤怒",
            "anxiety": "焦虑",
            "positive": "积极",
            "neutral": "中性",
        }.get(emotion.get("label", "neutral"), "中性")
        tone_guide = emotion_tone_guide(emotion.get("label", "neutral"))
    except Exception:
        emotion = {
            "label": "neutral",
            "label_cn": "中性",
            "intensity": 0.0,
            "matched": [],
            "should_handoff": False,
        }
        tone_guide = ""

    # P22a: 工具路由——订单/物流/库存意图 → 查真实 PG，结果注入生成
    # v0.48: 转人工公共前置（与 supervisor 开关无关，经典链路也生效）
    if request.session_id:
        try:
            from agent_base.config import deep_get as _dg
            from agent_base.config import load_yaml as _ly
            from agent_base.storage.pg import handoff_check, handoff_trigger

            # 情绪/关键词命中 → 自动触发转人工（active 会话不打断）
            if emotion.get("should_handoff"):
                handoff_trigger(request.session_id, f"情绪: {emotion.get('label', 'neutral')}")
            _ho_cfg = _ly("configs/app.yaml") or {}
            _ho = handoff_check(
                request.session_id,
                pending_timeout=int(_dg(_ho_cfg, "handoff.pending_timeout_min", 15)) * 60,
                idle_timeout=int(_dg(_ho_cfg, "handoff.idle_timeout_min", 20)) * 60,
            )
        except Exception:
            _ho = None
        if _ho and _ho.get("status") in ("active", "pending"):
            # v0.51: 转人工期间买家消息直接落库进客服通道（AI 停手，客服实时可见）
            try:
                from agent_base.storage.pg import chat_append

                chat_append(request.session_id, "user", request.question, owner=owner)
            except Exception:
                pass
            _ha = (
                "人工客服正在处理您的会话，请稍候…"
                if _ho.get("status") == "active"
                else "您的问题已转接人工客服，正在排队接入，请稍候…"
            )
            yield emit("done", answer=_ha, compliance_warnings=[])
            return

    # BUG-12: 用户消息尽早落库（在任何检索/澄清/早退分支之前），
    # 保证下一轮会话历史立即可用（指代锚定）、客服工作台实时可见
    if request.session_id:
        try:
            from agent_base.storage.chat_memory import append_chat_message

            append_chat_message(request.session_id, "user", request.question, owner=owner)
        except Exception:
            pass

    tool_used: str | None = None
    tool_result = ""
    try:
        import re as _re

        q = request.question
        q_lower = q.lower()
        from agent_base.agents.tools_ecommerce import check_stock, get_logistics, get_order

        order_match = _re.search(r"(ORD[A-Z0-9]{2,})", q, _re.IGNORECASE)
        if order_match:
            oid = order_match.group(1).upper()
            tool_used = "get_order"
            tool_result = get_order.invoke({"order_id": oid})
            logistics = get_logistics.invoke({"order_id": oid})
            if "未找到" not in logistics:
                tool_result += "；" + logistics
        elif any(k in q_lower for k in ["有货", "库存", "断货", "缺货", "补货"]):
            pid_match = _re.search(r"[P]\d{3}", q.upper())
            if pid_match:
                tool_used = "check_stock"
                tool_result = check_stock.invoke({"product_id": pid_match.group(0)})
        elif any(k in q_lower for k in ["订单", "物流", "发货", "快递", "到货", "签收"]):
            tool_used = "order_guidance"
            tool_result = (
                "用户询问订单/物流但未提供订单号，请引导用户提供订单号（如 ORD 开头）后查询。"
            )
    except Exception:
        tool_used, tool_result = None, ""

    # T9：多轮上下文——读会话历史（最近 64 条，注入生成 prompt，检索仍按当前问题）
    history: list[dict[str, Any]] = []
    if request.session_id:
        from agent_base.storage.chat_memory import get_chat_history

        history = get_chat_history(request.session_id, limit=64)
    # T4b: 会话商品名（供 enrich_reference 指代补全）
    current_product = _extract_current_product(history) if history else None
    history_products = _extract_all_product_names(history) if history else []

    # P28 升级：四级降级注入（档1 全量 / 档2 滑动窗口 / 档3 规则压缩 → 0 LLM，
    # 档4 才触发 LLM 摘要）。成本指标随 memory 事件输出，可观测。
    compaction: dict[str, Any] | None = None
    context_metrics: dict[str, Any] = {}
    if request.session_id:
        from agent_base.storage.chat_memory import (
            build_injectable_history,
            compact_chat_history,
            get_chat_history,
            get_context_config,
            get_history_meta,
            record_context_metrics,
        )

        _ctx_cfg = get_context_config()
        injectable_history, context_metrics = build_injectable_history(history, _ctx_cfg)
        if context_metrics.get("needs_compact"):
            compaction = compact_chat_history(request.session_id, llm_config)
            if compaction:
                history = get_chat_history(request.session_id, limit=64)
                injectable_history, context_metrics = build_injectable_history(history, _ctx_cfg)
        else:
            compaction = get_history_meta(request.session_id)
        history = injectable_history
        # P28-2 可观测：记录本次注入成本指标（聚合统计用，Redis 不可用静默跳过）
        record_context_metrics(request.session_id, context_metrics)

    # P19c/T9: 长期记忆画像注入（user_id 有值才读 user_memories，预算 1000 字）
    profile_context = ""
    profile_chars = 0
    if request.user_id:
        try:
            from agent_base.storage.memory import build_profile_context

            profile_context = build_profile_context(request.user_id, intent="", max_chars=1000)
            profile_chars = len(profile_context)
        except Exception:
            profile_context = ""

    # P33a：Supervisor 编排开关（SSE 流式）——开启时走主 Agent 调度
    try:
        from agent_base.config import deep_get, load_yaml

        _sv_cfg = load_yaml("configs/app.yaml") or {}
        _sv_enabled = bool(deep_get(_sv_cfg, "framework.supervisor.enabled", False))
    except Exception:
        _sv_enabled = False
    if _sv_enabled:
        # v0.47: 多 Agent 编排实现可选 langgraph / classic
        # 效率优化：Supervisor 流式缓存——重复问题命中直接秒回
        from agent_base.storage.cache import DATA_VERSION, cache_key, get_cached, set_cache

        _sv_ck = cache_key(
            request.question,
            {"v": DATA_VERSION, "session": request.session_id or "", "fw": "supervisor"},
        )
        _sv_cached = get_cached(_sv_ck)
        if _sv_cached and _sv_cached.get("answer"):
            yield emit("done", answer=_sv_cached["answer"], compliance_warnings=[])
            return

        # T7c：编排在独立线程执行，子 Agent 执行事件实时推送（执行链展示）
        import queue as _queue
        import threading as _threading

        _evt_q: _queue.Queue = _queue.Queue()
        _plan_holder: dict[str, Any] = {}
        _done_evt = _threading.Event()

        def _on_agent(
            agent_name: str,
            status: str,
            duration_ms: int | None = None,
            data: dict[str, Any] | None = None,
        ) -> None:
            _evt_q.put(
                {
                    "type": "agent",
                    "agent": agent_name,
                    "status": status,
                    "duration_ms": duration_ms,
                    **({"data": data} if data else {}),
                }
            )

        def _orchestrate_thread() -> None:
            # v0.49: 全 LangGraph——stream 流式执行（updates 累积状态 + messages 推 LLM token）
            # PostgresSaver 为同步 checkpointer（无 aget_tuple），必须用同步 graph.stream。
            def _stream_graph() -> None:
                from agent_base.agents.graph_supervisor import _get_graph

                # P1: 读取配置（max_steps / thread_id）
                try:
                    from agent_base.config import load_yaml as _ly
                    _oc = (_ly("configs/app.yaml") or {}).get("orchestration") or {}
                    _max_steps = int(_oc.get("max_steps", 30))
                except Exception:
                    _max_steps = 30

                graph = _get_graph()
                initial: dict[str, Any] = {
                    "question": request.question,
                    "session_id": request.session_id,
                    "user_id": request.user_id,
                    "constraints": constraints,
                    "clarify": False,
                    "dispatch": [],
                    "sources": [],
                    "trace": {},
                    "docs": [],
                    "tool_result": "",
                    "evidence": "",
                    "answer": "",
                    "history": [],
                    "profile": "",
                }
                acc: dict[str, Any] = {}
                try:
                    _cfg: dict[str, Any] = {"recursion_limit": _max_steps}
                    # PostgresSaver 必须带 thread_id：有会话用会话 ID（跨轮恢复），
                    # 无会话生成匿名线程 ID（checkpoint 不跨轮恢复）。
                    _thread_id = request.session_id or f"anon-{uuid.uuid4().hex[:12]}"
                    _cfg["configurable"] = {"thread_id": _thread_id}
                    for mode, data in graph.stream(
                        initial,
                        stream_mode=["updates", "messages"],
                        config=_cfg,
                        context={
                            "on_agent": _on_agent,
                            "runtime": runtime,
                            "rerank_cfg": runtime.get("rerank_config") or {},
                            "intent_classifier_cfg": runtime.get("intent_classifier_config"),
                            "llm_cfg": runtime.get("llm_config") or {},
                        },
                    ):
                        if mode == "messages":
                            chunk = data[0] if isinstance(data, (list, tuple)) else None
                            if chunk is None:
                                continue
                            content = getattr(chunk, "content", "") or ""
                            if content:
                                _evt_q.put({"type": "delta", "delta": content})
                            extra = getattr(chunk, "additional_kwargs", {}) or {}
                            reason = extra.get("reasoning_content") or extra.get("reasoning")
                            if reason:
                                _evt_q.put({"type": "thinking", "delta": str(reason)})
                        elif mode == "updates":
                            for _node, upd in (data or {}).items():
                                if not isinstance(upd, dict):
                                    continue
                                if upd.get("dispatch"):
                                    acc["dispatch"] = acc.get("dispatch", []) + list(upd["dispatch"])
                                for _k, _v in upd.items():
                                    if _k != "dispatch":
                                        acc[_k] = _v
                except Exception as exc:  # noqa: BLE001
                    _plan_holder["error"] = str(exc)
                _plan_holder["state"] = acc

            try:
                _stream_graph()
            except Exception as exc:  # noqa: BLE001
                _plan_holder["error"] = str(exc)
            finally:
                _done_evt.set()

        _threading.Thread(target=_orchestrate_thread, daemon=True).start()
        # 编排进行中：实时转发子 Agent 事件（mainstream Agent 执行链体验）
        while not _done_evt.is_set():
            try:
                evt = _evt_q.get(timeout=0.2)
                yield emit(evt["type"], **{k: v for k, v in evt.items() if k != "type"})
            except _queue.Empty:
                continue
        while True:
            try:
                evt = _evt_q.get_nowait()
                yield emit(evt["type"], **{k: v for k, v in evt.items() if k != "type"})
            except _queue.Empty:
                break
        _done_evt.wait()
        if "error" in _plan_holder:
            # 生产安全：异常详情只进日志，不回显给买家
            try:
                from agent_base.monitoring.logger import log_event

                log_event("ERROR", "streaming", "orchestration_failed", {
                    "error": str(_plan_holder["error"])[:300],
                })
            except Exception:
                pass
            yield emit("done", answer="抱歉，服务暂时开小差了，请稍后再试或联系人工客服。", compliance_warnings=[])
            return
        plan = _plan_holder.get("state") or {}
        plan.setdefault("sources", [])
        plan.setdefault("dispatch", [])
        plan.setdefault("trace", {})
        plan.setdefault("history", [])
        plan.setdefault("profile", "")
        plan.setdefault("evidence", "")
        plan.setdefault("tool_result", "")
        plan.setdefault("clarify", False)
        plan.setdefault("clarification", "")
        plan.setdefault("mode", "supervisor")
        plan.setdefault("intent", "general_qa")
        plan.setdefault("answer", "")

        sources = [
            {
                "doc_name": s.get("doc_name", ""),
                "section": s.get("section", ""),
                "source_file": "",
                "score": s.get("score"),
                "preview": str(s.get("content", ""))[:240],
                "image_urls": list(s.get("image_urls") or []),
                "video_urls": list(s.get("video_urls") or []),
                "poster_url": str(s.get("poster_url") or ""),
                "duration_sec": int(s.get("duration_sec") or 0),
                "media_id": s.get("media_id") or None,
            }
            for s in (plan["sources"] or [])[:6]
        ]
        yield emit("sources", sources=sources)
        # 媒体按需门控：仅用户明确要看图/视频时才下发
        if _media_requested(
            request.question,
            str(((plan["trace"] or {}).get("route") or {}).get("sub_intent") or ""),
        ):
            media_items = _media_for_stream(
                constraints, current_product, request.question, history_products
            )
            _combined_media = _combined_media_items(media_items, sources)
            if _combined_media:
                yield emit("media", media=_combined_media)
        yield emit(
            "trace",
            trace={
                "supervisor": {"enabled": True, "dispatch": plan["dispatch"], "mode": plan.get("mode", "supervisor")},
                "decision": (plan["trace"] or {}).get("decision", {}),
                "route": (plan["trace"] or {}).get("route", {}),
                "rewrite": (plan["trace"] or {}).get("rewrite", {}),
                "search_query": (plan["trace"] or {}).get("search_query", ""),
                "metadata_filter": (plan["trace"] or {}).get("metadata_filter", {}),
                "stage_counts": (plan["trace"] or {}).get("stage_counts", {}),
                "emotion": emotion,
                "tool": {"used": bool(plan["tool_result"]), "result": plan["tool_result"]} if plan["tool_result"] else None,
            },
        )
        # memory 事件（复用经典结构 + P28-2 成本指标展示）
        from agent_base.storage.chat_memory import (
            CONTEXT_BUDGET_CHARS,
            build_injectable_history,
            get_context_config,
        )

        _sv_history = plan["history"] or []
        _sv_chars = sum(len(m.get("content", "")) for m in _sv_history)
        _sv_injectable, _sv_ctx = build_injectable_history(_sv_history, get_context_config())
        _sv_cfg = get_context_config()
        _sv_threshold = int(float(_sv_cfg.get("compact_trigger_ratio", 0.8)) * 100)
        yield emit(
            "memory",
            memory={
                "session_id": request.session_id,
                "user_id": request.user_id,
                "storage": "redis" if (request.session_id or "").startswith("test-") else "pg",
                "history_count": _sv_ctx.get("raw_msgs", len(_sv_history)),
                "history_chars": _sv_ctx.get("raw_chars", _sv_chars),
                "history_budget": int(get_context_config().get("budget_chars", CONTEXT_BUDGET_CHARS)),
                "history_ratio": _sv_ctx.get("ratio", round(_sv_chars / CONTEXT_BUDGET_CHARS, 3)),
                "context_threshold": _sv_threshold,
                "inject_chars": _sv_ctx.get("inject_chars", _sv_chars),
                "context_tier": _sv_ctx.get("tier", 0),
                "context_saved_chars": _sv_ctx.get("saved_chars", 0),
                "context_saved_pct": _sv_ctx.get("saved_pct", 0.0),
                "context_est_tokens": _sv_ctx.get("est_tokens", 0),
                "context_summary_reused": _sv_ctx.get("summary_reused", False),
                "profile_chars": len(plan["profile"]),
                "profile_budget": 1000,
                "profile_snippet": plan["profile"][:200],
                "compaction": {"compacted": False, "rounds": 0},
            },
        )
        if plan["clarify"]:
            yield emit("done", answer=plan["clarification"], compliance_warnings=[])
            return
        if plan.get("handoff") or plan.get("mode") == "handoff":
            _ho = plan.get("handoff") or {}
            _ha = (
                "人工客服正在处理您的会话，请稍候…"
                if _ho.get("status") == "active"
                else "您的问题已转接人工客服，正在排队接入，请稍候…"
            )
            yield emit("done", answer=_ha, compliance_warnings=[])
            return
        # v0.49: answer 已由图内 generate 节点生成并流式推送（astream messages），这里直接收尾
        _sv_answer = plan.get("answer") or ""
        # 客服端「AI 召回依据」：supervisor 路径同样落库 assistant + sources（含完整 chunk 原文）
        if request.session_id and _sv_answer:
            try:
                from agent_base.storage.chat_memory import append_chat_message

                _sv_store_sources = [
                    {
                        "rank": i + 1,
                        "doc_id": str(s.get("doc_id", "") or ""),
                        "doc_name": str(s.get("doc_name", "") or s.get("section", "") or "未知文档"),
                        "section": str(s.get("section", "") or ""),
                        "source_file": str(s.get("source_file", "") or ""),
                        "score": s.get("score"),
                        "preview": str(s.get("content", "") or ""),
                    }
                    for i, s in enumerate((plan["sources"] or [])[:6])
                ]
                append_chat_message(
                    request.session_id,
                    "assistant",
                    _sv_answer,
                    owner=owner,
                    sources=_sv_store_sources,
                )
            except Exception:
                pass
        try:
            set_cache(_sv_ck, {"answer": _sv_answer, "safety": {}, "trace": {"results": []}})
        except Exception:
            pass
        # P13-05：supervisor 链路同样对话后异步提炼长期记忆
        try:
            if request.session_id and request.user_id:
                from agent_base.agents.tools_memory import maybe_async_extract

                import threading as _th

                def _run_sv_extract() -> None:
                    try:
                        maybe_async_extract(
                            request.session_id,
                            request.user_id,
                            model_cfg=llm_config,
                            history_count=len(_sv_history),
                        )
                    except Exception:
                        pass

                _th.Thread(target=_run_sv_extract, daemon=True).start()
        except Exception:
            pass
        yield emit("done", answer=_sv_answer, compliance_warnings=[])
        return

    # BUG-5：闲聊直答——跳过检索（经典链路，supervisor.enabled=false 时生效）
    from agent_base.agents.emotion import looks_like_chat

    if looks_like_chat(request.question):
        yield emit("sources", sources=[])
        yield emit(
            "trace",
            trace={
                "supervisor": {"enabled": False},
                "decision": {},
                "route": {},
                "emotion": emotion,
                "mode": "chat_direct",
            },
        )
        yield emit("stage", stage="generating", message="小满正在回应你...")
        from agent_base.agents.supervisor import compute_generation_temperature

        _chat_temp = compute_generation_temperature(
            intent="general_qa", emotion=emotion.get("label", "neutral"),
        )
        yield from _stream_llm_generation(
            request.question,
            "",
            "",
            [],
            "",
            llm_config,
            tone_guide,
            emit,
            request.session_id,
            owner=owner,
            temperature=_chat_temp,
            sources=[],
        )
        return

    # 1. 检索（不生成，只取证据与 trace）
    cfg = RetrievalConfig.from_runtime(runtime)
    cfg.llm.use_llm = False
    cfg.top_k = request.top_k
    cfg.candidate_k = request.candidate_k
    cfg.rerank = request.rerank
    cfg.product_name = constraints["product_name"]
    cfg.product_spec = constraints["product_spec"]
    cfg.category = constraints["category"]
    # 会话级意图 v2：读取用户画像（肤质/价位/尺码等），供意图层消解缺失需求
    profile: dict[str, Any] = {}
    try:
        if request.user_id:
            from agent_base.storage.memory import retrieve_memory

            for m in retrieve_memory(request.user_id):
                key = str(m.get("key") or "")
                value = m.get("value")
                if key and value not in (None, "", [], {}):
                    profile[key] = value
    except Exception:
        profile = {}
    cfg.profile = profile
    result = answer_question_with_trace(
        request.question,
        runtime["vector_store"],
        cfg,
        summary_store=runtime["summary_store"],
        sparse_store=runtime["sparse_store"],
        current_product=current_product,
    )
    payload = result.to_dict()
    trace = payload.get("trace", {})
    docs = list(result.trace.docs) if getattr(result.trace, "docs", None) else []

    # 会话级销售决策（v2）：读取阶段 → 规则决策 → 持久化 → 组装策略/话术
    sales_ctx: dict[str, Any] | None = None
    try:
        from agent_base.agents.sales_stage import build_sales_context

        route_dict = dict(trace.get("route") or {})
        route_dict.setdefault("emotion", emotion.get("label", "neutral"))
        sales_ctx = build_sales_context(
            route_dict,
            request.session_id,
            request.question,
            history=history,
            profile=profile,
        )
    except Exception:
        sales_ctx = None

    # sources 事件：文档名做主标题、章节做副标题、分数转前端百分比展示；
    # doc_id/chunk_id 仅作追踪锚点，前端不展示（主流 RAG 引用卡规范）。
    # Phase 3：图片知识库命中时带 image_urls（供前端引用卡/媒体区展示）。
    sources = []
    for i, d in enumerate(docs[:6]):
        meta = getattr(d, "metadata", None) or {}
        doc_id = str(meta.get("doc_id", "") or "")
        doc_name = _resolve_doc_name(doc_id, meta)
        sources.append(
            {
                "rank": i + 1,
                "doc_id": doc_id,
                "chunk_id": str(meta.get("chunk_id", "") or ""),
                "doc_name": doc_name,
                "section": meta.get("section", ""),
                "source_file": meta.get("source_file", ""),
                "score": (meta.get("rerank_score") or meta.get("vector_score")),
                "preview": (getattr(d, "page_content", "") or ""),
                "image_urls": list(meta.get("image_urls") or []),
                "video_urls": list(meta.get("video_urls") or []),
                "poster_url": str(meta.get("poster_url") or ""),
                "duration_sec": int(meta.get("duration_sec") or 0),
                "media_id": meta.get("media_id") or None,
            }
        )
    yield emit("sources", sources=sources)
    # 媒体按需门控：仅用户明确要看图/视频时才下发（商品图 + 命中的知识库媒体）
    if _media_requested(
        request.question,
        str((trace.get("route") or {}).get("sub_intent") or ""),
    ):
        media_items = _media_for_stream(
            constraints, current_product, request.question, history_products
        )
        _combined_media = _combined_media_items(media_items, sources)
        if _combined_media:
            yield emit("media", media=_combined_media)

    # trace 事件（轻量：route/decision/stage_counts）
    yield emit(
        "trace",
        trace={
            "route": trace.get("route", {}),
            "decision": trace.get("decision", {}),
            "rewrite": trace.get("rewrite", {}),
            "search_query": trace.get("search_query", ""),
            "metadata_filter": trace.get("metadata_filter", {}),
            "mode": trace.get("mode", "auto"),
            "rerank": trace.get("rerank", ""),
            "stage_counts": trace.get("stage_counts", {}),
            "fallback_used": trace.get("fallback_used", False),
            "enhancement": trace.get("enhancement", {}),
            "emotion": emotion,
            "tool": {"used": tool_used, "result": tool_result} if tool_used else None,
            "sales": sales_ctx,
        },
    )

    # P19c: trace.memory 记录上下文管理状态（前端记忆测试面板展示）
    # P28 升级：注入前 raw 用量 + 四级降级指标（tier/saved/est_tokens）一并展示
    history_chars = sum(len(m.get("content", "")) for m in history)
    from agent_base.storage.chat_memory import CONTEXT_BUDGET_CHARS, get_context_config

    _ctx_cfg = get_context_config()
    _raw_chars = context_metrics.get("raw_chars", history_chars)
    _budget = int(_ctx_cfg.get("budget_chars", CONTEXT_BUDGET_CHARS))
    _raw_ratio = round(_raw_chars / _budget, 3)
    _threshold = int(float(_ctx_cfg.get("compact_trigger_ratio", 0.8)) * 100)

    yield emit(
        "memory",
        memory={
            "session_id": request.session_id,
            "user_id": request.user_id,
            "storage": "redis" if (request.session_id or "").startswith("test-") else "pg",
            "history_count": context_metrics.get("raw_msgs", len(history)),
            "history_chars": _raw_chars,
            "history_budget": _budget,
            "history_ratio": _raw_ratio,
            "context_threshold": _threshold,
            "inject_chars": history_chars,
            "context_tier": context_metrics.get("tier", 0),
            "context_saved_chars": context_metrics.get("saved_chars", 0),
            "context_saved_pct": context_metrics.get("saved_pct", 0.0),
            "context_est_tokens": context_metrics.get("est_tokens", 0),
            "context_summary_reused": context_metrics.get("summary_reused", False),
            "profile_chars": profile_chars,
            "profile_budget": 1000,
            "profile_snippet": (profile_context[:200] if profile_context else ""),
            "compaction": {
                "compacted": bool(compaction and compaction.get("rounds")),
                "rounds": int((compaction or {}).get("rounds", 0)),
                "last_compacted_at": (compaction or {}).get("last_compacted_at"),
                "before_chars": int((compaction or {}).get("before_chars", 0)),
                "after_chars": int((compaction or {}).get("after_chars", 0)),
                "history": (compaction or {}).get("history", []),
            },
        },
    )

    if not use_llm or not docs:
        yield emit("done", answer=payload.get("answer", ""), compliance_warnings=[])
        return

    # 2. 流式生成（LCEL 官方链：ChatPromptTemplate | ChatDeepSeek，透传思考过程）
    evidence_chars = int(llm_config.get("evidence", {}).get("max_chars_per_doc", 1200))
    ctx = "\n\n".join(
        f"[{getattr(d, 'metadata', {}).get('section', '商品')}] {getattr(d, 'page_content', '')[:evidence_chars]}"
        for d in docs[:5]
    )
    # 动态温度（P0-2）：意图基底 + 情绪调节（emotion 已在上文规则通道检测）
    try:
        from agent_base.agents.supervisor import compute_generation_temperature

        _decision = (trace.get("decision") or {}) if isinstance(trace, dict) else {}
        _generation_temperature = compute_generation_temperature(
            intent=str(_decision.get("intent") or "general_qa"),
            emotion=str(emotion.get("label", "neutral")),
        )
    except Exception:
        _generation_temperature = 0.1
    yield from _stream_llm_generation(
        request.question,
        ctx,
        tool_result,
        history,
        profile_context,
        llm_config,
        tone_guide,
        emit,
        request.session_id,
        owner=owner,
        temperature=_generation_temperature,
        sources=sources,
        sales_strategy=(sales_ctx or {}).get("sales_strategy") or "",
        stage_guide=(sales_ctx or {}).get("guide") or "",
    )

    # P13-05：对话完成后异步提炼长期记忆（写门控 + 信任层级，不阻塞响应）
    try:
        if request.session_id and request.user_id:
            from agent_base.agents.tools_memory import maybe_async_extract

            # 后台线程执行（LLM 提炼 flash，几秒级），避免阻塞 SSE 收尾
            import threading as _th

            def _run_extract() -> None:
                try:
                    maybe_async_extract(
                        request.session_id,
                        request.user_id,
                        model_cfg=llm_config,
                        history_count=len(history),
                    )
                except Exception:
                    pass

            _th.Thread(target=_run_extract, daemon=True).start()
    except Exception:
        pass


def _stream_llm_generation(
    question: str,
    evidence_text: str,
    tool_result: str,
    history: list[dict[str, Any]],
    profile_context: str,
    llm_config: dict[str, Any],
    tone_guide: str,
    emit,
    session_id: str | None = None,
    owner: str = "",
    on_complete=None,
    temperature: float = 0.1,
    sources: list[dict[str, Any]] | None = None,
    sales_strategy: str = "",
    stage_guide: str = "",
):
    """流式生成事件序列（thinking/delta/done）——经典与 Supervisor 共用。

    Args:
        question: 用户问题。
        evidence_text: 检索证据文本（注入 prompt 供生成引用）。
        tool_result: 工具查询结果（权威数据）。
        history: 会话历史消息列表。
        profile_context: 用户画像文本。
        llm_config: LLM 配置（model/base_url/api_key_env 等）。
        tone_guide: 情绪话术指导（情绪路由产出）。
        emit: SSE 事件发射回调。
        session_id: 会话 ID（落库用）。
        owner: 消息归属用户。
        on_complete: 生成完成回调（可选）。
        temperature: 生成温度（动态温度：意图基底 + 情绪调节）。
        sources: 当轮检索召回依据（assistant 落库用，客服端 AI 召回依据展示）。
        sales_strategy: 导购策略提示词块（有购买信号/异议时注入系统提示词）。
        stage_guide: 会话级导购阶段话术指导（按阶段/动作注入）。
    """
    # 官方路线：langchain-deepseek 的 ChatDeepSeek（能解析 thinking 流，
    # 通用 ChatOpenAI 会丢弃 reasoning_content —— langchain-ai/langchain#35059）
    from langchain_deepseek import ChatDeepSeek

    api_key = os.getenv(llm_config.get("api_key_env", "ANTHROPIC_AUTH_TOKEN")) or "missing-key"
    model = ChatDeepSeek(
        model=llm_config.get("model") or "deepseek-reasoner",
        base_url=llm_config.get("base_url") or "https://api.deepseek.com",
        api_key=api_key,
        temperature=temperature,
        streaming=True,
    )
    history_block = ""
    if history:
        lines = []
        # T9：注入最近 32 条 × 1000 字，充分使用 256K 上下文
        for m in history[-32:]:
            role = "用户" if m.get("role") == "user" else "助手"
            lines.append(f"{role}：{str(m.get('content', ''))[:1000]}")
        history_block = "对话历史（仅供上下文参考，回答以商品资料为准）：\n" + "\n".join(lines) + "\n\n"
    profile_block = (profile_context + "\n\n") if profile_context else ""
    from agent_base.prompts import get_prompt

    system_prompt = get_prompt("qa", "system")
    if stage_guide:
        system_prompt = system_prompt + "\n\n" + stage_guide
    if sales_strategy:
        system_prompt = system_prompt + "\n\n" + sales_strategy
    prompt = (
        (tone_guide + "\n\n" if tone_guide else "")
        + history_block
        + profile_block
        + f"用户问题：{question}\n\n"
        + (f"系统查询结果（权威数据，直接采用）：{tool_result}\n\n" if tool_result else "")
        + f"商品/FAQ 资料：\n{evidence_text}\n\n"
        + "请像小满一样回答：用简洁的标准 Markdown 组织（小标题用 ##、要点用 -、"
        "对比/方案可用表格），结论先行，每块不超过 3 行，不写客套废话；"
        "资料缺失时只写一句“资料未标注该数值”。"
    )
    # LCEL 官方链：system 固定 + 用户消息模板 → ChatDeepSeek 流式
    from langchain_core.prompts import ChatPromptTemplate

    chain = (
        ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", "{user_message}")]
        )
        | model
    )
    stream = chain.stream({"user_message": prompt})
    full_parts: list[str] = []
    answer = ""
    error: str | None = None
    try:
        for chunk in stream:
            reasoning = (chunk.additional_kwargs or {}).get("reasoning_content")
            content = getattr(chunk, "content", "") or ""
            if reasoning:
                yield emit("thinking", delta=reasoning)
            if content:
                full_parts.append(content)
                yield emit("delta", delta=content)
        answer = "".join(full_parts)
    except Exception as exc:
        answer = "".join(full_parts)
        error = str(exc)
    # BUG-18: 先落库 assistant 再发 done——客户端收到 done 后立即拉历史
    # 能读到最新快照（user 消息已在流式开始前落库）
    if session_id and answer:
        from agent_base.storage.chat_memory import append_chat_message

        try:
            append_chat_message(session_id, "assistant", answer, owner=owner, sources=sources or [])
        except Exception:
            pass
    if on_complete is not None:
        try:
            on_complete(answer)
        except Exception:
            pass
    yield emit("done", answer=answer, compliance_warnings=[], **( {"error": error} if error else {} ))
