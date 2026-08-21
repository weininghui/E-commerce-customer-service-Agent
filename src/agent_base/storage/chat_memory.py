"""对话短期记忆（对话测试专用，P19c）。

设计：
- 对话测试（session_id 以 ``test-`` 开头）走 **Redis**：key ``chat:memory:{sid}``
  存最近消息 JSON，TTL 默认 7 天、每次对话自动续期——长时间不对话自动清空，
  不落 PG、不长期保留；
- 用户端会话（其他 session_id）保持原 PG ``chat_messages`` 路径不变；
- Redis 不可用时静默降级（返回空历史 / 跳过写入），不阻塞对话链路。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import redis


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
# 测试会话记忆 TTL：7 天（秒），每次对话刷新；可通过环境变量覆盖
CHAT_MEMORY_TTL = int(os.getenv("CHAT_MEMORY_TTL", str(7 * 24 * 3600)))
TEST_SESSION_PREFIX = "test-"
_KEY_PREFIX = "chat:memory:"
_META_PREFIX = "chat:meta:"
_CTX_STATS_KEY = "ctx:cost:stats"
_CTX_STATS_MAX = 500  # 聚合统计保留最近 500 条注入记录（防 Redis 无限膨胀）

# 上下文管理（P28）：历史注入字符预算，超阈值 80% 时自动 LLM 压缩
# T9：适配 DeepSeek V4 256K 上下文——历史预算 6 万字符（约占窗口 25-40%），
# 压缩在 80% 即 4.8 万字符时触发，长会话保留最近 8 条原文
CONTEXT_BUDGET_CHARS = 60000
COMPACT_TRIGGER_RATIO = 0.8
COMPACT_KEEP_RECENT = 8
COMPACT_SUMMARY_PROMPT = (
    "你是电商客服的对话摘要器。把以下用户与助手的对话压缩成一段 ≤400 字的中文摘要，"
    "保留：用户身份与偏好、咨询过的问题与结论、未解决的诉求、订单/商品提及。"
    "不编造新信息。只输出摘要文本。"
)

_client: redis.Redis | None = None


def _get_client() -> redis.Redis | None:
    """懒加载 Redis 客户端（连接失败可容忍）。"""
    global _client
    if _client is None:
        try:
            _client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,
            )
            _client.ping()
        except Exception:
            _client = None
    return _client


def is_test_session(session_id: str | None) -> bool:
    """是否为对话测试会话（走 Redis 短期记忆）。"""
    return bool(session_id and session_id.startswith(TEST_SESSION_PREFIX))


# ── 上下文成本控制（P28 升级：四级降级 + 可配置 + 可观测）──────────────────


def get_context_config() -> dict[str, Any]:
    """读取 configs/app.yaml 的 context 段；缺失/异常回退模块默认值。

    Returns:
        dict：budget_chars / tier 阈值 / 窗口参数 / 摘要参数 / 压缩模型档位。
    """
    defaults: dict[str, Any] = {
        "budget_chars": CONTEXT_BUDGET_CHARS,
        "tier_window_ratio": 0.25,
        "tier_rule_ratio": 0.50,
        "compact_trigger_ratio": COMPACT_TRIGGER_RATIO,
        "compact_keep_recent": COMPACT_KEEP_RECENT,
        "summary_max_chars": 400,
        "summarize_trim_chars": 8000,
        "window_recent_msgs": 16,
        "window_msg_chars": 800,
        "rule_first_msgs": 1,
        "rule_recent_msgs": 12,
        "llm_compact_model": "flash",
    }
    try:
        from agent_base.config import load_yaml

        ctx = (load_yaml("configs/app.yaml") or {}).get("context", {}) or {}
        for key in defaults:
            if key in ctx and ctx[key] is not None:
                defaults[key] = ctx[key]
    except Exception:
        pass
    return defaults


def _tier3_rule_compress(
    history: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    """档 3 规则压缩（0 LLM）：摘要复用 + 首条锚点 + 最近 N 条原文。

    - 若历史中已有「【对话摘要】」system 消息则直接复用（摘要幂等，不重复调 LLM）；
    - 保留首条锚点（用户身份/最初诉求）+ 最近 N 条原文；
    - 每条做 1000 字符安全上限（与历史注入口径一致）。

    Returns:
        (压缩后的注入历史, 是否复用了已有摘要)。
    """
    first_n = int(cfg.get("rule_first_msgs", 1))
    recent_n = int(cfg.get("rule_recent_msgs", 12))
    msg_cap = 1000

    summary_msgs = [
        m for m in history
        if str(m.get("role")) == "system" and "【对话摘要】" in str(m.get("content", ""))
    ]
    non_summary = [m for m in history if m not in summary_msgs]
    if not non_summary:
        return [], bool(summary_msgs)

    head = non_summary[:first_n]
    body = non_summary[first_n:]
    recent = body[-recent_n:] if len(body) > recent_n else body

    merged = [
        {**m, "content": str(m.get("content", ""))[:msg_cap]}
        for m in [*summary_msgs, *head, *recent]
    ]
    return merged, bool(summary_msgs)


def build_injectable_history(
    history: list[dict[str, Any]] | None,
    cfg: dict[str, Any] | None = None,
    max_msgs: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """四级降级选择注入历史，返回 (注入历史, 成本指标)。

    档位按「历史字符用量 / 预算」划分（0 LLM 手段优先，LLM 摘要兜底）：
      档 1 (<25%)  全量注入（质量优先，每条 1000 字安全上限）
      档 2 (<50%)  滑动窗口：最近 N 条 × 每条窗口上限
      档 3 (<80%)  规则压缩：摘要复用 + 首条锚点 + 最近 N 条（0 LLM）
      档 4 (>=80%) 规则压缩 + needs_compact=True（由调用方决定是否触发 LLM 摘要）

    Args:
        history: 会话历史消息列表（可能为空）。
        cfg: context 配置；None 时读 configs/app.yaml。
        max_msgs: 注入消息条数上限（默认不限制，档 1 全量）。

    Returns:
        (injectable, metrics)：
        metrics 含 tier/ratio/raw_chars/raw_msgs/inject_chars/inject_msgs/
        saved_chars/saved_pct/est_tokens/needs_compact/summary_reused。
    """
    cfg = cfg or get_context_config()
    budget = int(cfg.get("budget_chars", CONTEXT_BUDGET_CHARS)) or 1
    history = [m for m in (history or []) if isinstance(m, dict)]
    if not history:
        return [], {
            "tier": 0, "ratio": 0.0, "raw_chars": 0, "raw_msgs": 0,
            "inject_chars": 0, "inject_msgs": 0, "saved_chars": 0,
            "saved_pct": 0.0, "est_tokens": 0, "needs_compact": False,
            "summary_reused": False,
        }

    raw_chars = sum(len(str(m.get("content", ""))) for m in history)
    ratio = raw_chars / budget
    t_window = float(cfg.get("tier_window_ratio", 0.25))
    t_rule = float(cfg.get("tier_rule_ratio", 0.50))
    t_compact = float(cfg.get("compact_trigger_ratio", COMPACT_TRIGGER_RATIO))

    if max_msgs:
        history = history[-max_msgs:]
        raw_chars = sum(len(str(m.get("content", ""))) for m in history)
        ratio = raw_chars / budget

    if ratio < t_window:
        tier = 1
        injectable = [
            {**m, "content": str(m.get("content", ""))[:1000]}
            for m in history
        ]
        summary_reused = False
        needs_compact = False
    elif ratio < t_rule:
        tier = 2
        window_n = int(cfg.get("window_recent_msgs", 16))
        cap = int(cfg.get("window_msg_chars", 800))
        injectable = [
            {**m, "content": str(m.get("content", ""))[:cap]}
            for m in history[-window_n:]
        ]
        summary_reused = False
        needs_compact = False
    else:
        injectable, summary_reused = _tier3_rule_compress(history, cfg)
        needs_compact = ratio >= t_compact
        tier = 4 if needs_compact else 3

    inject_chars = sum(len(str(m.get("content", ""))) for m in injectable)
    saved_chars = max(0, raw_chars - inject_chars)
    metrics: dict[str, Any] = {
        "tier": tier,
        "ratio": round(ratio, 3),
        "raw_chars": raw_chars,
        "raw_msgs": len(history),
        "inject_chars": inject_chars,
        "inject_msgs": len(injectable),
        "saved_chars": saved_chars,
        "saved_pct": round(saved_chars / raw_chars, 3) if raw_chars else 0.0,
        "est_tokens": inject_chars // 2,
        "needs_compact": needs_compact,
        "summary_reused": summary_reused,
    }
    return injectable, metrics


def _key(session_id: str) -> str:
    return f"{_KEY_PREFIX}{session_id}"


def _redis_history(session_id: str, limit: int) -> list[dict[str, Any]]:
    client = _get_client()
    if client is None:
        return []
    try:
        raw = client.get(_key(session_id))
        if not raw:
            return []
        messages = json.loads(raw)
        if not isinstance(messages, list):
            return []
        return [m for m in messages[-limit:] if isinstance(m, dict)]
    except Exception:
        return []


def _redis_append(session_id: str, role: str, content: str, limit: int = 16) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        key = _key(session_id)
        raw = client.get(key)
        messages = json.loads(raw) if raw else []
        if not isinstance(messages, list):
            messages = []
        messages.append({"role": role, "content": content})
        messages = messages[-limit:]
        client.set(key, json.dumps(messages, ensure_ascii=False), ex=CHAT_MEMORY_TTL)
    except Exception:
        pass


def _pg_history(session_id: str, limit: int) -> list[dict[str, Any]]:
    try:
        from agent_base.storage.pg import chat_history

        return chat_history(session_id, limit=limit)
    except Exception:
        return []


def _pg_append(
    session_id: str,
    role: str,
    content: str,
    owner: str = "",
    sources: list[dict[str, Any]] | None = None,
) -> None:
    try:
        from agent_base.storage.pg import chat_append

        chat_append(session_id, role, content, owner=owner, sources=sources)
    except Exception:
        pass


def get_chat_history(session_id: str, limit: int = 8) -> list[dict[str, Any]]:
    """读会话历史：测试会话走 Redis（TTL 过期），其余走 PG。

    非 test- 会话若存在 Redis 上下文摘要（``chat:meta:{sid}``，PG 会话
    压缩产物），以 system 消息置顶合并返回——档 3 规则压缩可复用摘要（0 LLM）。
    """
    if is_test_session(session_id):
        return _redis_history(session_id, limit=limit)
    msgs = _pg_history(session_id, limit=limit)
    meta = get_history_meta(session_id)
    summary = str((meta or {}).get("summary") or "").strip()
    if summary and not any(str(m.get("role")) == "system" for m in msgs):
        return [{"role": "system", "content": f"【对话摘要】{summary}"}, *msgs]
    return msgs


def append_chat_message(
    session_id: str,
    role: str,
    content: str,
    owner: str = "",
    sources: list[dict[str, Any]] | None = None,
) -> None:
    """追加会话消息：测试会话写 Redis（TTL 续期），其余写 PG（SEC-2 绑定归属）。

    Args:
        session_id: 会话 ID。
        role: 消息角色（user/assistant/agent）。
        content: 消息内容。
        owner: 会话归属账号。
        sources: assistant 消息的检索召回依据（JSONB，客服端追溯用）。
    """
    if is_test_session(session_id):
        _redis_append(session_id, role, content)
    else:
        _pg_append(session_id, role, content, owner=owner, sources=sources)


def clear_chat_memory(session_id: str) -> bool:
    """清空会话短期记忆（Redis ``chat:memory`` + ``chat:meta``）。

    test- 与非 test- 会话都处理；Redis 不可用时返回 False 不阻塞。
    """
    client = _get_client()
    if client is None:
        return False
    try:
        return bool(client.delete(_key(session_id), f"{_META_PREFIX}{session_id}"))
    except Exception:
        return False


# ── 上下文压缩（P28）──────────────────────────────────────────────────────


def get_history_meta(session_id: str) -> dict[str, Any]:
    """读会话上下文压缩元数据（Redis；不可用返回默认值）。

    Args:
        session_id: 会话 ID。

    Returns:
        ``{rounds, last_compacted_at, before_chars, after_chars, history, summary}``。
    """
    default = {
        "rounds": 0,
        "last_compacted_at": None,
        "before_chars": 0,
        "after_chars": 0,
        "history": [],
        "summary": "",
    }
    client = _get_client()
    if client is None:
        return default
    try:
        raw = client.get(f"{_META_PREFIX}{session_id}")
        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            return default
        merged = {**default, **data}
        merged["history"] = list(merged.get("history") or [])[-5:]
        return merged
    except Exception:
        return default


def _save_history_meta(session_id: str, meta: dict[str, Any]) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.set(
            f"{_META_PREFIX}{session_id}",
            json.dumps(meta, ensure_ascii=False),
            ex=CHAT_MEMORY_TTL,
        )
    except Exception:
        pass


def record_context_metrics(
    session_id: str | None,
    metrics: dict[str, Any] | None,
) -> None:
    """记录一次注入的成本指标到聚合统计（Redis List，保留最近 N 条）。

    P28-2 可观测：每次对话注入后调用，供 ``get_context_cost_stats``
    聚合「档位分布 / 平均节省 / 估算 token / 摘要复用次数」。
    Redis 不可用时静默跳过，不阻塞主链路。
    """
    if not session_id or not metrics or not metrics.get("raw_msgs"):
        return
    client = _get_client()
    if client is None:
        return
    try:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "tier": metrics.get("tier", 0),
            "raw_chars": metrics.get("raw_chars", 0),
            "inject_chars": metrics.get("inject_chars", 0),
            "saved_chars": metrics.get("saved_chars", 0),
            "saved_pct": metrics.get("saved_pct", 0.0),
            "est_tokens": metrics.get("est_tokens", 0),
            "summary_reused": bool(metrics.get("summary_reused")),
            "needs_compact": bool(metrics.get("needs_compact")),
        }
        client.lpush(_CTX_STATS_KEY, json.dumps(entry, ensure_ascii=False))
        client.ltrim(_CTX_STATS_KEY, 0, _CTX_STATS_MAX - 1)
        client.expire(_CTX_STATS_KEY, CHAT_MEMORY_TTL)
    except Exception:
        pass


def get_context_cost_stats(limit: int = 200) -> dict[str, Any]:
    """聚合上下文成本指标（档位分布 / 节省 / token / 摘要复用）。

    Args:
        limit: 参与统计的最近注入记录条数。

    Returns:
        dict：{total, tier_distribution, avg_saved_pct, total_saved_chars,
        total_est_tokens, summary_reuse_count, recent_samples}。
    """
    default: dict[str, Any] = {
        "total": 0,
        "tier_distribution": {},
        "avg_saved_pct": 0.0,
        "total_saved_chars": 0,
        "total_est_tokens": 0,
        "summary_reuse_count": 0,
        "recent_samples": [],
    }
    client = _get_client()
    if client is None:
        return default
    try:
        raw_list = client.lrange(_CTX_STATS_KEY, 0, limit - 1)
        entries: list[dict[str, Any]] = []
        for raw in raw_list:
            try:
                entries.append(json.loads(raw))
            except Exception:
                continue
        if not entries:
            return default

        tier_dist: dict[str, int] = {}
        total_saved = 0
        total_tokens = 0
        reuse = 0
        for e in entries:
            tier = int(e.get("tier", 0))
            tier_dist[str(tier)] = tier_dist.get(str(tier), 0) + 1
            total_saved += int(e.get("saved_chars", 0))
            total_tokens += int(e.get("est_tokens", 0))
            if e.get("summary_reused"):
                reuse += 1

        avg_saved = (
            sum(float(e.get("saved_pct", 0.0)) for e in entries) / len(entries)
        )
        return {
            "total": len(entries),
            "tier_distribution": tier_dist,
            "avg_saved_pct": round(avg_saved, 3),
            "total_saved_chars": total_saved,
            "total_est_tokens": total_tokens,
            "summary_reuse_count": reuse,
            "recent_samples": entries[:20],
        }
    except Exception:
        return default


def _summarize_messages(messages: list[dict[str, Any]], llm_cfg: dict[str, Any] | None) -> str:
    """用 LLM 把旧消息压缩成一段摘要；失败返回空串（调用方静默降级）。

    成本控制：输入先裁剪到 ``summarize_trim_chars``（保留首条锚点 + 最近内容），
    模型档位按 ``context.llm_compact_model`` 选择 flash（默认）或主模型。
    """
    if not messages:
        return ""
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        cfg = get_context_config()
        trim_chars = int(cfg.get("summarize_trim_chars", 8000))
        summary_cap = int(cfg.get("summary_max_chars", 400))

        # 输入裁剪：保留首条锚点（身份/诉求）+ 最近内容，控制一次性 token 成本
        trimmed = _trim_for_summarize(messages, trim_chars)
        lines = "\n".join(
            f"{'用户' if m.get('role') == 'user' else '助手'}：{m.get('content', '')}"
            for m in trimmed
        )
        model = _build_compact_model(llm_cfg, cfg)
        if model is None:
            return ""
        chain = (
            ChatPromptTemplate.from_messages([
                ("system", COMPACT_SUMMARY_PROMPT),
                ("human", "对话：\n\n{text}"),
            ])
            | model
            | StrOutputParser()
        )
        summary = str(chain.invoke({"text": lines})).strip()
        return summary[:summary_cap]
    except Exception:
        return ""


def _trim_for_summarize(
    messages: list[dict[str, Any]],
    max_chars: int,
) -> list[dict[str, Any]]:
    """摘要输入裁剪：保留首条（锚点）+ 最近内容，总量不超过 max_chars。

    与官方 SummarizationMiddleware 的 ``trim_tokens_to_summarize`` 对应：
    摘要不必全量喂入——最旧、信息密度最低的中间轮次可以裁掉。
    """
    if not messages:
        return []
    total = sum(len(str(m.get("content", ""))) for m in messages)
    if total <= max_chars:
        return messages
    head = messages[:1]
    budget = max_chars - len(str(head[0].get("content", "")))
    tail: list[dict[str, Any]] = []
    used = 0
    for m in reversed(messages[1:]):
        content = str(m.get("content", ""))
        if used + len(content) > budget:
            rest = budget - used
            if rest > 0:
                tail.append({**m, "content": content[:rest]})
            break
        tail.append(m)
        used += len(content)
    return head + list(reversed(tail))


def _build_compact_model(llm_cfg: dict[str, Any] | None, cfg: dict[str, Any]):
    """按 context.llm_compact_model 选择摘要模型：flash（默认）或主模型。

    flash 档位复用 configs/app.yaml ``framework.agent.model_flash``，
    成本敏感（摘要不面向用户展示，flash 质量足够）。
    """
    try:
        from agent_base.config import deep_get, load_yaml
        from agent_base.retrieval.summarizer import _build_chat_model_with_timeout

        if str(cfg.get("llm_compact_model", "flash")) == "flash":
            app_cfg = load_yaml("configs/app.yaml") or {}
            flash_model = deep_get(app_cfg, "framework.agent.model_flash")
            if flash_model:
                merged = dict(llm_cfg or {})
                merged["model"] = flash_model
                merged.setdefault("provider", "langchain")
                merged.setdefault(
                    "base_url",
                    deep_get(app_cfg, "framework.agent.base_url")
                    or deep_get(app_cfg, "llm.base_url"),
                )
                merged.setdefault(
                    "api_key_env",
                    deep_get(app_cfg, "framework.agent.api_key_env")
                    or deep_get(app_cfg, "llm.api_key_env")
                    or "ANTHROPIC_AUTH_TOKEN",
                )
                model = _build_chat_model_with_timeout(merged, timeout=30)
                if model is not None:
                    return model
        return _build_chat_model_with_timeout(llm_cfg or {}, timeout=30)
    except Exception:
        return None


def compact_chat_history(
    session_id: str,
    llm_cfg: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """上下文压缩：历史字符超预算阈值时，LLM 摘要旧消息并保留最近 N 条原文。

    - test- 会话（Redis 短期记忆）：原地压缩，旧消息被摘要替换；
    - 真实会话（PG 短期记忆）：PG 原文不动（审计/客服追溯的权威存储），
      摘要写入 Redis ``chat:meta:{sid}``，由 ``get_chat_history`` 置顶复用，
      档 3 规则压缩可 0 LLM 复用该摘要。

    压缩记录统一写入 ``chat:meta:{sid}``，供记忆测试台展示
    「何时压缩 / 压缩前后用量 / 压缩历史」。

    Args:
        session_id: 会话 ID。
        llm_cfg: LLM 配置（provider/model/base_url/api_key_env）。

    Returns:
        压缩后的元数据 dict；未触发 / 失败返回 None。
    """
    if is_test_session(session_id):
        return _compact_redis(session_id, llm_cfg)
    return _compact_pg(session_id, llm_cfg)


def _compact_redis(
    session_id: str,
    llm_cfg: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Redis（test- 会话）压缩：旧消息原地替换为摘要。"""
    client = _get_client()
    if client is None:
        return None
    try:
        cfg = get_context_config()
        keep_recent = int(cfg.get("compact_keep_recent", COMPACT_KEEP_RECENT))
        budget = int(cfg.get("budget_chars", CONTEXT_BUDGET_CHARS))
        trigger = float(cfg.get("compact_trigger_ratio", COMPACT_TRIGGER_RATIO))

        raw = client.get(_key(session_id))
        if not raw:
            return None
        messages = json.loads(raw)
        if not isinstance(messages, list) or len(messages) <= keep_recent + 2:
            return None
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        if total_chars < budget * trigger:
            return None

        old = messages[:-keep_recent]
        recent = messages[-keep_recent:]
        summary = _summarize_messages(old, llm_cfg)
        if not summary:
            return None

        new_messages = [{"role": "system", "content": f"【对话摘要】{summary}"}, *recent]
        client.set(
            _key(session_id),
            json.dumps(new_messages, ensure_ascii=False),
            ex=CHAT_MEMORY_TTL,
        )
        after_chars = sum(len(str(m.get("content", ""))) for m in new_messages)

        meta = get_history_meta(session_id)
        meta["rounds"] = int(meta.get("rounds", 0)) + 1
        meta["last_compacted_at"] = datetime.now(timezone.utc).isoformat()
        meta["before_chars"] = total_chars
        meta["after_chars"] = after_chars
        hist = list(meta.get("history") or [])
        hist.append({
            "at": meta["last_compacted_at"],
            "before": total_chars,
            "after": after_chars,
            "rounds": meta["rounds"],
        })
        meta["history"] = hist[-5:]
        _save_history_meta(session_id, meta)
        return meta
    except Exception:
        return None


def _compact_pg(
    session_id: str,
    llm_cfg: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """PG（真实会话）压缩：原文不动，摘要写入 Redis ``chat:meta:{sid}``。"""
    client = _get_client()
    if client is None:
        return None
    try:
        cfg = get_context_config()
        keep_recent = int(cfg.get("compact_keep_recent", COMPACT_KEEP_RECENT))
        budget = int(cfg.get("budget_chars", CONTEXT_BUDGET_CHARS))
        trigger = float(cfg.get("compact_trigger_ratio", COMPACT_TRIGGER_RATIO))

        messages = _pg_history(session_id, limit=200)
        if not messages or len(messages) <= keep_recent + 2:
            return None
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        if total_chars < budget * trigger:
            return None

        old = messages[:-keep_recent]
        recent = messages[-keep_recent:]
        summary = _summarize_messages(old, llm_cfg)
        if not summary:
            return None

        after_chars = sum(len(str(m.get("content", ""))) for m in recent) + len(summary)
        meta = get_history_meta(session_id)
        meta["rounds"] = int(meta.get("rounds", 0)) + 1
        meta["last_compacted_at"] = datetime.now(timezone.utc).isoformat()
        meta["before_chars"] = total_chars
        meta["after_chars"] = after_chars
        meta["summary"] = summary
        hist = list(meta.get("history") or [])
        hist.append({
            "at": meta["last_compacted_at"],
            "before": total_chars,
            "after": after_chars,
            "rounds": meta["rounds"],
        })
        meta["history"] = hist[-5:]
        _save_history_meta(session_id, meta)
        return meta
    except Exception:
        return None
