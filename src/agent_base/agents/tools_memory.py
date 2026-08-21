"""Memory management tools (P13-02).

Four tools exposed to the e-commerce agent for long-term memory
operations, plus a session summarizer for end-of-session extraction.

Tools:
  - save_memory_tool
  - retrieve_memory_tool
  - update_memory_tool
  - delete_memory_tool
  - summarize_session (not a tool — called at end of conversation)
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig


def make_save_memory_tool():
    """创建 save_memory LangChain 工具。

    写入或覆盖用户记忆条目（如 skin_type: "油皮"），存储前做业务标签清洗。

    归属安全：user_id 不从工具参数取，而是从 RunnableConfig 注入
    （调用方从 token 解析），防止 LLM/前端伪造归属。
    """
    from agent_base.storage.memory import save_memory as _save, sanitize_key, sanitize_value

    def save_memory_tool(
        user_id: str,
        key: str,
        value: str,
        config: RunnableConfig,
        source: str = "conversation",
        confidence: float = 0.5,
    ) -> str:
        """保存用户记忆条目；同 key 覆盖旧值。

        Args:
            user_id: 已废弃——归属以调用方注入为准（config.user_id），忽略此参数。
            key: 记忆标签（如 skin_type、price_band、category）。
            value: 可 JSON 序列化的记忆值。
            config: RunnableConfig，归属 user_id 从中注入。
            source: 来源标签（conversation / import / admin）。
            confidence: 置信度 0-1（默认 0.5）。

        Returns:
            保存结果文本。
        """
        user_id = (config.get("configurable", {}) or {}).get("user_id", "")
        if not user_id:
            return "归属校验失败：未注入用户身份，拒绝写入"
        key = sanitize_key(key)
        value = sanitize_value(value)
        _save(user_id, key, value, source=source, confidence=confidence)
        return f"已保存: {key} = {value} (归属: {user_id})"

    from langchain_core.tools import tool
    return tool(save_memory_tool)


def make_retrieve_memory_tool():
    """创建 retrieve_memory LangChain 工具。

    读取用户画像（最多 top_k 条）。
    """

    def retrieve_memory_tool(
        user_id: str,
        keys: list[str] | None = None,
        top_k: int = 10,
        config: RunnableConfig | None = None,
    ) -> str:
        """读取用户记忆条目。

        Args:
            user_id: 用户标识（必填；config 注入时优先用注入值）。
            keys: 可选，按记忆键过滤。
            top_k: 最多返回条数。
            config: RunnableConfig，归属 user_id 从中注入。

        Returns:
            格式化后的记忆文本或提示文本。
        """
        user_id = ((config or {}).get("configurable", {}) or {}).get("user_id", user_id)
        from agent_base.storage.memory import retrieve_memory as _retrieve

        entries = _retrieve(user_id, keys=keys, top_k=top_k)
        if not entries:
            return "暂无用户画像记录。"
        lines = []
        for e in entries:
            val = e["value"]
            val_str = ", ".join(val) if isinstance(val, list) else str(val)
            lines.append(f"{e['key']}: {val_str} (置信度: {e['confidence']})")
        return "\n".join(lines)

    from langchain_core.tools import tool
    return tool(retrieve_memory_tool)


def make_update_memory_tool():
    """创建 update_memory LangChain 工具。

    更新已有记忆条目的置信度/来源（key + user_id 必须匹配）。
    """

    def update_memory_tool(
        user_id: str,
        key: str,
        config: RunnableConfig,
        confidence: float | None = None,
        source: str | None = None,
    ) -> str:
        """更新已有记忆条目的置信度或来源。

        Args:
            user_id: 用户标识（以 config 注入为准）。
            key: 待更新的记忆键。
            config: RunnableConfig，归属 user_id 从中注入。
            confidence: 新置信度（可选）。
            source: 新来源标签（可选）。

        Returns:
            更新结果文本。
        """
        user_id = (config.get("configurable", {}) or {}).get("user_id", "")
        if not user_id:
            return "归属校验失败：未注入用户身份，拒绝更新"
        from agent_base.storage.memory import retrieve_memory, save_memory as _save, sanitize_key

        key = sanitize_key(key)
        entries = retrieve_memory(user_id, keys=[key], top_k=1)
        if not entries:
            return f"未找到记忆项: {key}"

        entry = entries[0]
        new_conf = confidence if confidence is not None else entry["confidence"]
        new_source = source or entry["source"]
        _save(user_id, key, entry["value"], source=new_source, confidence=new_conf)
        return f"已更新: {key} (置信度: {new_conf})"

    from langchain_core.tools import tool
    return tool(update_memory_tool)


def make_delete_memory_tool():
    """创建 delete_memory LangChain 工具。

    删除单条记忆条目（用户遗忘权）。
    """

    def delete_memory_tool(user_id: str, key: str, config: RunnableConfig) -> str:
        """删除用户记忆条目。

        Args:
            user_id: 用户标识（以 config 注入为准）。
            key: 待删除的记忆键。
            config: RunnableConfig，归属 user_id 从中注入。

        Returns:
            删除结果文本。
        """
        user_id = (config.get("configurable", {}) or {}).get("user_id", "")
        if not user_id:
            return "归属校验失败：未注入用户身份，拒绝删除"
        from agent_base.storage.memory import delete_memory as _delete, sanitize_key

        key = sanitize_key(key)
        ok = _delete(user_id, key)
        return f"已删除: {key}" if ok else f"未找到记忆项: {key}"

    from langchain_core.tools import tool
    return tool(delete_memory_tool)


def maybe_async_extract(
    session_id: str,
    user_id: str,
    model_cfg: dict[str, Any] | None = None,
    history_count: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    """对话完成后按策略触发异步画像提炼（不阻塞响应流）。

    防抖策略（configs/app.yaml memory 段）：
      - 开启 async_extract_enabled 才触发；
      - 会话轮数 ≥ extract_every_rounds；
      - 距离上次提炼 ≥ extract_every_rounds 轮（Redis 记录提炼时间戳）。

    Args:
        session_id: 会话 ID。
        user_id: 用户标识（归属已由调用方从 token 解析）。
        model_cfg: LLM 配置。
        history_count: 当前会话历史条数（防抖用）。
        force: 强制提炼（管理端手动提炼走此入口，跳过轮数防抖）。

    Returns:
        dict：{triggered, skipped_reason, saved_count, rejected}。
    """
    try:
        from agent_base.storage.memory import get_memory_config, upsert_memory_guarded

        cfg = get_memory_config()
        if not force and not bool(cfg.get("async_extract_enabled", True)):
            return {"triggered": False, "skipped_reason": "async_disabled"}
        if not session_id or not user_id:
            return {"triggered": False, "skipped_reason": "missing_ids"}

        every = int(cfg.get("extract_every_rounds", 5))
        # 防抖：会话轮数不足 / 距上次提炼不足
        if not force and history_count < every * 2:
            return {"triggered": False, "skipped_reason": "too_few_rounds"}
        if not force:
            from agent_base.storage.chat_memory import get_chat_history

            last = _last_extract_at(session_id)
            if last is not None:
                # 简单轮数近似：历史条数变化小于 every 则视为未到下次提炼
                history_now = len(get_chat_history(session_id, limit=100))
                if history_now - last.get("history_count", 0) < every:
                    return {"triggered": False, "skipped_reason": "cooldown"}

        from agent_base.storage.chat_memory import get_chat_history

        history = get_chat_history(session_id, limit=32)
        if not history:
            return {"triggered": False, "skipped_reason": "empty_history"}

        items = summarize_session(history, user_id=user_id, model_cfg=model_cfg)
        if not items:
            return {"triggered": False, "skipped_reason": "no_items"}

        saved: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for item in items:
            guard = upsert_memory_guarded(
                user_id,
                item["key"],
                item["value"],
                source="conversation",
                confidence=float(item.get("confidence", 0.5)),
            )
            if guard.get("written"):
                saved.append({**item, "reason": "written"})
            else:
                rejected.append({**item, "reason": guard.get("reason")})

        _mark_extract_at(session_id, len(history))
        return {
            "triggered": True,
            "saved_count": len(saved),
            "saved": saved,
            "rejected": rejected,
            "rejected_count": len(rejected),
        }
    except Exception as exc:
        return {"triggered": False, "skipped_reason": f"error:{exc}"}


_EXTRACT_KEY_PREFIX = "memory:extract:"


def _last_extract_at(session_id: str) -> dict[str, Any] | None:
    """读上次提炼标记（Redis；不可用/无记录返回 None）。"""
    try:
        import json
        import os
        import redis

        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            socket_connect_timeout=2,
            socket_timeout=2,
            decode_responses=True,
        )
        raw = client.get(f"{_EXTRACT_KEY_PREFIX}{session_id}")
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _mark_extract_at(session_id: str, history_count: int) -> None:
    """记录提炼时间戳（Redis，TTL 7 天）。"""
    try:
        import json
        import os
        from datetime import datetime, timezone
        import redis

        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            socket_connect_timeout=2,
            socket_timeout=2,
            decode_responses=True,
        )
        client.set(
            f"{_EXTRACT_KEY_PREFIX}{session_id}",
            json.dumps({
                "at": datetime.now(timezone.utc).isoformat(),
                "history_count": history_count,
            }),
            ex=7 * 24 * 3600,
        )
    except Exception:
        pass


# ── 会话摘要器（P13-02）─────────────────────────────────────────────────────


def summarize_session(
    conversation: list[dict[str, str]],
    user_id: str = "",
    model_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract long-term memory entries from a completed conversation.

    Uses DeepSeek-flash to extract structured business labels (skin_type,
    category, price_band, size, intent, etc.) from the transcript.

    Args:
        conversation: List of {"role": "user"/"assistant", "content": "..."} turns.
        user_id: User identifier for the memory.
        model_cfg: LLM config (defaults to DeepSeek-flash).

    Returns:
        List of memory dicts {key, value, confidence, tier} ready for
        upsert_memory_guarded（信任层级已校准置信度）。
    """
    if not conversation:
        return []

    cfg = model_cfg or {
        "provider": "langchain",
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "temperature": 0.0,
    }

    transcript = "\n".join(
        f"{t['role']}: {t['content'][:300]}" for t in conversation[-10:]
    )

    from agent_base.prompts import get_prompt
    system_prompt = get_prompt("memory", "system")
    user_template = get_prompt("memory", "user_template")

    try:
        from agent_base.llms import build_chat_model
        model = build_chat_model(
            provider=cfg["provider"],
            model=cfg["model"],
            base_url=cfg["base_url"],
            api_key_env=cfg["api_key_env"],
            temperature=float(cfg.get("temperature", 0.0)),
        )
        if model is None:
            return []

        from agent_base.structured import MemoryItem, parse_json_or_none
        # P26b：直接以消息列表调用（system 提示词含 JSON 花括号，
        # 不用 ChatPromptTemplate 解析，避免被当成模板变量）
        from langchain_core.messages import HumanMessage, SystemMessage

        user_prompt = user_template.format(transcript=transcript)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        items: list[dict[str, Any]] | None = None
        try:
            resp = model.with_structured_output(list[MemoryItem]).invoke(messages)
            if isinstance(resp, list):
                items = [item.model_dump() for item in resp if isinstance(item, MemoryItem)]
        except Exception:
            items = None
        if items is None:
            # LCEL 兜底：同链无结构化输出（JSON 文本解析）
            try:
                resp = model.invoke(messages)
                text = getattr(resp, "content", str(resp)).strip()
            except Exception:
                text = ""
            data = parse_json_or_none(text)
            items = data if isinstance(data, list) else None
        if items:
            result: list[dict[str, Any]] = []
            for item in items:
                if not (isinstance(item, dict) and "key" in item and "value" in item):
                    continue
                tier = str(item.get("tier", "user_statement"))
                if tier not in {"user_statement", "tool_result", "agent_inference", "conflict_confirmed"}:
                    tier = "user_statement"
                # 信任层级定基础置信度；LLM 微调只允许在基础值附近小幅浮动
                from agent_base.storage.memory import trust_confidence

                base = trust_confidence(tier)
                llm_conf = float(item.get("confidence", 0.5))
                # agent_inference 基础分低于门槛，即使 LLM 给高分也压回门槛下
                if tier == "agent_inference":
                    confidence = min(base, llm_conf * 0.6)
                else:
                    confidence = max(base, min(1.0, llm_conf))
                result.append({
                    "key": item["key"],
                    "value": item["value"],
                    "confidence": round(confidence, 3),
                    "tier": tier,
                })
            return result
    except Exception:
        pass

    return []
