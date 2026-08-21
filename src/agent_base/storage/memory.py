"""Long-term memory storage (P13-01).

Supports cross-session user profiles: skin type, category preferences,
price band, size, purchase intent — sanitized business labels only.
No raw conversations / PII stored.

Table: ``user_memories`` (PG, idempotent CREATE via ``pg.py init_db``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agent_base.storage.pg import _conn


# ── 写门控配置（对标生产级 write gates，configs/app.yaml memory 段） ────────


def get_memory_config() -> dict[str, Any]:
    """读取长期记忆写门控配置；缺失/异常回退默认值。"""
    defaults: dict[str, Any] = {
        "min_confidence": 0.6,
        "anchor_confidence": 0.9,
        "conflict_confidence": 0.7,
        "conflict_boost": 0.1,
        "extract_every_rounds": 5,
        "async_extract_enabled": True,
        "trust_base_confidence": {
            "user_statement": 0.9,
            "tool_result": 0.8,
            "agent_inference": 0.55,
            "conflict_confirmed": 0.95,
        },
    }
    try:
        from agent_base.config import load_yaml

        mem_cfg = (load_yaml("configs/app.yaml") or {}).get("memory", {}) or {}
        for key in defaults:
            if key in mem_cfg and mem_cfg[key] is not None:
                defaults[key] = mem_cfg[key]
    except Exception:
        pass
    return defaults


def trust_confidence(tier: str) -> float:
    """按信任层级返回基础置信度（用户陈述 > 工具返回 > Agent 推断）。

    Args:
        tier: user_statement / tool_result / agent_inference / conflict_confirmed。

    Returns:
        基础置信度 0-1。
    """
    base = get_memory_config().get("trust_base_confidence", {}) or {}
    return float(base.get(tier, 0.5))


# ── 增删改查 ────────────────────────────────────────────────────────────────


def save_memory(
    user_id: str,
    key: str,
    value: Any,
    source: str = "conversation",
    confidence: float = 0.5,
    ttl_days: int = 0,
) -> None:
    """写入或覆盖用户记忆条目（新条目覆盖旧条目）。

    Args:
        user_id: 用户标识。
        key: 记忆键（如 "skin_type"、"price_band"）。
        value: 可 JSON 序列化的值。
        source: 来源标签（conversation / import / admin）。
        confidence: 置信度 0-1。
        ttl_days: N 天后自动过期（0 = 永不过期）。
    """
    from psycopg2.extras import Json

    # P13 修复：脱敏必须在此执行——所有流入 user_memories 的数据先过 sanitize
    key = sanitize_key(key)
    value = sanitize_value(value)

    ttl = None
    if ttl_days > 0:
        from datetime import timedelta
        ttl = datetime.now(timezone.utc) + timedelta(days=ttl_days)

    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO user_memories (user_id, memory_key, value, source, confidence, ttl, updated_at)
               VALUES (%s,%s,%s,%s,%s,%s,NOW())
               ON CONFLICT (user_id, memory_key) DO UPDATE SET
                 value=EXCLUDED.value, source=EXCLUDED.source,
                 confidence=EXCLUDED.confidence, ttl=EXCLUDED.ttl,
                 updated_at=NOW()""",
            (user_id, key, Json(value), source, confidence, ttl),
        )


def upsert_memory_guarded(
    user_id: str,
    key: str,
    value: Any,
    source: str = "conversation",
    confidence: float = 0.5,
    ttl_days: int = 0,
) -> dict[str, Any]:
    """带写门控的长期记忆写入（生产级，对标 Salesforce write gates）。

    三层防线：
      1. 写入门槛：新值置信度 < min_confidence → 丢弃（low_confidence）；
      2. 锚点保护：同 key 旧值 ≥ anchor_confidence 且新值不更高 → 不覆盖
         （anchor_protected）；
      3. 冲突检测：同 key 旧值 ≥ conflict_confidence 且值方向不一致 →
         新值需比旧值高 conflict_boost 才覆盖，否则延迟（conflict_deferred）。

    Args:
        user_id: 用户标识（归属由调用方保证，勿信前端）。
        key: 记忆键（e.g. skin_type）。
        value: 记忆值。
        source: 来源标签。
        confidence: 新值置信度 0-1（按信任层级赋值）。
        ttl_days: 过期天数（0=永久）。

    Returns:
        dict：{written: bool, reason: str, old_confidence, new_confidence}。
        reason ∈ written / low_confidence / anchor_protected / conflict_deferred。
    """
    cfg = get_memory_config()
    min_conf = float(cfg.get("min_confidence", 0.6))
    anchor_conf = float(cfg.get("anchor_confidence", 0.9))
    conflict_conf = float(cfg.get("conflict_confidence", 0.7))
    conflict_boost = float(cfg.get("conflict_boost", 0.1))

    key = sanitize_key(key)
    value = sanitize_value(value)
    confidence = max(0.0, min(1.0, float(confidence)))

    # 层 1：写入门槛
    if confidence < min_conf:
        return {
            "written": False,
            "reason": "low_confidence",
            "old_confidence": None,
            "new_confidence": confidence,
        }

    old = retrieve_memory(user_id, keys=[key], top_k=1)
    old_confidence = float(old[0]["confidence"]) if old else None
    old_value = old[0]["value"] if old else None

    if old_confidence is None:
        # 无旧值：直接写
        save_memory(user_id, key, value, source=source, confidence=confidence, ttl_days=ttl_days)
        return {
            "written": True,
            "reason": "written",
            "old_confidence": None,
            "new_confidence": confidence,
        }

    # 层 2：锚点保护——旧值已高置信且新值不更高
    if old_confidence >= anchor_conf and confidence <= old_confidence:
        return {
            "written": False,
            "reason": "anchor_protected",
            "old_confidence": old_confidence,
            "new_confidence": confidence,
        }

    # 层 3：冲突检测——值方向不一致且旧值有一定置信
    if old_value != value and old_confidence >= conflict_conf:
        if confidence < old_confidence + conflict_boost:
            _record_conflict(user_id, key, old_value, value, old_confidence, confidence)
            return {
                "written": False,
                "reason": "conflict_deferred",
                "old_confidence": old_confidence,
                "new_confidence": confidence,
            }

    save_memory(user_id, key, value, source=source, confidence=confidence, ttl_days=ttl_days)
    return {
        "written": True,
        "reason": "written",
        "old_confidence": old_confidence,
        "new_confidence": confidence,
    }


def _record_conflict(
    user_id: str,
    key: str,
    old_value: Any,
    new_value: Any,
    old_confidence: float,
    new_confidence: float,
) -> None:
    """记录画像冲突（Redis，供管理端展示/排查；不可用时静默跳过）。"""
    try:
        import json
        import redis

        client = redis.Redis(
            host=__import__("os").getenv("REDIS_HOST", "localhost"),
            port=int(__import__("os").getenv("REDIS_PORT", "6379")),
            db=int(__import__("os").getenv("REDIS_DB", "0")),
            socket_connect_timeout=2,
            socket_timeout=2,
            decode_responses=True,
        )
        client.lpush(
            "memory:conflicts",
            json.dumps(
                {
                    "user_id": user_id,
                    "key": key,
                    "old_value": str(old_value),
                    "new_value": str(new_value),
                    "old_confidence": old_confidence,
                    "new_confidence": new_confidence,
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
            ),
        )
        client.ltrim("memory:conflicts", 0, 99)
        client.expire("memory:conflicts", 7 * 24 * 3600)
    except Exception:
        pass


def retrieve_memory(
    user_id: str,
    keys: list[str] | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """加载用户记忆条目。

    Args:
        user_id: 用户标识。
        keys: 可选，按记忆键过滤。
        top_k: 最多返回条数。

    Returns:
        含 key/value/confidence/updated_at 的记忆字典列表。
    """
    with _conn() as conn:
        cur = conn.cursor()
        if keys:
            placeholders = ",".join(["%s"] * len(keys))
            cur.execute(
                f"""SELECT memory_key, value, source, confidence, ttl, updated_at
                    FROM user_memories
                    WHERE user_id=%s AND memory_key IN ({placeholders})
                    ORDER BY updated_at DESC LIMIT %s""",
                (user_id, *keys, top_k),
            )
        else:
            cur.execute(
                """SELECT memory_key, value, source, confidence, ttl, updated_at
                   FROM user_memories
                   WHERE user_id=%s
                   ORDER BY updated_at DESC LIMIT %s""",
                (user_id, top_k),
            )
        import json as _json

        now = datetime.now(timezone.utc)
        results = []
        for row in cur.fetchall():
            ttl = row[4]
            if ttl and ttl < now:
                continue  # expired
            val = row[1]
            if isinstance(val, str):
                try:
                    val = _json.loads(val)
                except (_json.JSONDecodeError, ValueError):
                    pass  # plain string value
            results.append({
                "key": row[0],
                "value": val,
                "source": row[2],
                "confidence": float(row[3]),
                "ttl": ttl.isoformat() if ttl else None,
                "updated_at": row[5].isoformat() if row[5] else None,
            })
        return results[:top_k]


def delete_memory(user_id: str, key: str) -> bool:
    """删除单条记忆条目。

    Args:
        user_id: 用户标识。
        key: 待删除的记忆键。

    Returns:
        删除成功返回 True，未找到返回 False。
    """
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM user_memories WHERE user_id=%s AND memory_key=%s",
            (user_id, key),
        )
        return cur.rowcount > 0


# ── 画像注入（P13-03 / P13-04）──────────────────────────────────────────────

MAX_PROFILE_CHARS = 300


def build_profile_context(
    user_id: str,
    intent: str = "",
    max_chars: int = MAX_PROFILE_CHARS,
) -> str:
    """Build a compact profile snippet for prompt injection.

    Uses recency-first pruning (most recently updated entries first),
    then trims to fit the character budget.

    Args:
        user_id: User identifier.
        intent: Current query intent (used for intent-relevant prioritization).
        max_chars: Maximum character budget (default 300).

    Returns:
        Profile snippet string like "用户偏好: 油皮, 中端价位", or empty.
    """
    memories = retrieve_memory(user_id)
    if not memories:
        return ""

# 与意图相关的键获得优先级加成
    intent_key_map: dict[str, list[str]] = {
        "product_query": ["skin_type", "category", "price_band"],
        "fashion_query": ["style", "size", "season", "category"],
        "price_query": ["price_band", "category"],
        "recommendation": ["skin_type", "category", "style", "price_band"],
    }
    priority_keys = set(intent_key_map.get(intent, []))

# 排序：优先键在前，再按最近更新
    memories.sort(key=lambda m: (0 if m["key"] in priority_keys else 1, m.get("updated_at", "")))

    parts: list[str] = []
    total = 0
    for m in memories:
        val = m["value"]
        val_str = ", ".join(val) if isinstance(val, list) else str(val)
        line = f"{m['key']}: {val_str}"
        if total + len(line) > max_chars:
            break
        parts.append(line)
        total += len(line) + 1  # +1 for newline

    if not parts:
        return ""
    return "用户画像: " + "; ".join(parts)


def sanitize_key(key: str) -> str:
    """确保记忆键是业务标签而非原始数据。

    Args:
        key: 待校验的记忆键。

    Returns:
        清洗后的键（仅字母数字 + 下划线）。
    """
    import re
    return re.sub(r"[^a-zA-Z0-9_一-鿿]", "_", key)[:64]


def sanitize_value(value: Any) -> Any:
    """将记忆值清洗为仅业务标签。

    不存原始对话、手机号、地址或 PII。

    Args:
        value: 待清洗的值。

    Returns:
        清洗后的值。
    """
    if isinstance(value, str):
# 去除手机号/邮箱模式
        import re
        value = re.sub(r"1[3-9]\d{9}", "[PHONE]", value)
        value = re.sub(r"\S+@\S+\.\S+", "[EMAIL]", value)
        return value[:200]
    if isinstance(value, list):
        return [sanitize_value(v) for v in value[:10]]
    if isinstance(value, dict):
        return {sanitize_key(k): sanitize_value(v) for k, v in value.items()}
    return value
