"""Postgres 结构化存储（P9-03）。

表设计：
  catalog   — 商品目录（PG 运行时真相源）
  documents — 内容版本管理（doc_id / version / status / updated_at）
  orders    — 电商 mock 工具升级为真实表（接口不变）
  inventory — 库存表

连接：postgresql://postgres:ragdb@localhost:5432/ragdb
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from typing import Any

import os

import psycopg2
import psycopg2.extras
from psycopg2 import pool as _pgpool

DB_URL = os.getenv("PG_URL", "postgresql://postgres:postgres@localhost:5432/ragdb")

# 生产级连接池（ThreadedConnectionPool）：避免每次查询新建连接，
# 单连接建连/销毁开销 + 打满 PG 连接数。池大小可用环境变量调优：
# PG_POOL_MIN（默认 2）/ PG_POOL_MAX（默认 20）。
_pool_lock = threading.Lock()
_pool: Any = None
_pool_disabled = os.getenv("PG_POOL_DISABLED", "").strip().lower() in {"1", "true", "yes"}


def _get_pool() -> Any:
    """惰性创建连接池（失败返回 None，回退每次新建连接）。"""
    global _pool
    if _pool_disabled:
        return None
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                try:
                    _pool = _pgpool.ThreadedConnectionPool(
                        int(os.getenv("PG_POOL_MIN", "2")),
                        int(os.getenv("PG_POOL_MAX", "20")),
                        DB_URL,
                    )
                except Exception:
                    _pool = None
    return _pool


@contextmanager
def _conn():
    """获取 psycopg2 连接（连接池优先；关闭 autocommit，事务性）。

    每次调用结束时 commit（正常）或 rollback（异常），连接归还池中复用；
    池不可用时回退为每次新建连接（保持既有行为）。
    """
    pool = _get_pool()
    if pool is not None:
        conn = pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            pool.putconn(conn)
        return
    conn = psycopg2.connect(DB_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """创建全部表（幂等，已存在则跳过）。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS catalog (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                brand       TEXT DEFAULT '',
                category    TEXT DEFAULT '',
                price_band  TEXT DEFAULT '',
                metadata    JSONB DEFAULT '{}',
                updated_at  TIMESTAMPTZ DEFAULT NOW()
            );
            -- 商品图文素材（预置素材，不依赖大模型实时生成）
            CREATE TABLE IF NOT EXISTS product_media (
                id          BIGSERIAL PRIMARY KEY,
                product_id  TEXT NOT NULL,
                media_type  TEXT NOT NULL DEFAULT 'image',
                url         TEXT NOT NULL,
                title       TEXT DEFAULT '',
                sort_order  INT DEFAULT 0,
                source      TEXT DEFAULT 'preset',
                status      TEXT DEFAULT 'active',
                created_at  TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_product_media_product
                ON product_media (product_id, sort_order);
            -- 图片知识库（Phase 2）：图片文档级管理，独立于商品素材表 product_media
            CREATE TABLE IF NOT EXISTS media_documents (
                id            BIGSERIAL PRIMARY KEY,
                original_name TEXT DEFAULT '',
                url           TEXT NOT NULL,
                description   TEXT DEFAULT '',
                ocr_text      TEXT DEFAULT '',
                product_id    TEXT DEFAULT '',
                source_type   TEXT DEFAULT 'upload',
                status        TEXT DEFAULT 'pending',
                mime_type     TEXT DEFAULT '',
                size_bytes    INT DEFAULT 0,
                created_at    TIMESTAMPTZ DEFAULT NOW(),
                updated_at    TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_media_documents_status ON media_documents (status);
            CREATE INDEX IF NOT EXISTS idx_media_documents_product ON media_documents (product_id);
            -- 视频知识库扩展列（幂等迁移：老库直接补列，不重建表）
            ALTER TABLE media_documents ADD COLUMN IF NOT EXISTS video_urls JSONB DEFAULT '[]';
            ALTER TABLE media_documents ADD COLUMN IF NOT EXISTS poster_url TEXT DEFAULT '';
            ALTER TABLE media_documents ADD COLUMN IF NOT EXISTS duration_sec INT DEFAULT 0;
            ALTER TABLE media_documents ADD COLUMN IF NOT EXISTS parse_type TEXT DEFAULT 'image';
            -- 文件清洗草稿（两段式入库第一段：解析清洗 → 人工修改 → 推送到精审）
            CREATE TABLE IF NOT EXISTS clean_drafts (
                id            BIGSERIAL PRIMARY KEY,
                original_name TEXT DEFAULT '',
                engine        TEXT DEFAULT '',
                raw_text      TEXT DEFAULT '',
                cleaned_text  TEXT DEFAULT '',
                status        TEXT DEFAULT 'pending',
                created_at    TIMESTAMPTZ DEFAULT NOW(),
                updated_at    TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_clean_drafts_status ON clean_drafts (status);
            -- P30: 切分参数自定义覆盖（doc_type 主键；分隔符 JSON 数组）
            CREATE TABLE IF NOT EXISTS chunk_profile_overrides (
                doc_type       TEXT PRIMARY KEY,
                chunk_size     INT NOT NULL DEFAULT 0,
                chunk_overlap  INT NOT NULL DEFAULT 0,
                separators     JSONB NOT NULL DEFAULT '[]',
                updated_by     TEXT DEFAULT '',
                updated_at     TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS documents (
                doc_id      TEXT NOT NULL,
                version     INT NOT NULL DEFAULT 1,
                content     TEXT DEFAULT '',
                chunk_ids   TEXT[] DEFAULT '{}',
                status      TEXT DEFAULT 'active',
                metadata    JSONB DEFAULT '{}',
                created_at  TIMESTAMPTZ DEFAULT NOW(),
                updated_at  TIMESTAMPTZ DEFAULT NOW(),
                deleted_at  TIMESTAMPTZ DEFAULT NULL,
                PRIMARY KEY (doc_id, version)
            );
            -- P11-02: 生产级订单/用户/库存表
            CREATE TABLE IF NOT EXISTS users (
                user_id     TEXT PRIMARY KEY,
                nickname    TEXT DEFAULT '',
                level       TEXT DEFAULT 'normal',
                password_hash TEXT DEFAULT '',
                created_at  TIMESTAMPTZ DEFAULT NOW()
            );
            -- 买家账号与运营账号分表：users 承载买家登录（兼容旧表补列）
            ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT DEFAULT '';
            -- v0.53: 手机号 / 微信登录（唯一索引防重复绑定；未填时允许多个空串）
            ALTER TABLE users ADD COLUMN IF NOT EXISTS phone TEXT DEFAULT '';
            ALTER TABLE users ADD COLUMN IF NOT EXISTS wechat_openid TEXT DEFAULT '';
            CREATE UNIQUE INDEX IF NOT EXISTS uq_users_phone
                ON users (phone) WHERE phone <> '';
            CREATE UNIQUE INDEX IF NOT EXISTS uq_users_wechat
                ON users (wechat_openid) WHERE wechat_openid <> '';
            CREATE TABLE IF NOT EXISTS orders (
                order_id       TEXT PRIMARY KEY,
                order_no       TEXT UNIQUE NOT NULL,
                user_id        TEXT NOT NULL REFERENCES users(user_id),
                status         TEXT DEFAULT 'pending',
                total_amount   NUMERIC(10,2) DEFAULT 0,
                discount_amount NUMERIC(10,2) DEFAULT 0,
                pay_amount     NUMERIC(10,2) DEFAULT 0,
                currency       TEXT DEFAULT 'CNY',
                address_snapshot JSONB,
                payment_info   JSONB,
                version        INT DEFAULT 1,
                created_at     TIMESTAMPTZ DEFAULT NOW(),
                updated_at     TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS order_items (
                order_id     TEXT NOT NULL REFERENCES orders(order_id),
                product_id   TEXT NOT NULL,
                product_name TEXT NOT NULL,
                price        NUMERIC(10,2) NOT NULL,
                quantity     INT NOT NULL,
                amount       NUMERIC(10,2) NOT NULL,
                PRIMARY KEY (order_id, product_id)
            );
            CREATE TABLE IF NOT EXISTS order_status_log (
                id           BIGSERIAL PRIMARY KEY,
                order_id     TEXT NOT NULL REFERENCES orders(order_id),
                from_status  TEXT DEFAULT '',
                to_status    TEXT NOT NULL,
                operator     TEXT DEFAULT 'system',
                note         TEXT DEFAULT '',
                created_at   TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS inventory (
                product_id   TEXT PRIMARY KEY,
                quantity     INT DEFAULT 0,
                reserved     INT DEFAULT 0,
                status       TEXT DEFAULT 'on_sale',
                version      INT DEFAULT 0,
                updated_at   TIMESTAMPTZ DEFAULT NOW()
            );
            -- P13-01: 长期记忆表（跨会话用户画像）
            CREATE TABLE IF NOT EXISTS user_memories (
                user_id     TEXT NOT NULL,
                memory_key  TEXT NOT NULL,        -- 品类偏好/价格带/肤质/尺码/诉求
                value       JSONB NOT NULL,
                source      TEXT DEFAULT 'conversation',
                confidence  NUMERIC(3,2) DEFAULT 0.5,
                ttl         TIMESTAMPTZ,
                created_at  TIMESTAMPTZ DEFAULT NOW(),
                updated_at  TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (user_id, memory_key)
            );
            -- P14-01: 文档打标策略表（预审 Agent + 人工审核 + 手段路由）
            CREATE TABLE IF NOT EXISTS document_strategy (
                doc_id      TEXT NOT NULL,
                doc_type    TEXT DEFAULT '',
                strategy    TEXT[] DEFAULT '{}',
                reviewer    TEXT DEFAULT '',
                status      TEXT DEFAULT 'pending',
                updated_at  TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (doc_id)
            );
            -- P20: 知识入库暂存表（上传=暂存+自动预审，approved 才真正入库）
            CREATE TABLE IF NOT EXISTS document_staging (
                doc_id          TEXT PRIMARY KEY,
                filename        TEXT DEFAULT '',
                content         TEXT NOT NULL,
                category        TEXT DEFAULT '',
                status          TEXT DEFAULT 'pending',
                review_round    INT DEFAULT 1,
                first_review    JSONB DEFAULT '{}'::jsonb,
                reject_reason   TEXT DEFAULT '',
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                updated_at      TIMESTAMPTZ DEFAULT NOW()
            );
            -- P20: 对话消息（流式多轮上下文 + 会话列表）
            CREATE TABLE IF NOT EXISTS chat_messages (
                id          BIGSERIAL PRIMARY KEY,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                sources     JSONB,
                created_at  TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                ON chat_messages (session_id, created_at);
            CREATE TABLE IF NOT EXISTS chat_handoffs (
                session_id     TEXT PRIMARY KEY,
                status         TEXT NOT NULL DEFAULT 'pending',
                reason         TEXT DEFAULT '',
                agent_name     TEXT DEFAULT '',
                created_at     TIMESTAMPTZ DEFAULT NOW(),
                last_active_at TIMESTAMPTZ DEFAULT NOW(),
                resolved_at    TIMESTAMPTZ DEFAULT NULL
            );
            ALTER TABLE chat_handoffs ADD COLUMN IF NOT EXISTS rating INTEGER DEFAULT 0;
            ALTER TABLE chat_handoffs ADD COLUMN IF NOT EXISTS rating_comment TEXT DEFAULT '';
            ALTER TABLE chat_handoffs ADD COLUMN IF NOT EXISTS rated_at TIMESTAMPTZ DEFAULT NULL;
            -- SEC-2: 会话归属（买家只能读自己的会话；首条消息写入时绑定 owner）
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id  TEXT PRIMARY KEY,
                owner       TEXT NOT NULL DEFAULT '',
                created_at  TIMESTAMPTZ DEFAULT NOW(),
                updated_at  TIMESTAMPTZ DEFAULT NOW()
            );
            -- 全链路评测：批次 + 用例明细（意图→检索→生成→打分）
            CREATE TABLE IF NOT EXISTS eval_runs (
                id              BIGSERIAL PRIMARY KEY,
                name            TEXT DEFAULT '',
                total_cases     INT DEFAULT 0,
                intent_acc      REAL DEFAULT 0,
                recall_acc      REAL DEFAULT 0,
                fact_acc        REAL DEFAULT 0,
                compliance_acc  REAL DEFAULT 0,
                faithfulness_acc REAL DEFAULT 0,
                relevancy_acc    REAL DEFAULT 0,
                overall         REAL DEFAULT 0,
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS eval_cases (
                id              BIGSERIAL PRIMARY KEY,
                run_id          BIGINT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
                question        TEXT DEFAULT '',
                expected_intent TEXT DEFAULT '',
                actual_intent   TEXT DEFAULT '',
                intent_hit      BOOLEAN DEFAULT FALSE,
                expected_source TEXT DEFAULT '',
                recall_hit      BOOLEAN DEFAULT FALSE,
                expected_facts  JSONB DEFAULT '[]'::jsonb,
                fact_hits       INT DEFAULT 0,
                fact_total      INT DEFAULT 0,
                compliance_ok   BOOLEAN DEFAULT TRUE,
                faithfulness    REAL DEFAULT 0,
                relevancy       REAL DEFAULT 0,
                answer          TEXT DEFAULT '',
                sources         JSONB DEFAULT '[]'::jsonb,
                error           TEXT DEFAULT '',
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );
            -- P12-02: 意图规则库（版本化，支持管理端编辑与回滚）
            CREATE TABLE IF NOT EXISTS intent_rules (
                intent      TEXT NOT NULL,
                keywords    TEXT[] DEFAULT '{}',
                sections    TEXT[] DEFAULT '{}',
                examples    TEXT[] DEFAULT '{}',
                priority    NUMERIC(4,1) DEFAULT 1.0,
                version     INT DEFAULT 1,
                status      TEXT DEFAULT 'active',
                updated_at  TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (intent, version)
            );
            CREATE TABLE IF NOT EXISTS alias_rules (
                alias       TEXT NOT NULL,
                canonical   TEXT NOT NULL,
                updated_at  TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (alias, canonical)
            );
            CREATE TABLE IF NOT EXISTS faq (
                id          TEXT PRIMARY KEY,
                category    TEXT DEFAULT '',
                question    TEXT NOT NULL,
                answer      TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS idf_cache (
                version      TEXT PRIMARY KEY,
                table_json   JSONB NOT NULL,
                updated_at   TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS log_events (
                id          BIGSERIAL PRIMARY KEY,
                ts          TIMESTAMPTZ DEFAULT NOW(),
                level       TEXT NOT NULL,
                module      TEXT NOT NULL,
                event       TEXT NOT NULL,
                request_id  TEXT DEFAULT '-',
                data        JSONB DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_log_events_ts ON log_events (ts);
            CREATE TABLE IF NOT EXISTS product_reviews (
                id          BIGSERIAL PRIMARY KEY,
                product_id  TEXT NOT NULL,
                rating      INTEGER DEFAULT 5,
                content     TEXT NOT NULL,
                sentiment   TEXT DEFAULT 'positive',
                source      TEXT DEFAULT 'platform',
                created_at  TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_reviews_product ON product_reviews (product_id);
            CREATE TABLE IF NOT EXISTS product_combos (
                id          BIGSERIAL PRIMARY KEY,
                name        TEXT NOT NULL,
                product_ids JSONB NOT NULL,
                scenario    TEXT DEFAULT '',
                description TEXT DEFAULT '',
                created_at  TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS customer_cases (
                id          BIGSERIAL PRIMARY KEY,
                product_id  TEXT NOT NULL,
                skin_type   TEXT DEFAULT '',
                scenario    TEXT DEFAULT '',
                duration    TEXT DEFAULT '',
                result      TEXT DEFAULT '',
                created_at  TIMESTAMPTZ DEFAULT NOW()
            );
            -- 观测底座：Token 用量埋点（模型调用统一落库）
            CREATE TABLE IF NOT EXISTS token_usage (
                id              BIGSERIAL PRIMARY KEY,
                ts              TIMESTAMPTZ DEFAULT NOW(),
                request_id      TEXT DEFAULT '-',
                session_id      TEXT DEFAULT '',
                agent           TEXT DEFAULT '',
                model           TEXT DEFAULT '',
                source          TEXT DEFAULT '',
                prompt_tokens   INT DEFAULT 0,
                completion_tokens INT DEFAULT 0,
                total_tokens    INT DEFAULT 0,
                latency_ms      INT DEFAULT 0,
                ok              BOOLEAN DEFAULT TRUE,
                error           TEXT DEFAULT '',
                data            JSONB DEFAULT '{}'::jsonb
            );
            CREATE INDEX IF NOT EXISTS idx_token_usage_ts ON token_usage (ts);
            CREATE INDEX IF NOT EXISTS idx_token_usage_agent ON token_usage (agent);
            -- 观测底座：工具调用记录（成功/失败/耗时）
            CREATE TABLE IF NOT EXISTS tool_calls (
                id          BIGSERIAL PRIMARY KEY,
                ts          TIMESTAMPTZ DEFAULT NOW(),
                request_id  TEXT DEFAULT '-',
                session_id  TEXT DEFAULT '',
                agent       TEXT DEFAULT '',
                tool_name   TEXT NOT NULL,
                params      JSONB DEFAULT '{}'::jsonb,
                ok          BOOLEAN DEFAULT TRUE,
                error       TEXT DEFAULT '',
                latency_ms  INT DEFAULT 0,
                result_preview TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_tool_calls_ts ON tool_calls (ts);
            CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls (tool_name);
            -- 评测反馈：失败归因 + 数据飞轮状态流转
            CREATE TABLE IF NOT EXISTS eval_feedback (
                id              BIGSERIAL PRIMARY KEY,
                case_id         BIGINT DEFAULT 0,
                run_id          BIGINT DEFAULT 0,
                question        TEXT DEFAULT '',
                failure_type    TEXT DEFAULT '',
                detail          TEXT DEFAULT '',
                status          TEXT DEFAULT 'pending',
                regression      REAL DEFAULT 0,
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                updated_at      TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_eval_feedback_status ON eval_feedback (status);
            CREATE INDEX IF NOT EXISTS idx_eval_feedback_type ON eval_feedback (failure_type);
            -- 异步任务队列：知识流水线/图片生成等耗时操作持久化入队（PG 真相源）
            CREATE TABLE IF NOT EXISTS task_queue (
                id          BIGSERIAL PRIMARY KEY,
                task_type   TEXT NOT NULL,
                payload     JSONB DEFAULT '{}'::jsonb,
                status      TEXT DEFAULT 'pending',
                result      JSONB DEFAULT '{}'::jsonb,
                error       TEXT DEFAULT '',
                attempts    INT DEFAULT 0,
                owner       TEXT DEFAULT '',
                picked_by   TEXT DEFAULT '',
                created_at  TIMESTAMPTZ DEFAULT NOW(),
                started_at  TIMESTAMPTZ,
                finished_at TIMESTAMPTZ
            );
            CREATE INDEX IF NOT EXISTS idx_task_queue_status ON task_queue (status);
            CREATE INDEX IF NOT EXISTS idx_task_queue_type ON task_queue (task_type);
            CREATE INDEX IF NOT EXISTS idx_task_queue_created ON task_queue (created_at);
        """)
        conn.commit()


# --- 结构化日志冷层（ERROR/WARNING 审计） ---


def insert_log_event(
    level: str,
    module: str,
    event: str,
    request_id: str = "-",
    data: dict[str, Any] | None = None,
) -> None:
    """写入冷层日志（PG log_events），失败静默（日志不阻塞业务）。

    Args:
        level: WARNING / ERROR。
        module: 模块名。
        event: 事件名。
        request_id: 请求链路 ID。
        data: 附加结构化数据（JSONB）。
    """
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO log_events (level, module, event, request_id, data) "
                "VALUES (%s, %s, %s, %s, %s)",
                (level, module, event, request_id, psycopg2.extras.Json(data or {})),
            )
    except Exception:
        pass


def record_token_usage(
    *,
    request_id: str = "-",
    session_id: str = "",
    agent: str = "",
    model: str = "",
    source: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    latency_ms: int = 0,
    ok: bool = True,
    error: str = "",
    data: dict[str, Any] | None = None,
) -> None:
    """写入 Token 用量（观测底座），失败静默不阻塞业务。

    ok=False 时记录 LLM 调用失败原因（超时/重试耗尽/解析失败），
    供失败事件明细面板展示"每次失败的原因"。
    """
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO token_usage "
                "(request_id, session_id, agent, model, source, prompt_tokens, "
                " completion_tokens, total_tokens, latency_ms, ok, error, data) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    request_id,
                    session_id,
                    agent,
                    model,
                    source,
                    int(prompt_tokens or 0),
                    int(completion_tokens or 0),
                    int(total_tokens or 0),
                    int(latency_ms or 0),
                    bool(ok),
                    str(error)[:500],
                    psycopg2.extras.Json(data or {}),
                ),
            )
            conn.commit()
    except Exception:
        pass


def record_tool_call(
    *,
    request_id: str = "-",
    session_id: str = "",
    agent: str = "",
    tool_name: str,
    params: dict[str, Any] | None = None,
    ok: bool = True,
    error: str = "",
    latency_ms: int = 0,
    result_preview: str = "",
) -> None:
    """写入工具调用记录（观测底座），失败静默不阻塞业务。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO tool_calls "
                "(request_id, session_id, agent, tool_name, params, ok, error, "
                " latency_ms, result_preview) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    request_id,
                    session_id,
                    agent,
                    tool_name,
                    psycopg2.extras.Json(params or {}),
                    bool(ok),
                    str(error)[:500],
                    int(latency_ms or 0),
                    str(result_preview)[:500],
                ),
            )
            conn.commit()
    except Exception:
        pass


def token_usage_stats(
    days: int = 7,
    group_by: str = "day",
) -> dict[str, Any]:
    """Token 用量聚合：按天或按 Agent，含估算成本（元）。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT ts, COALESCE(NULLIF(agent,''),'unknown'), model, "
                "prompt_tokens, completion_tokens, total_tokens, latency_ms "
                "FROM token_usage WHERE ts >= NOW() - (%s || ' days')::INTERVAL",
                (str(days),),
            )
            raw = cur.fetchall()

        from agent_base.monitoring.costs import estimate_cost

        buckets: dict[str, dict[str, float]] = {}
        for ts, agent, model, pt, ct, tt, lat in raw:
            key = ts.strftime("%Y-%m-%d") if group_by == "day" else str(agent)
            b = buckets.setdefault(
                key,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "latency_sum": 0.0,
                    "cost": 0.0,
                },
            )
            b["calls"] += 1
            b["prompt_tokens"] += int(pt or 0)
            b["completion_tokens"] += int(ct or 0)
            b["total_tokens"] += int(tt or 0)
            b["latency_sum"] += int(lat or 0)
            b["cost"] += estimate_cost(str(model or ""), int(pt or 0), int(ct or 0))

        rows = [
            {
                "group": key,
                "calls": int(b["calls"]),
                "prompt_tokens": int(b["prompt_tokens"]),
                "completion_tokens": int(b["completion_tokens"]),
                "total_tokens": int(b["total_tokens"]),
                "avg_latency_ms": round(b["latency_sum"] / max(1, b["calls"]), 1),
                "cost": round(b["cost"], 4),
            }
            for key, b in buckets.items()
        ]
        if group_by == "agent":
            rows.sort(key=lambda r: r["total_tokens"], reverse=True)
        else:
            rows.sort(key=lambda r: r["group"])
        return {
            "group_by": group_by,
            "days": days,
            "rows": rows,
            "total_tokens": sum(r["total_tokens"] for r in rows),
            "total_cost": round(sum(r["cost"] for r in rows), 4),
        }
    except Exception:
        return {
            "group_by": group_by,
            "days": days,
            "rows": [],
            "total_tokens": 0,
            "total_cost": 0,
        }


def tool_calls_stats(days: int = 7) -> dict[str, Any]:
    """工具调用统计：按工具聚合调用数/成功率/平均耗时。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT tool_name, COUNT(*), "
                "SUM(CASE WHEN ok THEN 1 ELSE 0 END), "
                "AVG(latency_ms), SUM(CASE WHEN ok THEN 0 ELSE 1 END) "
                "FROM tool_calls WHERE ts >= NOW() - (%s || ' days')::INTERVAL "
                "GROUP BY tool_name ORDER BY COUNT(*) DESC",
                (str(days),),
            )
            rows = cur.fetchall()
            return {
                "days": days,
                "rows": [
                    {
                        "tool": r[0],
                        "calls": r[1],
                        "success": r[2] or 0,
                        "failed": r[4] or 0,
                        "success_rate": round((r[2] or 0) / max(1, r[1]), 4),
                        "avg_latency_ms": round(r[3] or 0, 1),
                    }
                    for r in rows
                ],
            }
    except Exception:
        return {"days": days, "rows": []}


def failure_stats(days: int = 7) -> dict[str, Any]:
    """失败统计：按模块聚合 ERROR 日志 + 评测失败归因。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT COALESCE(NULLIF(module,''),'unknown'), COUNT(*) "
                "FROM log_events WHERE level='ERROR' "
                "AND ts >= NOW() - (%s || ' days')::INTERVAL "
                "GROUP BY module ORDER BY COUNT(*) DESC",
                (str(days),),
            )
            by_module = [{"module": r[0], "count": r[1]} for r in cur.fetchall()]
            cur.execute(
                "SELECT COALESCE(NULLIF(failure_type,''),'unknown'), COUNT(*) "
                "FROM eval_feedback WHERE created_at >= NOW() - (%s || ' days')::INTERVAL "
                "GROUP BY failure_type ORDER BY COUNT(*) DESC",
                (str(days),),
            )
            by_type = [{"failure_type": r[0], "count": r[1]} for r in cur.fetchall()]
            return {
                "days": days,
                "by_module": by_module,
                "by_failure_type": by_type,
            }
    except Exception:
        return {"days": days, "by_module": [], "by_failure_type": []}


def recent_failure_events(days: int = 7, limit: int = 50) -> dict[str, Any]:
    """最近失败事件明细：统一 错误日志 / 工具调用 / LLM 调用 三类失败，携带每次失败的原因。

    生产监控口径：先落原始事件（原因 + 上下文 + 链路 ID），聚合统计只是派生品——
    面板因此能看到"第几次失败、原因是什么"，而不只是失败次数。
    """
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT ts, source, module, reason, request_id, data FROM ("
                " SELECT ts, 'error_log' AS source, "
                "   COALESCE(NULLIF(module,''),'unknown') AS module, "
                "   CASE WHEN data ? 'error' THEN data->>'error' ",
                "        WHEN data ? 'exc' THEN data->>'exc' ",
                "        WHEN data ? 'message' THEN data->>'message' ",
                "        WHEN data ? 'detail' THEN data->>'detail' ",
                "        ELSE event END AS reason, request_id, data "
                " FROM log_events WHERE level='ERROR' "
                "   AND ts >= NOW() - (%s || ' days')::INTERVAL "
                " UNION ALL "
                " SELECT ts, 'tool_call' AS source, tool_name AS module, "
                "   CASE WHEN NULLIF(error,'') IS NULL THEN '工具调用失败（未记录原因）' ELSE error END, ",
                "   request_id, jsonb_build_object('agent', agent, 'params', params, 'latency_ms', latency_ms) "
                " FROM tool_calls WHERE ok = FALSE "
                "   AND ts >= NOW() - (%s || ' days')::INTERVAL "
                " UNION ALL "
                " SELECT ts, 'llm_call' AS source, "
                "   COALESCE(NULLIF(agent,''), NULLIF(model,''), 'llm') AS module, ",
                "   CASE WHEN NULLIF(error,'') IS NULL THEN '模型调用失败（未记录原因）' ELSE error END, ",
                "   request_id, jsonb_build_object('model', model, 'agent', agent, 'latency_ms', latency_ms) "
                " FROM token_usage WHERE ok = FALSE "
                "   AND ts >= NOW() - (%s || ' days')::INTERVAL "
                ") f ORDER BY ts DESC LIMIT %s",
                (str(days), str(days), str(days), int(limit)),
            )
            rows = cur.fetchall()
            cur.execute(
                "SELECT (SELECT COUNT(*) FROM log_events WHERE level='ERROR' AND ts >= NOW() - (%s || ' days')::INTERVAL) "
                " + (SELECT COUNT(*) FROM tool_calls WHERE ok=FALSE AND ts >= NOW() - (%s || ' days')::INTERVAL) "
                " + (SELECT COUNT(*) FROM token_usage WHERE ok=FALSE AND ts >= NOW() - (%s || ' days')::INTERVAL)",
                (str(days), str(days), str(days)),
            )
            total = int((cur.fetchone() or [0])[0])
            return {
                "days": days,
                "total": total,
                "events": [
                    {
                        "ts": str(r[0]),
                        "source": r[1],
                        "module": r[2],
                        "reason": (r[3] or "")[:2000],
                        "request_id": r[4],
                        "data": r[5] or {},
                    }
                    for r in rows
                ],
            }
    except Exception:
        return {"days": days, "total": 0, "events": []}


def upsert_eval_feedback(
    *,
    case_id: int = 0,
    run_id: int = 0,
    question: str = "",
    failure_type: str = "",
    detail: str = "",
    status: str = "pending",
) -> int:
    """写入/更新评测反馈（失败归因 + 数据飞轮状态）。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO eval_feedback "
                "(case_id, run_id, question, failure_type, detail, status) "
                "VALUES (%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (id) DO NOTHING RETURNING id",
                (int(case_id), int(run_id), question, failure_type, detail, status),
            )
            row = cur.fetchone()
            conn.commit()
            return int(row[0]) if row else 0
    except Exception:
        return 0


def eval_feedback_list(status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """评测反馈列表（数据飞轮状态流转可见）。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            if status:
                cur.execute(
                    "SELECT id, case_id, run_id, question, failure_type, detail, status, "
                    "regression, created_at, updated_at "
                    "FROM eval_feedback WHERE status=%s "
                    "ORDER BY id DESC LIMIT %s",
                    (status, int(limit)),
                )
            else:
                cur.execute(
                    "SELECT id, case_id, run_id, question, failure_type, detail, status, "
                    "regression, created_at, updated_at "
                    "FROM eval_feedback ORDER BY id DESC LIMIT %s",
                    (int(limit),),
                )
            rows = cur.fetchall()
            return [
                {
                    "id": r[0],
                    "case_id": r[1],
                    "run_id": r[2],
                    "question": r[3],
                    "failure_type": r[4],
                    "detail": r[5],
                    "status": r[6],
                    "regression": r[7],
                    "created_at": str(r[8]),
                    "updated_at": str(r[9]),
                }
                for r in rows
            ]
    except Exception:
        return []


def update_eval_feedback_status(feedback_id: int, status: str, regression: float = 0) -> bool:
    """更新评测反馈状态（数据飞轮：待补料→已补充→已回归）。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE eval_feedback SET status=%s, regression=%s, "
                "updated_at=NOW() WHERE id=%s",
                (status, float(regression), int(feedback_id)),
            )
            conn.commit()
            return cur.rowcount > 0
    except Exception:
        return False


# --- 异步任务队列（PG 持久化：入队/认领/完成/回收） ---

_TASK_COLUMNS = (
    "id, task_type, payload, status, result, error, attempts, owner, "
    "picked_by, created_at, started_at, finished_at"
)


def _task_row_to_dict(row: Any) -> dict[str, Any]:
    """任务行 → dict（JSONB 字段反序列化）。"""
    return {
        "id": row[0],
        "task_type": row[1],
        "payload": row[2] or {},
        "status": row[3],
        "result": row[4] or {},
        "error": row[5] or "",
        "attempts": int(row[6] or 0),
        "owner": row[7] or "",
        "picked_by": row[8] or "",
        "created_at": str(row[9]),
        "started_at": str(row[10]) if row[10] else None,
        "finished_at": str(row[11]) if row[11] else None,
    }


def task_enqueue(task_type: str, payload: dict[str, Any] | None = None, owner: str = "") -> int:
    """任务入队，返回任务 id（0 表示入队失败）。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO task_queue (task_type, payload, owner) "
                "VALUES (%s, %s::jsonb, %s) RETURNING id",
                (task_type, psycopg2.extras.Json(payload or {}), owner),
            )
            conn.commit()
            row = cur.fetchone()
            return int(row[0]) if row else 0
    except Exception:
        return 0


def task_claim_next(worker: str, task_types: list[str] | None = None) -> dict[str, Any] | None:
    """原子认领队首 pending 任务（FOR UPDATE SKIP LOCKED，多 worker 安全）。

    Args:
        worker: worker 标识（认领者）。
        task_types: 限定认领的任务类型；None 表示不限。

    Returns:
        任务 dict（已置 running），无可认领任务返回 None。
    """
    try:
        with _conn() as conn:
            cur = conn.cursor()
            if task_types:
                cur.execute(
                    "UPDATE task_queue SET status='running', started_at=NOW(), "
                    "picked_by=%s, attempts=attempts+1 "
                    "WHERE id = ("
                    "  SELECT id FROM task_queue WHERE status='pending' "
                    "  AND task_type = ANY(%s) "
                    "  ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED"
                    ") RETURNING " + _TASK_COLUMNS,
                    (worker, list(task_types)),
                )
            else:
                cur.execute(
                    "UPDATE task_queue SET status='running', started_at=NOW(), "
                    "picked_by=%s, attempts=attempts+1 "
                    "WHERE id = ("
                    "  SELECT id FROM task_queue WHERE status='pending' "
                    "  ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED"
                    ") RETURNING " + _TASK_COLUMNS,
                    (worker,),
                )
            conn.commit()
            row = cur.fetchone()
            return _task_row_to_dict(row) if row else None
    except Exception:
        return None


def task_finish(
    task_id: int,
    status: str = "done",
    result: dict[str, Any] | None = None,
    error: str = "",
) -> bool:
    """任务收尾：done / failed（写 result/error/finished_at）。"""
    if status not in {"done", "failed"}:
        status = "failed"
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE task_queue SET status=%s, result=%s::jsonb, error=%s, "
                "finished_at=NOW() WHERE id=%s",
                (status, psycopg2.extras.Json(result or {}), error[:500], int(task_id)),
            )
            conn.commit()
            return cur.rowcount > 0
    except Exception:
        return False


def task_get(task_id: int) -> dict[str, Any] | None:
    """读取单个任务。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT " + _TASK_COLUMNS + " FROM task_queue WHERE id=%s", (int(task_id),))
            row = cur.fetchone()
            return _task_row_to_dict(row) if row else None
    except Exception:
        return None


def task_list(status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """任务列表（默认全部状态，按 id 倒序）。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            if status:
                cur.execute(
                    "SELECT " + _TASK_COLUMNS + " FROM task_queue WHERE status=%s "
                    "ORDER BY id DESC LIMIT %s",
                    (status, int(limit)),
                )
            else:
                cur.execute(
                    "SELECT " + _TASK_COLUMNS + " FROM task_queue "
                    "ORDER BY id DESC LIMIT %s",
                    (int(limit),),
                )
            return [_task_row_to_dict(r) for r in cur.fetchall()]
    except Exception:
        return []


def task_cancel(task_id: int) -> bool:
    """停止 pending 任务；running 任务标记为 failed 并记录停止原因。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE task_queue SET status=CASE WHEN status='pending' THEN 'cancelled' ELSE 'failed' END, "
                "error=COALESCE(NULLIF(error,''), '已手动停止'), finished_at=NOW() "
                "WHERE id=%s AND status IN ('pending','running')",
                (int(task_id),),
            )
            return cur.rowcount > 0
    except Exception:
        return False


def task_delete(task_id: int) -> bool:
    """删除任务记录。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM task_queue WHERE id=%s", (int(task_id),))
            return cur.rowcount > 0
    except Exception:
        return False


def task_reap_stale(older_than_seconds: int = 300) -> int:
    """回收僵死任务：running 超过 N 秒的重新置为 pending（worker 崩溃自愈）。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE task_queue SET status='pending', picked_by='' "
                "WHERE status='running' AND started_at < NOW() - (%s || ' seconds')::INTERVAL",
                (int(older_than_seconds),),
            )
            conn.commit()
            return int(cur.rowcount or 0)
    except Exception:
        return 0


def purge_expired_logs(retention_days: int = 90) -> int:
    """删除超期冷层日志，返回删除条数。

    Args:
        retention_days: 保留天数（默认 90）。

    Returns:
        删除的日志条数；失败返回 0。
    """
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM log_events WHERE ts < NOW() - (%s || ' days')::INTERVAL",
                (str(retention_days),),
            )
            return cur.rowcount
    except Exception:
        return 0


# --- 文档版本管理 ---

def doc_upsert(
    doc_id: str,
    chunk_ids: list[str],
    metadata: dict[str, Any] | None = None,
    content: str = "",
) -> int:
    """文档写入：版本自增，存 content + chunk_ids。

    Args:
        doc_id: 文档标识。
        chunk_ids: 向量库中的 chunk ID 列表。
        metadata: 可选元数据字典。
        content: 文档全文（P16：PG 为真相源）。

    Returns:
        新版本号。
    """
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COALESCE(MAX(version),0) FROM documents WHERE doc_id=%s", (doc_id,))
        new_version = cur.fetchone()[0] + 1
        cur.execute("""
            INSERT INTO documents (doc_id, version, content, chunk_ids, status, metadata)
            VALUES (%s,%s,%s,%s,'active',%s)
        """, (doc_id, new_version, content, chunk_ids, psycopg2.extras.Json(metadata or {})))
# 旧版本标记为 archived
        cur.execute("UPDATE documents SET status='archived' WHERE doc_id=%s AND version<%s", (doc_id, new_version))
        # 版本保留策略（默认 10）：超过上限软删最旧版本
        cur.execute(
            "SELECT version FROM documents WHERE doc_id=%s AND status <> 'deleted' ORDER BY version",
            (doc_id,),
        )
        all_versions = [r[0] for r in cur.fetchall()]
        if len(all_versions) > 10:
            for v in all_versions[: len(all_versions) - 10]:
                cur.execute(
                    "UPDATE documents SET status='deleted', updated_at=NOW() "
                    "WHERE doc_id=%s AND version=%s",
                    (doc_id, v),
                )
        return new_version


def doc_delete(doc_id: str) -> None:
    """软删除：全部版本置 status='deleted' + deleted_at（回收站）。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE documents SET status='deleted', updated_at=NOW(), deleted_at=NOW() WHERE doc_id=%s",
            (doc_id,),
        )


def doc_set_status(doc_id: str, status: str) -> bool:
    """切换文档最新版本状态（archive → archived / activate → active）。

    只改最新版本（doc_upsert 会把旧版本自动标 archived，文档管理按最新版本状态展示）。

    Args:
        doc_id: Document ID.
        status: 目标状态（active / archived）。

    Returns:
        True 如果最新版本存在并被更新。
    """
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE documents SET status=%s, updated_at=NOW() "
            "WHERE doc_id=%s AND version=(SELECT MAX(version) FROM documents WHERE doc_id=%s)",
            (status, doc_id, doc_id),
        )
        return cur.rowcount > 0


def doc_delete_version(doc_id: str, version: int) -> bool:
    """软删单个历史版本（active 版本不可删）。

    Args:
        doc_id: Document ID.
        version: 目标版本号。

    Returns:
        True 删除成功；False 版本不存在或为 active（不可删）。
    """
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT status FROM documents WHERE doc_id=%s AND version=%s", (doc_id, version))
        row = cur.fetchone()
        if not row:
            return False
        if row[0] == "active":
            return False
        cur.execute(
            "UPDATE documents SET status='deleted', updated_at=NOW(), deleted_at=NOW() "
            "WHERE doc_id=%s AND version=%s",
            (doc_id, version),
        )
        return True


def purge_deleted_documents(days: int = 30) -> dict[str, int]:
    """物理清理已删除（软删）超过 N 天的记录（生产审计期满后定时执行）。

    同时清理对应 document_strategy 已删除标签，以及超过保留期的历史版本。

    Args:
        days: 删除超过多少天的记录（默认 30）。

    Returns:
        {"documents": 清理行数, "strategy": 清理标签数}。
    """
    from datetime import datetime, timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result: dict[str, int] = {"documents": 0, "strategy": 0}
    with _conn() as conn:
        cur = conn.cursor()
        # 先找"所有版本都 deleted"的 doc_id（删 documents 前，避免误删 pending 标签）
        cur.execute(
            "SELECT doc_id FROM documents GROUP BY doc_id "
            "HAVING COUNT(*) FILTER (WHERE status <> 'deleted') = 0"
        )
        fully_deleted = [r[0] for r in cur.fetchall()]
        # 物理删除软删超过保留期的 documents 行
        cur.execute(
            "DELETE FROM documents WHERE status='deleted' AND COALESCE(deleted_at, updated_at) < %s",
            (cutoff,),
        )
        result["documents"] = cur.rowcount
        # 清理已物理删除文档的孤儿标签（保留 pending 等未入库文档的标签）
        if fully_deleted:
            placeholders = ",".join(["%s"] * len(fully_deleted))
            cur.execute(
                f"DELETE FROM document_strategy WHERE doc_id IN ({placeholders})",
                fully_deleted,
            )
            result["strategy"] = cur.rowcount
    return result


def doc_trash_list(retention_days: int = 30) -> list[dict[str, Any]]:
    """列出软删除文档（回收站），按 doc_id 去重（取最新删除版本）。"""
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT doc_id, version, status, deleted_at, metadata, chunk_ids "
            "FROM documents WHERE status='deleted' ORDER BY doc_id, version DESC"
        )
        rows = cur.fetchall()

    best: dict[str, dict[str, Any]] = {}
    version_counts: dict[str, int] = {}
    for doc_id, version, status, deleted_at, metadata, chunk_ids in rows:
        version_counts[doc_id] = version_counts.get(doc_id, 0) + 1
        if doc_id in best:
            continue
        remaining_days: int | None = None
        if deleted_at is not None:
            expire_at = deleted_at + timedelta(days=retention_days)
            remaining_secs = (expire_at - now).total_seconds()
            remaining_days = max(0, int(remaining_secs // 86400) + (1 if remaining_secs % 86400 > 0 else 0))
        best[doc_id] = {
            "doc_id": doc_id,
            "version": version,
            "status": status,
            "deleted_at": deleted_at.isoformat() if deleted_at else None,
            "remaining_days": remaining_days,
            "version_count": version_counts[doc_id],
            "chunk_count": len(chunk_ids or []),
            "metadata": metadata or {},
        }
    result = sorted(best.values(), key=lambda x: x["deleted_at"] or "", reverse=True)
    return result


def doc_recover(doc_id: str) -> dict[str, Any] | None:
    """恢复回收站文档：最新版本转 active，旧版本转 archived，清空 deleted_at。

    Returns:
        供重建索引用的 dict（doc_id/version/content/metadata/chunk_ids）；不在回收站返回 None。
    """
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT version, content, metadata, chunk_ids FROM documents "
            "WHERE doc_id=%s AND status='deleted' ORDER BY version DESC LIMIT 1",
            (doc_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        version, content, metadata, chunk_ids = row
        cur.execute(
            "UPDATE documents SET status='active', deleted_at=NULL, updated_at=NOW() "
            "WHERE doc_id=%s AND version=%s",
            (doc_id, version),
        )
        cur.execute(
            "UPDATE documents SET status='archived' "
            "WHERE doc_id=%s AND version<>%s AND status='deleted'",
            (doc_id, version),
        )
        return {
            "doc_id": doc_id,
            "version": version,
            "content": content or "",
            "metadata": metadata or {},
            "chunk_ids": chunk_ids or [],
        }


def doc_set_chunk_ids(doc_id: str, version: int, chunk_ids: list[str]) -> bool:
    """更新指定版本的 chunk_ids（回收站恢复重建索引用）。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE documents SET chunk_ids=%s, updated_at=NOW() "
            "WHERE doc_id=%s AND version=%s",
            (chunk_ids, doc_id, version),
        )
        return cur.rowcount > 0


def doc_purge(doc_id: str) -> bool:
    """物理删除文档全部版本及策略标签（永久删除）。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM documents WHERE doc_id=%s", (doc_id,))
        deleted = cur.rowcount
        cur.execute("DELETE FROM document_strategy WHERE doc_id=%s", (doc_id,))
        try:
            cur.execute("DELETE FROM document_staging WHERE doc_id=%s", (doc_id,))
        except Exception:
            pass
        return deleted > 0


def doc_versions(doc_id: str) -> list[dict[str, Any]]:
    """获取文档全部版本（含 content），排除软删除版本。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT doc_id,version,content,chunk_ids,status,created_at FROM documents "
            "WHERE doc_id=%s AND status <> 'deleted' ORDER BY version DESC",
            (doc_id,))
        return [{"doc_id":r[0],"version":r[1],"content":r[2] or "","chunk_ids":r[3],"status":r[4],"created_at":r[5].isoformat() if r[5] else None} for r in cur.fetchall()]


def doc_list(status: str | None = None) -> list[dict[str, Any]]:
    """列出已入库文档（documents 表，非 deleted），按 doc_id 去重取当前版本。

    P27 职责边界：文档管理 = 管理已入库文档（RAG 数据核心），
    只展示 active/archived，不混入审核台状态（pending/returned）。

    Args:
        status: 可选过滤（active / archived）；None 返回全部非 deleted。

    Returns:
        [{doc_id, version, status, chunk_ids, created_at, updated_at}]。
    """
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT doc_id, version, status, chunk_ids, created_at, updated_at, metadata "
            "FROM documents WHERE status <> 'deleted'"
        )
        rows = cur.fetchall()
    # 按 doc_id 去重：保留最新版本（version 最大的行）
    best: dict[str, dict[str, Any]] = {}
    for r in rows:
        doc_id = r[0]
        version = r[1]
        if doc_id in best and version <= best[doc_id]["version"]:
            continue
        best[doc_id] = {
            "doc_id": doc_id,
            "version": version,
            "status": r[2],
            "chunk_ids": r[3] or [],
            "created_at": r[4].isoformat() if r[4] else None,
            "updated_at": r[5].isoformat() if r[5] else None,
            "metadata": r[6] or {},
        }
    result = sorted(best.values(), key=lambda x: x["updated_at"] or "", reverse=True)
    # P30: 先按 doc_id 取最新版本，再按最新版本 status 过滤——
    # 避免旧版本（doc_upsert 自动标 archived）在 active/archived 两个查询里重复出现
    if status:
        result = [d for d in result if d.get("status") == status]
    return result


def doc_restore_version(doc_id: str, version: int) -> dict[str, Any] | None:
    """恢复指定版本：返回 content + chunk_ids + metadata 供重建索引。

    Args:
        doc_id: 文档标识。
        version: 目标版本号。

    Returns:
        含 content + chunk_ids + metadata 的字典；版本不存在返回 None。
    """
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT content, chunk_ids, metadata FROM documents WHERE doc_id=%s AND version=%s",
            (doc_id, version),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"content": row[0] or "", "chunk_ids": row[1] or [], "metadata": row[2] or {}}


def doc_switch_version(
    doc_id: str,
    version: int,
    chunk_ids: list[str],
) -> None:
    """切换 active 版本指针（回滚语义，不生成新版本）。

    目标版本置 active 并更新 chunk_ids（与重建后的 Qdrant 对齐），
    其余非 deleted 版本置 archived。

    Args:
        doc_id: Document ID.
        version: 目标版本号。
        chunk_ids: 重建后的 chunk_ids（与 Qdrant 一致）。
    """
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE documents SET status='archived' "
            "WHERE doc_id=%s AND status <> 'deleted'",
            (doc_id,),
        )
        cur.execute(
            "UPDATE documents SET status='active', chunk_ids=%s "
            "WHERE doc_id=%s AND version=%s",
            (chunk_ids, doc_id, version),
        )


def _migrate_documents_add_content():
    """幂等迁移：缺少 content 列时补建（v0.24.0）。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='documents' AND column_name='content'"
        )
        if not cur.fetchone():
            cur.execute("ALTER TABLE documents ADD COLUMN content TEXT DEFAULT ''")


def _migrate_documents_add_deleted_at():
    """幂等迁移：缺少 deleted_at 列时补建（v0.46 回收站）。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ")
            # 历史遗留软删行补写 deleted_at（用 updated_at 兜底），保证保留期/剩余天数可计算
            cur.execute(
                "UPDATE documents SET deleted_at = updated_at "
                "WHERE status='deleted' AND deleted_at IS NULL"
            )
    except Exception:
        pass


def _migrate_chat_messages_add_sources():
    """幂等迁移：chat_messages 缺少 sources 列时补建（客服端 AI 召回依据）。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS sources JSONB"
            )
    except Exception:
        pass


def _migrate_eval_columns() -> None:
    """幂等迁移：评测表补 LCEL 指标列（faithfulness/relevancy）。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS faithfulness_acc REAL DEFAULT 0")
            cur.execute("ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS relevancy_acc REAL DEFAULT 0")
            cur.execute("ALTER TABLE eval_cases ADD COLUMN IF NOT EXISTS faithfulness REAL DEFAULT 0")
            cur.execute("ALTER TABLE eval_cases ADD COLUMN IF NOT EXISTS relevancy REAL DEFAULT 0")
    except Exception:
        pass


def _migrate_eval_ragas_columns() -> None:
    """幂等迁移：评测表补 RAGAS 四指标列（faithfulness/relevancy/context_precision/context_recall）。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            for col in (
                "ragas_faithfulness_acc",
                "ragas_relevancy_acc",
                "ragas_context_precision_acc",
                "ragas_context_recall_acc",
            ):
                cur.execute(f"ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS {col} REAL DEFAULT 0")
            for col in (
                "ragas_faithfulness",
                "ragas_relevancy",
                "ragas_context_precision",
                "ragas_context_recall",
            ):
                cur.execute(f"ALTER TABLE eval_cases ADD COLUMN IF NOT EXISTS {col} REAL DEFAULT 0")
    except Exception:
        pass


def _migrate_token_usage_failure_columns() -> None:
    """幂等迁移：token_usage 补 ok/error 列（记录 LLM 调用失败原因，支撑失败事件明细面板）。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("ALTER TABLE token_usage ADD COLUMN IF NOT EXISTS ok BOOLEAN DEFAULT TRUE")
            cur.execute("ALTER TABLE token_usage ADD COLUMN IF NOT EXISTS error TEXT DEFAULT ''")
    except Exception:
        pass


# --- 种子数据 ---

# ── P12-02: 意图规则库 CRUD ──


def intent_seed_from_yaml(yaml_path: str = "configs/domain/ecommerce.yaml", force: bool = True) -> int:
    """从 YAML 种子导入意图到 PG（首次启动或重建时使用）。

    默认 force=True：每次启动都会将 YAML 变更写入 PG（版本化），
    确保 YAML 是意图管理的真相源，PG 作为运行时缓存。

    Args:
        yaml_path: 领域配置 YAML 路径。
        force: True 时总是覆盖更新；False 时仅导入不存在的意图。

    Returns:
        新导入/更新的意图数量。
    """
    import yaml as _yaml
    from pathlib import Path as _Path

    yp = _Path(yaml_path)
    if not yp.exists():
        return 0

    data = _yaml.safe_load(yp.read_text(encoding="utf-8"))
    intents_data = data.get("intents", {})
    count = 0
    for name, cfg in intents_data.items():
        if not force:
            existing = intent_get(name)
            if existing and existing.get("status") == "active":
                continue
        keywords = cfg.get("keywords", [])
        sections = cfg.get("sections", [])
        examples = cfg.get("examples", [])
        priority = float(cfg.get("priority", 1.0))
        intent_upsert(name, keywords=keywords, sections=sections,
                      examples=examples, priority=priority)
        count += 1
    return count


def intent_upsert(
    intent: str,
    keywords: list[str] | None = None,
    sections: list[str] | None = None,
    examples: list[str] | None = None,
    priority: float = 1.0,
) -> int:
    """更新意图规则，version+1 保留历史。

    Args:
        intent: 意图名。
        keywords: 关键词列表。
        sections: 目标章节列表。
        examples: few-shot 示例列表。
        priority: 优先级。

    Returns:
        新版本号。
    """
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COALESCE(MAX(version),0) FROM intent_rules WHERE intent=%s", (intent,))
        new_version = cur.fetchone()[0] + 1
        cur.execute("""
            INSERT INTO intent_rules (intent, keywords, sections, examples, priority, version, status)
            VALUES (%s,%s,%s,%s,%s,%s,'active')
        """, (intent, keywords or [],
              sections or [], examples or [], priority, new_version))
        cur.execute(
            "UPDATE intent_rules SET status='archived', updated_at=NOW() WHERE intent=%s AND version<%s",
            (intent, new_version))
        return new_version


def intent_list(include_archived: bool = False) -> list[dict[str, Any]]:
    """列出所有意图的最新 active 版本（含字段详情）。

    Args:
        include_archived: 是否包含已归档版本。

    Returns:
        意图列表。
    """
    with _conn() as conn:
        cur = conn.cursor()
        if include_archived:
            cur.execute(
                "SELECT intent,keywords,sections,examples,priority,version,status,updated_at FROM intent_rules ORDER BY intent,version DESC")
        else:
            cur.execute(
                "SELECT DISTINCT ON (intent) intent,keywords,sections,examples,priority,version,status,updated_at FROM intent_rules WHERE status='active' ORDER BY intent,version DESC")
        return [
            {"intent": r[0], "keywords": r[1], "sections": r[2],
             "examples": r[3], "priority": float(r[4]),
             "version": int(r[5]), "status": r[6],
             "updated_at": r[7].isoformat() if r[7] else None}
            for r in cur.fetchall()
        ]


def intent_get(intent: str) -> dict[str, Any] | None:
    """获取单个意图的最新 active 版本。

    Args:
        intent: 意图名。

    Returns:
        意图 dict，不存在时返回 None。
    """
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT intent,keywords,sections,examples,priority,version,status,updated_at FROM intent_rules WHERE intent=%s AND status='active' ORDER BY version DESC LIMIT 1",
            (intent,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "intent": row[0], "keywords": row[1], "sections": row[2],
            "examples": row[3], "priority": float(row[4]),
            "version": int(row[5]), "status": row[6],
            "updated_at": row[7].isoformat() if row[7] else None,
        }


def intent_add_examples(intent: str, new_examples: list[str]) -> int:
    """追加 few-shot 示例（version+1）。

    Args:
        intent: 意图名。
        new_examples: 新增示例列表。

    Returns:
        新版本号。
    """
    existing = intent_get(intent)
    if not existing:
        return 0
    merged = list(dict.fromkeys(list(existing.get("examples", [])) + new_examples))
    return intent_upsert(
        intent,
        keywords=existing.get("keywords", []),
        sections=existing.get("sections", []),
        examples=merged,
        priority=existing.get("priority", 1.0),
    )


def intent_restore(intent: str, version: int) -> dict[str, Any] | None:
    """回滚意图到指定版本（把该版本激活，其余版本标记 archived）。

    Args:
        intent: 意图名。
        version: 目标版本号。

    Returns:
        回滚后的意图 dict，版本不存在时返回 None。
    """
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT intent,keywords,sections,examples,priority FROM intent_rules WHERE intent=%s AND version=%s",
            (intent, version))
        row = cur.fetchone()
        if not row:
            return None
        # 全版本标记 archived
        cur.execute("UPDATE intent_rules SET status='archived', updated_at=NOW() WHERE intent=%s", (intent,))
        # 目标版本激活
        cur.execute("UPDATE intent_rules SET status='active', updated_at=NOW() WHERE intent=%s AND version=%s", (intent, version))
        return {
            "intent": row[0], "keywords": row[1], "sections": row[2],
            "examples": row[3], "priority": float(row[4]),
            "version": version, "status": "active",
        }


def intent_versions(intent: str) -> list[dict[str, Any]]:
    """列出意图的所有版本历史。

    Args:
        intent: 意图名。

    Returns:
        版本列表（按 version DESC）。
    """
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT intent,version,keywords,sections,examples,priority,status,updated_at FROM intent_rules WHERE intent=%s ORDER BY version DESC",
            (intent,))
        return [
            {"intent": r[0], "version": int(r[1]), "keywords": r[2],
             "sections": r[3], "examples": r[4], "priority": float(r[5]),
             "status": r[6], "updated_at": r[7].isoformat() if r[7] else None}
            for r in cur.fetchall()
        ]


def intent_to_domain_dict() -> dict[str, dict[str, Any]]:
    """将 PG 中的 active 意图转成 domain adapter 兼容的 intents dict。

    优先读 PG（生产运维即时生效），PG 为空时返回 None 由调用方降级 YAML。

    Returns:
        {intent: {keywords, sections, examples, priority}} 或空 dict。
    """
    rows = intent_list(include_archived=False)
    if not rows:
        return {}
    return {
        r["intent"]: {
            "keywords": r["keywords"],
            "sections": r["sections"],
            "examples": r["examples"],
            "priority": r["priority"],
        }
        for r in rows
    }


# ── P14-01: 文档打标策略 CRUD ──


def _migrate_document_strategy_p19() -> None:
    """P19: 打标表升级为审核状态机（幂等迁移，列不存在才加）。"""
    with _conn() as conn:
        cur = conn.cursor()
        for column, ddl in [
            ("review_round", "INT DEFAULT 1"),
            ("first_review", "JSONB DEFAULT '{}'::jsonb"),
            ("fine_reviewer", "TEXT DEFAULT ''"),
            ("reject_reason", "TEXT DEFAULT ''"),
            ("reviewed_at", "TIMESTAMPTZ"),
        ]:
            cur.execute(
                f"ALTER TABLE document_strategy ADD COLUMN IF NOT EXISTS {column} {ddl}"
            )


def strategy_upsert(
    doc_id: str,
    doc_type: str = "",
    strategy: list[str] | None = None,
    reviewer: str = "",
    status: str = "pending_fine_review",
    review_round: int = 1,
    first_review: dict[str, Any] | None = None,
    reject_reason: str = "",
    reviewed_at: str | None = None,
) -> None:
    """Insert or update a document strategy tag (P19 review state machine).

    Args:
        doc_id: Document ID.
        doc_type: One of product_detail/faq/category_guide/metadata_doc.
        strategy: List of indexing strategies.
        reviewer: Human reviewer identifier.
        status: pending_fine_review | approved | returned.
        review_round: 打回重审轮次（1 起）。
        first_review: agent 初审快照（type/confidence/reasoning）。
        reject_reason: 打回理由。
        reviewed_at: 精审时间。
    """
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO document_strategy
                (doc_id, doc_type, strategy, reviewer, status, review_round,
                 first_review, reject_reason, reviewed_at, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
            ON CONFLICT (doc_id) DO UPDATE SET
                doc_type=EXCLUDED.doc_type, strategy=EXCLUDED.strategy,
                reviewer=EXCLUDED.reviewer, status=EXCLUDED.status,
                review_round=EXCLUDED.review_round,
                first_review=EXCLUDED.first_review,
                reject_reason=EXCLUDED.reject_reason,
                reviewed_at=EXCLUDED.reviewed_at, updated_at=NOW()
        """, (
            doc_id, doc_type, strategy or [], reviewer, status,
            review_round,
            json.dumps(first_review or {}, ensure_ascii=False),
            reject_reason, reviewed_at,
        ))


def strategy_list(doc_id: str | None = None) -> list[dict[str, Any]]:
    """列出文档策略标签。

    Args:
        doc_id: 可选，按文档 ID 过滤。

    Returns:
        策略记录列表。
    """
    with _conn() as conn:
        cur = conn.cursor()
        if doc_id:
            cur.execute(
                "SELECT doc_id,doc_type,strategy,reviewer,status,review_round,"
                "first_review,reject_reason,reviewed_at,updated_at "
                "FROM document_strategy WHERE doc_id=%s",
                (doc_id,))
        else:
            cur.execute(
                "SELECT doc_id,doc_type,strategy,reviewer,status,review_round,"
                "first_review,reject_reason,reviewed_at,updated_at "
                "FROM document_strategy ORDER BY updated_at DESC")
        return [
            {"doc_id": r[0], "doc_type": r[1], "strategy": r[2],
            "reviewer": r[3], "status": r[4],
             "review_round": r[5],
             "first_review": (
                 json.loads(r[6]) if isinstance(r[6], str) else (r[6] or {})
             ),
             "reject_reason": r[7] or "",
             "reviewed_at": r[8].isoformat() if r[8] else None,
             "updated_at": r[9].isoformat() if r[9] else None}
            for r in cur.fetchall()
        ]


def strategy_get(doc_id: str) -> dict[str, Any] | None:
    """获取单个策略标签（不存在返回 None）。

    Args:
        doc_id: 文档标识。

    Returns:
        策略记录字典或 None。
    """
    rows = strategy_list(doc_id=doc_id)
    return rows[0] if rows else None


def list_document_ids() -> list[str]:
    """列出 documents 表中全部 doc_id（旧版标签种子用）。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT doc_id FROM documents")
        return [r[0] for r in cur.fetchall()]


def strategy_delete(doc_id: str) -> None:
    """删除文档策略标签。

    Args:
        doc_id: 文档标识。
    """
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM document_strategy WHERE doc_id=%s", (doc_id,))


# ── P20: 知识入库暂存（上传=暂存+自动预审，approved 才真正入库） ──


def staging_upsert(
    doc_id: str,
    content: str,
    filename: str = "",
    category: str = "",
    status: str = "pending",
    review_round: int = 1,
    first_review: dict[str, Any] | None = None,
    reject_reason: str = "",
) -> None:
    """插入或更新暂存文档（P20 上传管道）。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO document_staging
                (doc_id, filename, content, category, status, review_round,
                 first_review, reject_reason, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NOW())
            ON CONFLICT (doc_id) DO UPDATE SET
                filename=EXCLUDED.filename, content=EXCLUDED.content,
                category=EXCLUDED.category, status=EXCLUDED.status,
                review_round=EXCLUDED.review_round,
                first_review=EXCLUDED.first_review,
                reject_reason=EXCLUDED.reject_reason, updated_at=NOW()
        """, (doc_id, filename, content, category, status, review_round,
              json.dumps(first_review or {}, ensure_ascii=False), reject_reason))


def staging_get(doc_id: str) -> dict[str, Any] | None:
    """按 doc_id 获取暂存文档。"""
    rows = staging_list(doc_id=doc_id)
    return rows[0] if rows else None


def staging_list(
    doc_id: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """列出暂存文档（可选 doc_id / status 过滤）。"""
    with _conn() as conn:
        cur = conn.cursor()
        sql = (
            "SELECT doc_id,filename,content,category,status,review_round,"
            "first_review,reject_reason,created_at,updated_at "
            "FROM document_staging"
        )
        conds: list[str] = []
        params: list[Any] = []
        if doc_id:
            conds.append("doc_id=%s")
            params.append(doc_id)
        if status:
            conds.append("status=%s")
            params.append(status)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY updated_at DESC"
        cur.execute(sql, params)
        return [
            {"doc_id": r[0], "filename": r[1], "content": r[2], "category": r[3],
             "status": r[4], "review_round": r[5],
             "first_review": (
                 json.loads(r[6]) if isinstance(r[6], str) else (r[6] or {})
             ),
             "reject_reason": r[7] or "",
             "created_at": r[8].isoformat() if r[8] else None,
             "updated_at": r[9].isoformat() if r[9] else None}
            for r in cur.fetchall()
        ]


def staging_delete(doc_id: str) -> None:
    """删除暂存文档。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM document_staging WHERE doc_id=%s", (doc_id,))


def staging_find_by_content(content: str) -> dict[str, Any] | None:
    """按内容精确查找暂存文档（去重辅助）。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT doc_id FROM document_staging WHERE content=%s "
            "ORDER BY updated_at DESC LIMIT 1",
            (content,),
        )
        row = cur.fetchone()
        return staging_get(row[0]) if row else None


def staging_find_by_filename(filename: str) -> dict[str, Any] | None:
    """按文件名查找暂存文档（同文件重复上传辅助）。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT doc_id FROM document_staging WHERE filename=%s "
            "ORDER BY updated_at DESC LIMIT 1",
            (filename,),
        )
        row = cur.fetchone()
        return staging_get(row[0]) if row else None


# ── P20: 对话消息（流式多轮上下文 + 会话列表） ──


def handoff_trigger(session_id: str, reason: str = "") -> dict[str, Any]:
    """触发转人工：创建或刷新 pending 记录；active 会话保持 active（不降级）。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_handoffs (session_id, status, reason, created_at, last_active_at) "
            "VALUES (%s, 'pending', %s, NOW(), NOW()) "
            "ON CONFLICT (session_id) DO UPDATE SET "
            "status='pending', reason=EXCLUDED.reason, last_active_at=NOW(), resolved_at=NULL, "
            "rating=0, rating_comment='', rated_at=NULL "
            "WHERE chat_handoffs.status <> 'active'",
            (session_id, reason),
        )
        # BUG-17: 返回 DB 实际最新状态（active 会话重复触发时保持 active）
        cur.execute("SELECT status FROM chat_handoffs WHERE session_id=%s", (session_id,))
        row = cur.fetchone()
    status = row[0] if row else "pending"
    return {"session_id": session_id, "status": status}


def handoff_check(
    session_id: str,
    pending_timeout: int = 900,
    idle_timeout: int = 1200,
) -> dict[str, Any] | None:
    """查询会话转人工状态（含超时惰性流转）；None = 无转人工记录。"""
    from datetime import datetime, timezone

    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT session_id, status, reason, created_at, last_active_at, agent_name "
            "FROM chat_handoffs WHERE session_id=%s",
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        status = row[1]
        last_active = row[4]
        now = datetime.now(timezone.utc)
        if status == "pending" and last_active is not None:
            if (now - last_active).total_seconds() > pending_timeout:
                cur.execute(
                    "UPDATE chat_handoffs SET status='expired' WHERE session_id=%s AND status='pending'",
                    (session_id,),
                )
                status = "expired"
        elif status == "active" and last_active is not None:
            if (now - last_active).total_seconds() > idle_timeout:
                cur.execute(
                    "UPDATE chat_handoffs SET status='closed', resolved_at=NOW() "
                    "WHERE session_id=%s AND status='active'",
                    (session_id,),
                )
                status = "closed"
        return {
            "session_id": session_id,
            "status": status,
            "reason": row[2] or "",
            "created_at": row[3].isoformat() if row[3] else None,
            "last_active_at": last_active.isoformat() if last_active else None,
            "agent_name": row[5] or "",
        }


def handoff_queue(pending_timeout: int = 900, idle_timeout: int = 1200) -> list[dict[str, Any]]:
    """待接入队列：先超时流转，再返回 pending + active 会话（含买家身份与画像）。"""
    from datetime import datetime, timezone

    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE chat_handoffs SET status='expired' "
            "WHERE status='pending' AND last_active_at < NOW() - INTERVAL '1 second' * %s",
            (pending_timeout,),
        )
        cur.execute(
            "UPDATE chat_handoffs SET status='closed', resolved_at=NOW() "
            "WHERE status='active' AND last_active_at < NOW() - INTERVAL '1 second' * %s",
            (idle_timeout,),
        )
        cur.execute(
            "SELECT h.session_id, h.status, h.reason, h.created_at, h.last_active_at, "
            "(SELECT m.content FROM chat_messages m WHERE m.session_id=h.session_id AND m.role='user' "
            " ORDER BY m.created_at DESC LIMIT 1) AS last_user_msg, "
            "s.owner, u.display_name "
            "FROM chat_handoffs h "
            "LEFT JOIN chat_sessions s ON s.session_id = h.session_id "
            "LEFT JOIN admin_users u ON u.username = s.owner "
            "WHERE h.status IN ('pending','active','resolved','closed') "
            "ORDER BY CASE h.status WHEN 'pending' THEN 0 WHEN 'active' THEN 1 ELSE 2 END, h.created_at ASC "
            "LIMIT 200"
        )
        rows = cur.fetchall()
    now = datetime.now(timezone.utc)
    out: list[dict[str, Any]] = []
    for r in rows:
        created_at = r[3]
        wait_secs = int((now - created_at).total_seconds()) if created_at else 0
        out.append({
            "session_id": r[0],
            "status": r[1],
            "reason": r[2] or "",
            "created_at": created_at.isoformat() if created_at else None,
            "last_active_at": r[4].isoformat() if r[4] else None,
            "waiting_secs": max(0, wait_secs),
            "last_user_message": r[5] or "",
            "owner": r[6] or "",
            "display_name": r[7] or r[6] or "",
            "profile": _buyer_profile(r[6] or ""),
        })
    return out


def _buyer_profile(owner: str) -> str:
    """买家画像摘要（user_memories 业务标签，供客服洞察面板展示）。

    Args:
        owner: 会话归属账号（username）。

    Returns:
        "肤质:干皮; 偏好:保湿" 形式的摘要；无画像返回空串。
    """
    if not owner:
        return ""
    try:
        from agent_base.storage.memory import retrieve_memory

        memories = retrieve_memory(owner, top_k=5)
        parts = []
        for m in memories:
            key = str(m.get("key", ""))
            value = m.get("value", "")
            if isinstance(value, list):
                value = ",".join(str(v) for v in value)
            if key:
                parts.append(f"{key}:{value}")
        return "; ".join(parts)
    except Exception:
        return ""


def handoff_reply(session_id: str, content: str, agent_name: str = "") -> dict[str, Any]:
    """人工回复：写 chat_messages(role=agent) + 置 active + 刷新活跃时间。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_messages (session_id, role, content) VALUES (%s, 'agent', %s)",
            (session_id, content),
        )
        cur.execute(
            "UPDATE chat_handoffs SET status='active', last_active_at=NOW(), agent_name=%s "
            "WHERE session_id=%s",
            (agent_name, session_id),
        )
    return {"ok": True, "session_id": session_id, "status": "active"}


def handoff_resolve(session_id: str, mode: str = "ai") -> dict[str, Any]:
    """转回 AI（resolved）或关闭（closed），记录保留。"""
    # 'ai' / 'resolved' 均表示问题已解决（resolved）；'close' 表示未解决直接关闭
    status = "resolved" if mode in ("ai", "resolved") else "closed"
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE chat_handoffs SET status=%s, resolved_at=NOW() WHERE session_id=%s",
            (status, session_id),
        )
    return {"ok": True, "session_id": session_id, "status": status}


def handoff_rate(session_id: str, rating: int, comment: str = "") -> dict[str, Any]:
    """人工客服结束后，由用户端写入服务评分。"""
    rating = max(1, min(5, int(rating or 0)))
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE chat_handoffs SET rating=%s, rating_comment=%s, rated_at=NOW() "
            "WHERE session_id=%s AND status IN ('resolved','closed')",
            (rating, str(comment or "")[:500], session_id),
        )
    return {"ok": True, "session_id": session_id, "rating": rating}


def handoff_rating_get(session_id: str) -> dict[str, Any]:
    """查询用户已提交的客服评分。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT rating, rating_comment, rated_at FROM chat_handoffs WHERE session_id=%s",
                (session_id,),
            )
            row = cur.fetchone()
        if not row:
            return {"session_id": session_id, "rating": 0, "comment": "", "rated_at": None}
        return {
            "session_id": session_id,
            "rating": int(row[0] or 0),
            "comment": str(row[1] or ""),
            "rated_at": row[2].isoformat() if row[2] else None,
        }
    except Exception:
        return {"session_id": session_id, "rating": 0, "comment": "", "rated_at": None}


def handoff_recover_if_needed(session_id: str, role: str) -> None:
    """用户发消息时恢复会话转人工状态。

    expired 重新排队人工；closed 清除记录重新排队；
    resolved（已解决）保留历史记录（AI 正常对话，不抹掉管理端已解决统计）。
    """
    if role != "user":
        return
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE chat_handoffs SET status='pending', last_active_at=NOW(), resolved_at=NULL "
                "WHERE session_id=%s AND status='expired'",
                (session_id,),
            )
            cur.execute(
                "DELETE FROM chat_handoffs WHERE session_id=%s AND status='closed'",
                (session_id,),
            )
    except Exception:
        pass


def chat_append(
    session_id: str,
    role: str,
    content: str,
    owner: str = "",
    sources: list[dict[str, Any]] | None = None,
) -> None:
    """Append a chat message (session history for multi-turn context).

    SEC-2: 首次写入时绑定会话归属（owner），已存在的会话保持原 owner。

    Args:
        session_id: 会话 ID。
        role: 消息角色（user/assistant/agent）。
        content: 消息内容。
        owner: 会话归属账号。
        sources: 检索召回依据（assistant 消息写入，JSONB）。
    """
    handoff_recover_if_needed(session_id, role)
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_messages (session_id, role, content, sources) VALUES (%s,%s,%s,%s)",
            (session_id, role, content, psycopg2.extras.Json(sources or []) if sources else None),
        )
        if owner:
            cur.execute(
                "INSERT INTO chat_sessions (session_id, owner) VALUES (%s,%s) "
                "ON CONFLICT (session_id) DO NOTHING",
                (session_id, owner),
            )


def session_owner(session_id: str) -> str:
    """SEC-2: 查询会话归属账号；无记录返回空串。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT owner FROM chat_sessions WHERE session_id=%s", (session_id,))
        row = cur.fetchone()
    return row[0] if row else ""


def ensure_session_owner(session_id: str, owner: str) -> None:
    """给未绑定 owner 的会话绑定归属（BUG-15 修复）。

    转人工时如会话尚未绑定归属（如新建会话未发消息），此处补绑；
    owner 为空直接跳过；已存在 owner 不覆盖（ON CONFLICT DO NOTHING）。
    """
    if not owner or not session_id:
        return
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_sessions (session_id, owner) VALUES (%s, %s) "
            "ON CONFLICT (session_id) DO NOTHING",
            (session_id, owner),
        )


def chat_history(session_id: str, limit: int = 12) -> list[dict[str, Any]]:
    """获取会话最近消息（按时间从旧到新）。

    BUG-21：同时间戳批量落库时 ORDER BY created_at 不稳定，
    追加 id 兜底保证消息顺序与写入一致（客服端轮次标注/提问回查依赖）。
    """
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT session_id, role, content, created_at, sources FROM chat_messages "
            "WHERE session_id=%s ORDER BY created_at DESC, id DESC LIMIT %s",
            (session_id, limit),
        )
        rows = cur.fetchall()
    rows.reverse()
    return [
        {
            "role": r[1],
            "content": r[2],
            "created_at": r[3].isoformat() if r[3] else None,
            "sources": r[4],
        }
        for r in rows
    ]


def chat_sessions() -> list[dict[str, Any]]:
    """会话列表：按 session_id 分组（最新时间 + 首条用户消息做标题）。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT session_id, MAX(created_at) AS latest FROM chat_messages "
            "GROUP BY session_id ORDER BY latest DESC"
        )
        sessions = cur.fetchall()
    out: list[dict[str, Any]] = []
    for session_id, latest in sessions:
        title = ""
        try:
            for m in chat_history(session_id, limit=100):
                if m["role"] == "user":
                    title = m["content"][:20] + ("..." if len(m["content"]) > 20 else "")
                    break
        except Exception:
            pass
        out.append({
            "session_id": session_id,
            "title": title,
            "updated_at": latest.isoformat() if latest else None,
        })
    return out


def chat_sessions_for_owner(owner: str) -> list[dict[str, Any]]:
    """返回指定用户拥有的会话列表，供用户端侧边栏恢复历史会话。"""
    if not owner:
        return []
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT s.session_id, s.updated_at, "
                "(SELECT m.content FROM chat_messages m "
                " WHERE m.session_id = s.session_id AND m.role='user' "
                " ORDER BY m.created_at DESC LIMIT 1) AS title "
                "FROM chat_sessions s WHERE s.owner=%s ORDER BY s.updated_at DESC",
                (owner,),
            )
            rows = cur.fetchall()
        return [
            {
                "session_id": str(r[0]),
                "title": str(r[2] or "")[:24] or "新对话",
                "updated_at": r[1].isoformat() if r[1] else None,
            }
            for r in rows
        ]
    except Exception:
        return []


def delete_chat_session(session_id: str) -> bool:
    """删除会话及关联数据（P1.5-7a 删会话清短期记忆）。

    清理范围：
    - ``chat_messages``：会话消息原文（短期记忆）；
    - ``chat_sessions``：会话归属绑定；
    - ``chat_handoffs``：转人工记录（避免管理端队列出现孤儿会话）。

    长期记忆（``user_memories``）属于用户画像，**不随会话删除**。

    Args:
        session_id: 会话 ID。

    Returns:
        是否有匹配的会话消息被删除。
    """
    if not session_id:
        return False
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM chat_messages WHERE session_id=%s", (session_id,))
        deleted = cur.rowcount > 0
        cur.execute("DELETE FROM chat_sessions WHERE session_id=%s", (session_id,))
        cur.execute("DELETE FROM chat_handoffs WHERE session_id=%s", (session_id,))
        # LangGraph PostgresSaver checkpoint（thread_id=session_id）：会话删除时一并清理，
        # 防止 checkpoint 表随会话累积膨胀（表不存在时静默跳过）。
        try:
            cur.execute("DELETE FROM checkpoint_writes WHERE thread_id=%s", (session_id,))
            cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id=%s", (session_id,))
            cur.execute("DELETE FROM checkpoints WHERE thread_id=%s", (session_id,))
        except Exception:
            pass
    return deleted
def purge_old_chat_messages(days: int = 30) -> dict[str, int]:
    """清理超过保留期的会话消息（P1.5-7c PG 短期记忆保留策略）。

    只删 ``chat_messages`` 中超过保留期的消息，并清理失去消息的
    ``chat_sessions`` 归属孤儿；``chat_handoffs`` 为运营统计数据，保留。

    Args:
        days: 保留天数（消息创建时间距今超过该值即删除）。

    Returns:
        ``{"messages": 删除消息数, "orphan_sessions": 清理归属孤儿数}``。
    """
    if not days or days <= 0:
        return {"messages": 0, "orphan_sessions": 0}
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM chat_messages WHERE created_at < NOW() - INTERVAL '1 day' * %s",
            (days,),
        )
        deleted = max(0, cur.rowcount or 0)
        cur.execute(
            "DELETE FROM chat_sessions s WHERE NOT EXISTS "
            "(SELECT 1 FROM chat_messages m WHERE m.session_id = s.session_id)"
        )
        orphan = max(0, cur.rowcount or 0)
    return {"messages": deleted, "orphan_sessions": orphan}


def eval_run_insert(
    name: str,
    total_cases: int,
    intent_acc: float,
    recall_acc: float,
    fact_acc: float,
    compliance_acc: float,
    faithfulness_acc: float = 0,
    relevancy_acc: float = 0,
    overall: float = 0,
    ragas_faithfulness_acc: float = 0,
    ragas_relevancy_acc: float = 0,
    ragas_context_precision_acc: float = 0,
    ragas_context_recall_acc: float = 0,
) -> int:
    """插入一次全链路评测批次，返回 run_id。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO eval_runs (name, total_cases, intent_acc, recall_acc, fact_acc, compliance_acc, "
            "faithfulness_acc, relevancy_acc, overall, ragas_faithfulness_acc, ragas_relevancy_acc, "
            "ragas_context_precision_acc, ragas_context_recall_acc) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
            (name, total_cases, intent_acc, recall_acc, fact_acc, compliance_acc,
             faithfulness_acc, relevancy_acc, overall, ragas_faithfulness_acc, ragas_relevancy_acc,
             ragas_context_precision_acc, ragas_context_recall_acc),
        )
        row = cur.fetchone()
    return row[0] if row else 0


def eval_case_insert(
    run_id: int,
    question: str,
    expected_intent: str,
    actual_intent: str,
    intent_hit: bool,
    expected_source: str,
    recall_hit: bool,
    expected_facts: list[str],
    fact_hits: int,
    fact_total: int,
    compliance_ok: bool,
    answer: str,
    sources: list[dict[str, Any]],
    error: str,
    faithfulness: float = 0,
    relevancy: float = 0,
    ragas_faithfulness: float = 0,
    ragas_relevancy: float = 0,
    ragas_context_precision: float = 0,
    ragas_context_recall: float = 0,
) -> int:
    """插入一条评测用例明细，返回 id。"""
    import json as _json

    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO eval_cases (run_id, question, expected_intent, actual_intent, intent_hit, "
            "expected_source, recall_hit, expected_facts, fact_hits, fact_total, compliance_ok, "
            "faithfulness, relevancy, answer, sources, error, ragas_faithfulness, ragas_relevancy, "
            "ragas_context_precision, ragas_context_recall) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
            (
                run_id, question, expected_intent, actual_intent, intent_hit,
                expected_source, recall_hit, _json.dumps(expected_facts, ensure_ascii=False),
                fact_hits, fact_total, compliance_ok, faithfulness, relevancy,
                answer, _json.dumps(sources, ensure_ascii=False), error,
                ragas_faithfulness, ragas_relevancy, ragas_context_precision, ragas_context_recall,
            ),
        )
        row = cur.fetchone()
    return row[0] if row else 0


def eval_run_list(limit: int = 30) -> list[dict[str, Any]]:
    """评测批次列表（最新在前）。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, total_cases, intent_acc, recall_acc, fact_acc, compliance_acc, "
            "faithfulness_acc, relevancy_acc, overall, created_at, ragas_faithfulness_acc, "
            "ragas_relevancy_acc, ragas_context_precision_acc, ragas_context_recall_acc "
            "FROM eval_runs ORDER BY id DESC LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "name": r[1],
            "total_cases": r[2],
            "intent_acc": float(r[3] or 0),
            "recall_acc": float(r[4] or 0),
            "fact_acc": float(r[5] or 0),
            "compliance_acc": float(r[6] or 0),
            "faithfulness_acc": float(r[7] or 0),
            "relevancy_acc": float(r[8] or 0),
            "overall": float(r[9] or 0),
            "created_at": r[10].isoformat() if r[10] else None,
            "ragas_faithfulness_acc": float(r[11] or 0),
            "ragas_relevancy_acc": float(r[12] or 0),
            "ragas_context_precision_acc": float(r[13] or 0),
            "ragas_context_recall_acc": float(r[14] or 0),
        }
        for r in rows
    ]


def eval_run_get(run_id: int) -> dict[str, Any] | None:
    """评测批次 + 用例明细。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, total_cases, intent_acc, recall_acc, fact_acc, compliance_acc, "
            "faithfulness_acc, relevancy_acc, overall, created_at, ragas_faithfulness_acc, "
            "ragas_relevancy_acc, ragas_context_precision_acc, ragas_context_recall_acc "
            "FROM eval_runs WHERE id=%s",
            (run_id,),
        )
        r = cur.fetchone()
        if not r:
            return None
        run = {
            "id": r[0],
            "name": r[1],
            "total_cases": r[2],
            "intent_acc": float(r[3] or 0),
            "recall_acc": float(r[4] or 0),
            "fact_acc": float(r[5] or 0),
            "compliance_acc": float(r[6] or 0),
            "faithfulness_acc": float(r[7] or 0),
            "relevancy_acc": float(r[8] or 0),
            "overall": float(r[9] or 0),
            "created_at": r[10].isoformat() if r[10] else None,
            "ragas_faithfulness_acc": float(r[11] or 0),
            "ragas_relevancy_acc": float(r[12] or 0),
            "ragas_context_precision_acc": float(r[13] or 0),
            "ragas_context_recall_acc": float(r[14] or 0),
        }
        cur.execute(
            "SELECT id, question, expected_intent, actual_intent, intent_hit, expected_source, recall_hit, "
            "expected_facts, fact_hits, fact_total, compliance_ok, faithfulness, relevancy, answer, sources, error, "
            "ragas_faithfulness, ragas_relevancy, ragas_context_precision, ragas_context_recall "
            "FROM eval_cases WHERE run_id=%s ORDER BY id",
            (run_id,),
        )
        import json as _json

        cases = []
        for c in cur.fetchall():
            try:
                facts = _json.loads(c[7] or "[]")
            except Exception:
                facts = []
            try:
                srcs = _json.loads(c[14] or "[]")
            except Exception:
                srcs = []
            cases.append({
                "id": c[0],
                "question": c[1],
                "expected_intent": c[2],
                "actual_intent": c[3],
                "intent_hit": bool(c[4]),
                "expected_source": c[5],
                "recall_hit": bool(c[6]),
                "expected_facts": facts,
                "fact_hits": int(c[8] or 0),
                "fact_total": int(c[9] or 0),
                "compliance_ok": bool(c[10]),
                "faithfulness": float(c[11] or 0),
                "relevancy": float(c[12] or 0),
                "answer": c[13],
                "sources": srcs,
                "error": c[15],
                "ragas_faithfulness": float(c[16] or 0),
                "ragas_relevancy": float(c[17] or 0),
                "ragas_context_precision": float(c[18] or 0),
                "ragas_context_recall": float(c[19] or 0),
            })
        run["cases"] = cases
        return run


# 导入时自动初始化
ALIAS_SEED: dict[str, list[str]] = {
    "玻尿酸精华": ["玻尿酸保湿精华液"],
    "玻尿酸": ["玻尿酸保湿精华液", "玻尿酸补水面膜"],
    "神经酰胺": ["神经酰胺修护面霜"],
    "氨基酸洁面": ["氨基酸温和洁面乳"],
    "烟酰胺": ["烟酰胺焕亮精华"],
    "水杨酸": ["水杨酸净痘精华"],
    "防晒": ["轻透防晒乳 SPF50+ PA++++", "UPF50+防晒衣"],
    "T恤": ["白色纯棉T恤"],
    "白T": ["白色纯棉T恤"],
    "碎花裙": ["法式碎花连衣裙"],
    "连衣裙": ["法式碎花连衣裙", "莫代尔连衣裙"],
    "阔腿裤": ["高腰阔腿裤", "冰丝阔腿裤"],
    "衬衫": ["醋酸衬衫"],
    "洁面": ["氨基酸温和洁面乳"],
    "面霜": ["神经酰胺修护面霜"],
    "眼霜": ["视黄醇抗皱眼霜"],
    "面膜": ["玻尿酸补水面膜"],
    "祛痘": ["水杨酸净痘精华"],
    "控油": ["烟酰胺焕亮精华"],
    "抗老": ["视黄醇抗皱眼霜"],
    "多少钱": ["价格", "售价", "价位"],
    "贵不贵": ["价格", "售价", "划算"],
    "贵吗": ["价格", "售价"],
    "便宜吗": ["价格", "优惠"],
    "怎么用": ["使用方法", "用法", "用量"],
    "适合我吗": ["适合"],
    "能退货吗": ["退货", "退款"],
    "有货吗": ["库存", "有货", "断货"],
    "怎么搭": ["搭配"],
}


def alias_seed_from_json(
    json_path: str | None = None,
    force: bool = False,
) -> int:
    """别名种子入库（P32c：运行时读 PG；种子源内置 ALIAS_SEED）。

    幂等：已存在的 (alias, canonical) 不重复插入；force=True 时先清空重建。

    Args:
        json_path: 可选 aliases JSON 种子文件（缺省用内置 ALIAS_SEED）。
        force: True 时清空 alias_rules 后重建。

    Returns:
        插入的行数。
    """
    if json_path:
        from pathlib import Path
        import json as _json

        p = Path(json_path)
        if not p.exists():
            return 0
        data = _json.loads(p.read_text(encoding="utf-8"))
    else:
        data = ALIAS_SEED
    if not isinstance(data, dict):
        return 0
    rows = [
        (str(alias).lower(), str(canonical))
        for alias, canonicals in data.items()
        for canonical in (canonicals or [])
    ]
    with _conn() as conn:
        cur = conn.cursor()
        if force:
            cur.execute("DELETE FROM alias_rules")
        cur.executemany(
            "INSERT INTO alias_rules (alias, canonical) VALUES (%s,%s) "
            "ON CONFLICT (alias, canonical) DO NOTHING",
            rows,
        )
    return len(rows)


def alias_list() -> dict[str, list[str]]:
    """读 PG alias_rules → {alias: [canonical, ...]}（运行时数据源）。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT alias, canonical FROM alias_rules ORDER BY canonical")
            out: dict[str, list[str]] = {}
            for alias, canonical in cur.fetchall():
                out.setdefault(str(alias), []).append(str(canonical))
            return out
    except Exception:
        return {}


# --- FAQ 表（内置种子 → PG 运行时；json 文件已淘汰） ---

FAQ_SEED: list[tuple[str, str, str, str]] = [
    ("F001", "物流", "下单后多久发货？", "现货商品一般 48 小时内发货，预售商品以页面标注时间为准。"),
    ("F002", "物流", "发什么快递？", "默认中通/圆通快递，偏远地区时效可能延长 1-3 天。"),
    ("F003", "售后", "支持七天无理由退货吗？", "未拆封商品支持 7 天无理由退货，运费按平台规则承担；已拆封商品仅质量问题可退。"),
    ("F004", "售后", "过敏了可以退款吗？", "使用后出现过敏可申请过敏无忧退款，需提供过敏凭证（如医院诊断），审核通过后退款。"),
    ("F005", "发票", "可以开发票吗？", "支持电子发票，下单时填写抬头，发货后 7 天内发送到邮箱。"),
    ("F006", "成分", "怎么看成分表？", "商品详情页有完整成分表；需要查询具体成分浓度可在详情页或联系客服。"),
    ("F007", "使用", "使用新产品需要注意什么？", "建议先在耳后或手臂内侧做小范围测试，确认无不适后再上脸使用。"),
    ("F008", "效期", "产品保质期多久？", "未开封产品保质期一般 3 年，开封后请按包装标注的使用期限使用。"),
]


def faq_seed(force: bool = False) -> int:
    """FAQ 内置种子入库（P32c 同款模式：运行时读 PG，种子内置不依赖文件）。

    Args:
        force: True 时清空 faq 表后重建。

    Returns:
        插入的行数。
    """
    rows = FAQ_SEED
    with _conn() as conn:
        cur = conn.cursor()
        if force:
            cur.execute("DELETE FROM faq")
        cur.executemany(
            "INSERT INTO faq (id, category, question, answer) VALUES (%s,%s,%s,%s) "
            "ON CONFLICT (id) DO NOTHING",
            rows,
        )
    return len(rows)


def faq_title_map() -> dict[str, str]:
    """读 PG faq → {id: question}（运行时数据源，内置种子）。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, question FROM faq")
            return {str(r[0]): str(r[1]) for r in cur.fetchall()}
    except Exception:
        return {}


# --- IDF 缓存（PG 替代 json 文件；source_count 为语料版本键） ---


def idf_cache_load(version: str) -> dict[str, float] | None:
    """按语料内容指纹读取 IDF 缓存表。

    Args:
        version: Qdrant chunk 内容指纹（语料版本标记）。

    Returns:
        IDF 表；未命中或失败返回 None。
    """
    try:
        import json as _json

        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT table_json FROM idf_cache WHERE version=%s", (version,))
            row = cur.fetchone()
        if not row:
            return None
        data = row[0] if isinstance(row[0], dict) else _json.loads(row[0])
        return {str(k): float(v) for k, v in (data or {}).items()}
    except Exception:
        return None


def idf_cache_save(version: str, table: dict[str, float]) -> None:
    """写入 IDF 缓存表（upsert by version 指纹），失败静默。

    Args:
        version: Qdrant chunk 内容指纹。
        table: IDF 词表。
    """
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO idf_cache (version, table_json) VALUES (%s, %s) "
                "ON CONFLICT (version) DO UPDATE SET table_json=EXCLUDED.table_json, updated_at=NOW()",
                (version, psycopg2.extras.Json(table)),
            )
    except Exception:
        pass


# --- 商品目录与图文素材种子数据 ---

CATALOG_SEED: list[tuple[str, str, str, str, str]] = [
    ("P001", "玻尿酸保湿精华液", "花颜集", "精华", "中端"),
    ("P002", "神经酰胺修护面霜", "花颜集", "面霜", "中端"),
    ("P003", "氨基酸温和洁面乳", "花颜集", "洁面", "平价"),
    ("P004", "轻透防晒乳", "花颜集", "防晒", "中端"),
    ("P005", "烟酰胺焕亮精华", "花颜集", "精华", "中端"),
    ("P006", "玻尿酸补水面膜", "花颜集", "面膜", "中端"),
    ("P007", "视黄醇抗皱眼霜", "花颜集", "眼霜", "高端"),
    ("P008", "积雪草舒缓精华", "花颜集", "精华", "中端"),
    ("P009", "水杨酸净痘精华", "花颜集", "精华", "中端"),
    ("P010", "玻尿酸水乳套装", "花颜集", "套装", "中端"),
    ("P011", "山茶花润肤油", "花颜集", "护肤油", "高端"),
    ("P012", "美白防晒隔离霜", "花颜集", "防晒", "中端"),
    ("P013", "白色纯棉T恤", "简素", "T恤", "平价"),
    ("P014", "法式碎花连衣裙", "简素", "连衣裙", "中端"),
    ("P015", "高腰阔腿裤", "简素", "裤装", "中端"),
    ("P016", "纯色针织开衫", "简素", "针织", "平价"),
    ("P017", "醋酸衬衫", "简素", "衬衫", "中端"),
    ("P018", "UPF50+防晒衣", "简素", "防晒衣", "中端"),
    ("P019", "冰丝阔腿裤", "简素", "裤装", "平价"),
    ("P020", "莫代尔连衣裙", "简素", "连衣裙", "中端"),
]


def catalog_seed(force: bool = False) -> int:
    """将内置商品目录写入 PG catalog（幂等）。"""
    with _conn() as conn:
        cur = conn.cursor()
        if force:
            cur.execute("DELETE FROM catalog")
        for pid, name, brand, category, price_band in CATALOG_SEED:
            cur.execute(
                "INSERT INTO catalog (id, name, brand, category, price_band) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name, "
                "brand=EXCLUDED.brand, category=EXCLUDED.category, "
                "price_band=EXCLUDED.price_band, updated_at=NOW()",
                (pid, name, brand, category, price_band),
            )
    return len(CATALOG_SEED)


def media_seed(force: bool = False) -> int:
    """为每个商品写入一张预置商品图（SVG 动态渲染，不依赖生图模型）。"""
    rows = []
    media_templates = [
        ("main", "商品主图", 0),
        ("detail", "商品详情图", 1),
        ("scene", "使用场景图", 2),
    ]
    for pid, _, _, _, _ in CATALOG_SEED:
        for style, title, order in media_templates:
            rows.append(
                (
                    pid,
                    "image",
                    f"/media/products/{pid.lower()}.svg?style={style}",
                    title,
                    order,
                    "preset",
                    "active",
                )
            )
    with _conn() as conn:
        cur = conn.cursor()
        # 预置素材以 product_id + source 作为幂等键，避免每次导入重复追加。
        if force:
            cur.execute("DELETE FROM product_media")
        else:
            cur.execute("DELETE FROM product_media WHERE source='preset'")
        cur.executemany(
            "INSERT INTO product_media "
            "(product_id, media_type, url, title, sort_order, source, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (id) DO NOTHING",
            rows,
        )
    return len(rows)


def media_for_product_ids(product_ids: list[str], limit: int = 12) -> list[dict[str, Any]]:
    """按商品 ID 列表查询启用中的图文素材，按商品和排序返回。"""
    if not product_ids:
        return []
    ids = [str(x) for x in product_ids if x]
    if not ids:
        return []
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, product_id, media_type, url, title, sort_order, source "
                "FROM product_media "
                "WHERE status='active' AND product_id = ANY(%s) "
                "ORDER BY product_id, sort_order, id LIMIT %s",
                (ids, limit),
            )
            return [
                {
                    "id": r[0],
                    "product_id": r[1],
                    "media_type": r[2],
                    "url": r[3],
                    "title": r[4],
                    "sort_order": r[5],
                    "source": r[6],
                }
                for r in cur.fetchall()
            ]
    except Exception:
        return []


def media_for_product_cards(product_ids: list[str], limit: int = 6) -> list[dict[str, Any]]:
    """多商品推荐场景：每个商品只返回一张主图和一条视频，避免素材刷屏。"""
    if not product_ids:
        return []
    ids = [str(x) for x in product_ids if x]
    if not ids:
        return []
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT m.product_id, m.media_type, m.url, m.title, m.sort_order, c.name "
                "FROM product_media m JOIN catalog c ON c.id = m.product_id "
                "WHERE m.status='active' AND m.product_id = ANY(%s) "
                "ORDER BY m.product_id, "
                "CASE WHEN m.media_type='image' AND m.sort_order=0 THEN 0 "
                "     WHEN m.media_type='video' THEN 1 ELSE 2 END, m.sort_order, m.id",
                (ids,),
            )
            grouped: dict[str, list[dict[str, Any]]] = {}
            for r in cur.fetchall():
                grouped.setdefault(str(r[0]), []).append(
                    {
                        "product_id": str(r[0]),
                        "media_type": r[1],
                        "url": r[2],
                        "title": r[3],
                        "sort_order": int(r[4] or 0),
                        "product_name": r[5],
                    }
                )
        cards: list[dict[str, Any]] = []
        for pid in ids:
            rows = grouped.get(pid, [])
            if not rows:
                continue
            main = next((x for x in rows if x["media_type"] == "image" and x["sort_order"] == 0), rows[0])
            video = next((x for x in rows if x["media_type"] == "video"), None)
            cards.append(main)
            if video:
                cards.append(video)
            if sum(1 for x in cards if x["media_type"] == "image") >= limit:
                break
        return cards
    except Exception:
        return []


# --- 游客体系（v0.53）：升级迁移 + 孤儿清理 ---


def migrate_guest_to_user(guest_uid: str, new_username: str) -> dict[str, Any]:
    """游客 → 真实账号升级：把游客的会话/长期记忆迁移到新账号（历史不丢）。

    迁移范围：
    - chat_sessions.owner：guest → new（会话消息按 session_id 存储，随会话走）；
    - user_memories：逐条复制到新账号（已存在同 key 不覆盖），再删游客条目；
    - Redis 短期记忆按 session 存储，随会话归属生效，无需单独迁移。

    Args:
        guest_uid: 游客账号（guest_ 开头）。
        new_username: 升级后的真实账号（手机号/密码登录账号）。

    Returns:
        {ok, sessions, memories}；任一步失败返回 {ok: False}（不阻断登录）。
    """
    if not guest_uid or not guest_uid.startswith("guest_"):
        return {"ok": False, "reason": "not-guest"}
    if not new_username or guest_uid == new_username:
        return {"ok": False, "reason": "bad-target"}
    try:
        with _conn() as conn:
            cur = conn.cursor()
            # 1) 会话归属迁移（消息按 session_id 存储，自动跟随）
            cur.execute(
                "UPDATE chat_sessions SET owner=%s, updated_at=NOW() WHERE owner=%s",
                (new_username[:60], guest_uid),
            )
            sessions = cur.rowcount
            # 2) 长期记忆迁移：新账号已有同 key 不覆盖，游客条目迁移后删除
            cur.execute(
                "INSERT INTO user_memories (user_id, memory_key, value, source, confidence, ttl) "
                "SELECT %s, memory_key, value, source, confidence, ttl FROM user_memories "
                "WHERE user_id=%s ON CONFLICT (user_id, memory_key) DO NOTHING",
                (new_username[:60], guest_uid),
            )
            memories = cur.rowcount
            cur.execute("DELETE FROM user_memories WHERE user_id=%s", (guest_uid,))
            # 3) 删除游客账号行（token 鉴权不依赖用户行，旧 token 自然失效于身份概念）
            cur.execute("DELETE FROM users WHERE user_id=%s", (guest_uid,))
            conn.commit()
        return {"ok": True, "sessions": int(sessions), "memories": int(memories)}
    except Exception:
        return {"ok": False, "reason": "db-error"}


def cleanup_guests(retention_days: int = 30) -> dict[str, Any]:
    """清理长期不活跃的游客账号及其会话/记忆/消息（孤儿数据回收）。

    Args:
        retention_days: 游客保留天数（按会话最后活跃时间，默认 30）。

    Returns:
        {users, sessions, messages, memories}；失败返回全 0。
    """
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cutoff = "NOW() - make_interval(days => %s)"
            # 1) 找过期游客的会话
            cur.execute(
                "SELECT session_id FROM chat_sessions "
                f"WHERE owner LIKE 'guest\\_%' AND updated_at < {cutoff}",
                (int(retention_days),),
            )
            old_sessions = [r[0] for r in cur.fetchall()]
            # 2) 删消息/转人工记录（按会话）
            messages = 0
            for sid in old_sessions:
                cur.execute("DELETE FROM chat_messages WHERE session_id=%s", (sid,))
                messages += cur.rowcount
                cur.execute("DELETE FROM chat_handoffs WHERE session_id=%s", (sid,))
            # 3) 删会话归属
            cur.execute(
                "DELETE FROM chat_sessions "
                f"WHERE owner LIKE 'guest\\_%' AND updated_at < {cutoff}",
                (int(retention_days),),
            )
            sessions = cur.rowcount
            # 4) 删过期游客的长期记忆 + 账号行
            cur.execute(
                "DELETE FROM user_memories WHERE user_id LIKE 'guest\\_%' "
                f"AND updated_at < {cutoff}",
                (int(retention_days),),
            )
            memories = cur.rowcount
            cur.execute(
                "DELETE FROM users WHERE user_id LIKE 'guest\\_%' "
                f"AND created_at < {cutoff}",
                (int(retention_days),),
            )
            users = cur.rowcount
            conn.commit()
        return {
            "users": int(users),
            "sessions": int(sessions),
            "messages": int(messages),
            "memories": int(memories),
        }
    except Exception:
        return {"users": 0, "sessions": 0, "messages": 0, "memories": 0}

# --- 图片知识库（Phase 2）：media_documents 增删改查 ---


def media_document_create(
    *,
    original_name: str = "",
    url: str = "",
    mime_type: str = "",
    size_bytes: int = 0,
    source_type: str = "upload",
    description: str = "",
    parse_type: str = "image",
) -> int:
    """新增图片/视频文档记录；返回自增 id，失败返回 0。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO media_documents "
                "(original_name, url, mime_type, size_bytes, source_type, description, parse_type) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (
                    original_name[:300],
                    url[:600],
                    mime_type[:80],
                    int(size_bytes or 0),
                    source_type[:40],
                    description[:2000],
                    (parse_type or "image")[:20],
                ),
            )
            row = cur.fetchone()
            conn.commit()
            return int(row[0]) if row else 0
    except Exception:
        return 0


def media_document_list(
    *,
    status: str | None = None,
    product_id: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """图片知识库列表（可按审核状态 / 绑定商品过滤），失败返回 []。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            sql = (
                "SELECT id, original_name, url, description, ocr_text, product_id, "
                "source_type, status, mime_type, size_bytes, created_at, updated_at, "
                "video_urls, poster_url, duration_sec, parse_type "
                "FROM media_documents WHERE 1=1"
            )
            args: list[Any] = []
            if status:
                sql += " AND status=%s"
                args.append(status)
            if product_id:
                sql += " AND product_id=%s"
                args.append(product_id)
            sql += " ORDER BY id DESC LIMIT %s"
            args.append(int(limit))
            cur.execute(sql, tuple(args))
            return [
                {
                    "id": r[0],
                    "original_name": r[1],
                    "url": r[2],
                    "description": r[3],
                    "ocr_text": r[4],
                    "product_id": r[5],
                    "source_type": r[6],
                    "status": r[7],
                    "mime_type": r[8],
                    "size_bytes": r[9],
                    "created_at": str(r[10]),
                    "updated_at": str(r[11]),
                    "video_urls": r[12] or [],
                    "poster_url": r[13] or "",
                    "duration_sec": int(r[14] or 0),
                    "parse_type": r[15] or "image",
                }
                for r in cur.fetchall()
            ]
    except Exception:
        return []


def media_document_get(media_id: int) -> dict[str, Any] | None:
    """单条图片文档记录；不存在/失败返回 None。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, original_name, url, description, ocr_text, product_id, "
                "source_type, status, mime_type, size_bytes, created_at, updated_at, "
                "video_urls, poster_url, duration_sec, parse_type "
                "FROM media_documents WHERE id=%s",
                (int(media_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "original_name": row[1],
                "url": row[2],
                "description": row[3],
                "ocr_text": row[4],
                "product_id": row[5],
                "source_type": row[6],
                "status": row[7],
                "mime_type": row[8],
                "size_bytes": row[9],
                "created_at": str(row[10]),
                "updated_at": str(row[11]),
                "video_urls": row[12] or [],
                "poster_url": row[13] or "",
                "duration_sec": int(row[14] or 0),
                "parse_type": row[15] or "image",
            }
    except Exception:
        return None


def media_document_bind(media_id: int, product_id: str) -> bool:
    """绑定图片到商品；记录不存在/失败返回 False。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE media_documents SET product_id=%s, updated_at=NOW() WHERE id=%s",
                (product_id[:40], int(media_id)),
            )
            conn.commit()
            return cur.rowcount > 0
    except Exception:
        return False


def media_document_set_status(media_id: int, status: str) -> bool:
    """审核状态流转（pending/approved/rejected）；记录不存在/失败返回 False。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE media_documents SET status=%s, updated_at=NOW() WHERE id=%s",
                (status[:20], int(media_id)),
            )
            conn.commit()
            return cur.rowcount > 0
    except Exception:
        return False


def media_document_update_parse(
    media_id: int,
    *,
    description: str = "",
    ocr_text: str = "",
    video_urls: list[str] | None = None,
    poster_url: str = "",
    duration_sec: int = 0,
    parse_type: str = "",
) -> bool:
    """写入解析结果（OCR/视觉理解/视频抽帧）；记录不存在/失败返回 False。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            if video_urls is not None:
                cur.execute(
                    "UPDATE media_documents SET description=%s, ocr_text=%s, "
                    "video_urls=%s, poster_url=%s, duration_sec=%s, parse_type=%s, "
                    "updated_at=NOW() WHERE id=%s",
                    (
                        description[:4000],
                        ocr_text[:20000],
                        json.dumps([str(u)[:600] for u in video_urls], ensure_ascii=False),
                        poster_url[:600],
                        int(duration_sec or 0),
                        (parse_type or "image")[:20],
                        int(media_id),
                    ),
                )
            else:
                cur.execute(
                    "UPDATE media_documents SET description=%s, ocr_text=%s, updated_at=NOW() "
                    "WHERE id=%s",
                    (description[:4000], ocr_text[:20000], int(media_id)),
                )
            conn.commit()
            return cur.rowcount > 0
    except Exception:
        return False


def media_document_delete(media_id: int) -> dict[str, Any] | None:
    """删除图片文档记录并返回原记录（调用方据此清理磁盘文件）；不存在/失败返回 None。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM media_documents WHERE id=%s RETURNING id, url, original_name",
                (int(media_id),),
            )
            row = cur.fetchone()
            conn.commit()
            if not row:
                return None
            return {"id": row[0], "url": row[1], "original_name": row[2]}
    except Exception:
        return None


# --- 文件清洗草稿（两段式入库第一段） ---


def clean_draft_create(
    *,
    original_name: str = "",
    engine: str = "",
    raw_text: str = "",
) -> int:
    """新建清洗草稿；返回自增 id，失败返回 0。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO clean_drafts (original_name, engine, raw_text) "
                "VALUES (%s,%s,%s) RETURNING id",
                (original_name[:300], engine[:40], raw_text[:200000]),
            )
            row = cur.fetchone()
            conn.commit()
            return int(row[0]) if row else 0
    except Exception:
        return 0


def clean_draft_get(draft_id: int) -> dict[str, Any] | None:
    """单条清洗草稿（含原文与清洗后文本）；不存在/失败返回 None。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, original_name, engine, raw_text, cleaned_text, status, "
                "created_at, updated_at FROM clean_drafts WHERE id=%s",
                (int(draft_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "original_name": row[1],
                "engine": row[2],
                "raw_text": row[3],
                "cleaned_text": row[4],
                "status": row[5],
                "created_at": str(row[6]),
                "updated_at": str(row[7]),
            }
    except Exception:
        return None


def clean_draft_list(limit: int = 100) -> list[dict[str, Any]]:
    """清洗草稿列表（按创建倒序）；失败返回 []。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, original_name, engine, status, created_at, updated_at "
                "FROM clean_drafts ORDER BY id DESC LIMIT %s",
                (int(limit),),
            )
            return [
                {
                    "id": r[0],
                    "original_name": r[1],
                    "engine": r[2],
                    "status": r[3],
                    "created_at": str(r[4]),
                    "updated_at": str(r[5]),
                }
                for r in cur.fetchall()
            ]
    except Exception:
        return []


def clean_draft_update(draft_id: int, cleaned_text: str) -> bool:
    """保存人工清洗后的文本；记录不存在/失败返回 False。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE clean_drafts SET cleaned_text=%s, updated_at=NOW() WHERE id=%s",
                (cleaned_text[:200000], int(draft_id)),
            )
            conn.commit()
            return cur.rowcount > 0
    except Exception:
        return False


def clean_draft_set_status(draft_id: int, status: str) -> bool:
    """草稿状态流转（pending/pushed/discarded）。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE clean_drafts SET status=%s, updated_at=NOW() WHERE id=%s",
                (status[:20], int(draft_id)),
            )
            conn.commit()
            return cur.rowcount > 0
    except Exception:
        return False


def clean_draft_delete(draft_id: int) -> bool:
    """删除清洗草稿；不存在/失败返回 False。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM clean_drafts WHERE id=%s", (int(draft_id),))
            conn.commit()
            return cur.rowcount > 0
    except Exception:
        return False


try:
    init_db()
    _migrate_documents_add_content()
    _migrate_documents_add_deleted_at()
    _migrate_chat_messages_add_sources()
    _migrate_eval_columns()
    _migrate_eval_ragas_columns()
    _migrate_token_usage_failure_columns()
    _migrate_document_strategy_p19()
    catalog_seed()
    media_seed()
    intent_seed_from_yaml()
    alias_seed_from_json()
except Exception:
    # PG 不可用时静默跳过（评测/测试环境不需要意图管理表）
    pass


# --- P2 多源知识库：评价/搭配/案例 CRUD ---


def review_insert(
    product_id: str,
    content: str,
    rating: int = 5,
    sentiment: str = "positive",
    source: str = "platform",
) -> int:
    """插入一条用户评价，返回 id。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO product_reviews (product_id, rating, content, sentiment, source) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (product_id, rating, content, sentiment, source),
        )
        row = cur.fetchone()
    return row[0] if row else -1


def review_list(product_id: str | None = None, sentiment: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """查询评价列表（按商品/情感过滤）。"""
    with _conn() as conn:
        cur = conn.cursor()
        clauses = []
        params: list[Any] = []
        if product_id:
            clauses.append("product_id = %s")
            params.append(product_id)
        if sentiment:
            clauses.append("sentiment = %s")
            params.append(sentiment)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        cur.execute(
            f"SELECT id, product_id, rating, content, sentiment, source, created_at "
            f"FROM product_reviews {where} ORDER BY created_at DESC LIMIT %s",
            params + [limit],
        )
        return [_review_row(r) for r in cur.fetchall()]


def _review_row(row: Any) -> dict[str, Any]:
    return {
        "id": row[0], "product_id": row[1], "rating": row[2],
        "content": row[3], "sentiment": row[4], "source": row[5],
        "created_at": str(row[6]) if row[6] else "",
    }


def combo_insert(name: str, product_ids: list[str], scenario: str = "", description: str = "") -> int:
    """插入一条搭配方案，返回 id。"""
    import json as _json
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO product_combos (name, product_ids, scenario, description) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (name, _json.dumps(product_ids, ensure_ascii=False), scenario, description),
        )
        row = cur.fetchone()
    return row[0] if row else -1


def combo_list(scenario: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """查询搭配方案列表（按场景过滤）。"""
    with _conn() as conn:
        cur = conn.cursor()
        if scenario:
            cur.execute(
                "SELECT id, name, product_ids, scenario, description, created_at "
                "FROM product_combos WHERE scenario ILIKE %s ORDER BY created_at DESC LIMIT %s",
                (f"%{scenario}%", limit),
            )
        else:
            cur.execute(
                "SELECT id, name, product_ids, scenario, description, created_at "
                "FROM product_combos ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
        return [_combo_row(r) for r in cur.fetchall()]


def _combo_row(row: Any) -> dict[str, Any]:
    import json as _json
    return {
        "id": row[0], "name": row[1],
        "product_ids": _json.loads(str(row[2])) if isinstance(row[2], str) else (row[2] or []),
        "scenario": row[3], "description": row[4],
        "created_at": str(row[5]) if row[5] else "",
    }


def case_insert(product_id: str, skin_type: str = "", scenario: str = "", duration: str = "", result: str = "") -> int:
    """插入一条客户案例，返回 id。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO customer_cases (product_id, skin_type, scenario, duration, result) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (product_id, skin_type, scenario, duration, result),
        )
        row = cur.fetchone()
    return row[0] if row else -1


def case_list(product_id: str | None = None, skin_type: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """查询客户案例列表（按商品/肤质过滤）。"""
    with _conn() as conn:
        cur = conn.cursor()
        clauses = []
        params: list[Any] = []
        if product_id:
            clauses.append("product_id = %s")
            params.append(product_id)
        if skin_type:
            clauses.append("skin_type = %s")
            params.append(skin_type)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        cur.execute(
            f"SELECT id, product_id, skin_type, scenario, duration, result, created_at "
            f"FROM customer_cases {where} ORDER BY created_at DESC LIMIT %s",
            params + [limit],
        )
        return [_case_row(r) for r in cur.fetchall()]


def _case_row(row: Any) -> dict[str, Any]:
    return {
        "id": row[0], "product_id": row[1], "skin_type": row[2],
        "scenario": row[3], "duration": row[4], "result": row[5],
        "created_at": str(row[6]) if row[6] else "",
    }



# ── P30: 切分参数自定义覆盖（doc_type → 分隔符/块大小/重叠，运营可编辑） ──

def chunk_override_get(doc_type: str) -> dict[str, Any] | None:
    """读取某 doc_type 的切分参数覆盖；无覆盖返回 None。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT chunk_size, chunk_overlap, separators, updated_by, updated_at "
                "FROM chunk_profile_overrides WHERE doc_type=%s",
                (str(doc_type),),
            )
            row = cur.fetchone()
            if not row:
                return None
            seps = row[2] if isinstance(row[2], list) else json.loads(row[2] or "[]")
            return {
                "doc_type": str(doc_type),
                "chunk_size": row[0],
                "chunk_overlap": row[1],
                "separators": [str(s) for s in seps],
                "updated_by": row[3] or "",
                "updated_at": str(row[4]) if row[4] else "",
            }
    except Exception:
        return None


def chunk_override_upsert(
    doc_type: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
    updated_by: str = "admin",
) -> bool:
    """写入/更新某 doc_type 的切分参数覆盖。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO chunk_profile_overrides (doc_type, chunk_size, chunk_overlap, separators, updated_by, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, NOW()) "
                "ON CONFLICT (doc_type) DO UPDATE SET chunk_size=EXCLUDED.chunk_size, "
                "chunk_overlap=EXCLUDED.chunk_overlap, separators=EXCLUDED.separators, "
                "updated_by=EXCLUDED.updated_by, updated_at=NOW()",
                (str(doc_type), int(chunk_size), int(chunk_overlap), psycopg2.extras.Json([str(s) for s in separators]), updated_by),
            )
            return True
    except Exception:
        return False


def chunk_override_delete(doc_type: str) -> bool:
    """删除覆盖（回到代码默认档位）。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM chunk_profile_overrides WHERE doc_type=%s", (str(doc_type),))
            return True
    except Exception:
        return False


def chunk_override_list() -> list[dict[str, Any]]:
    """全部覆盖列表（管理端展示）。"""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT doc_type, chunk_size, chunk_overlap, separators, updated_by, updated_at "
                "FROM chunk_profile_overrides ORDER BY doc_type"
            )
            rows = cur.fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            seps = row[3] if isinstance(row[3], list) else json.loads(row[3] or "[]")
            out.append({
                "doc_type": row[0],
                "chunk_size": row[1],
                "chunk_overlap": row[2],
                "separators": [str(s) for s in seps],
                "updated_by": row[4] or "",
                "updated_at": str(row[5]) if row[5] else "",
            })
        return out
    except Exception:
        return []
