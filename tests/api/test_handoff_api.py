"""v0.48：转人工 HITL 测试——触发/队列/回复/转回/超时/自动恢复/supervisor 状态判断。"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from agent_base.storage.pg import _conn, chat_append, chat_history, handoff_check


def _session_id() -> str:
    return f"handoff_{uuid.uuid4().hex[:8]}"


def _cleanup(session_id: str) -> None:
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM chat_handoffs WHERE session_id=%s", (session_id,))
            cur.execute("DELETE FROM chat_messages WHERE session_id=%s", (session_id,))
            cur.execute("DELETE FROM chat_sessions WHERE session_id=%s", (session_id,))
    except Exception:
        pass


def test_trigger_and_queue(client: TestClient, user_headers: dict[str, str], agent_headers: dict[str, str]):
    sid = _session_id()
    try:
        r = client.post(f"/api/handoff/{sid}", json={"reason": "用户要求人工"}, headers=user_headers)
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "pending"

        queue = client.get("/api/handoff/queue", headers=agent_headers).json()["handoffs"]
        item = next((h for h in queue if h["session_id"] == sid), None)
        assert item is not None, "触发后应出现在待接入队列"
        assert item["status"] == "pending"
        assert item["reason"] == "用户要求人工"
        assert item["waiting_secs"] >= 0
    finally:
        _cleanup(sid)


def test_handoff_trigger_keeps_active_status(
    client: TestClient,
    user_headers: dict[str, str],
    agent_headers: dict[str, str],
):
    """BUG-17：active 会话重复触发转人工 → 返回 active 而非 pending。"""
    sid = _session_id()
    try:
        r = client.post(f"/api/handoff/{sid}", json={}, headers=user_headers)
        assert r.status_code == 200 and r.json()["status"] == "pending", r.text
        r1 = client.post(f"/api/handoff/{sid}/reply", json={"content": "收到"}, headers=agent_headers)
        assert r1.status_code == 200 and r1.json()["status"] == "active", r1.text
        r2 = client.post(f"/api/handoff/{sid}", json={}, headers=user_headers)
        assert r2.status_code == 200, r2.text
        assert r2.json()["status"] == "active", f"active 会话重复触发应保持 active: {r2.text}"
    finally:
        _cleanup(sid)


def test_reply_records_agent_name(
    client: TestClient,
    user_headers: dict[str, str],
    agent_headers: dict[str, str],
):
    """BUG-19：客服回复后 agent_name 落库（by_agent 统计依赖，后端从 token 解析）。"""
    sid = _session_id()
    client.post(f"/api/handoff/{sid}", json={}, headers=user_headers)
    try:
        r = client.post(f"/api/handoff/{sid}/reply", json={"content": "你好"}, headers=agent_headers)
        assert r.status_code == 200, r.text
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT agent_name FROM chat_handoffs WHERE session_id=%s", (sid,))
            row = cur.fetchone()
        assert row and row[0] == "agent", f"agent_name 应为 agent，实际: {row}"
    finally:
        _cleanup(sid)


def test_reply_and_resolve(client: TestClient, user_headers: dict[str, str], agent_headers: dict[str, str]):
    sid = _session_id()
    client.post(f"/api/handoff/{sid}", json={}, headers=user_headers)
    try:
        r = client.post(f"/api/handoff/{sid}/reply", json={"content": "您好，我是人工客服小王"}, headers=agent_headers)
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "active"

        history = chat_history(sid, limit=10)
        agent_msg = [m for m in history if m["role"] == "agent"]
        assert agent_msg and agent_msg[-1]["content"] == "您好，我是人工客服小王"

        r2 = client.post(f"/api/handoff/{sid}/resolve", json={"mode": "ai"}, headers=agent_headers)
        assert r2.status_code == 200, r2.text
        assert r2.json()["status"] == "resolved"
    finally:
        _cleanup(sid)


def test_pending_timeout_expired(client: TestClient, user_headers: dict[str, str], agent_headers: dict[str, str]):
    sid = _session_id()
    client.post(f"/api/handoff/{sid}", json={}, headers=user_headers)
    try:
        old = datetime.now(timezone.utc) - timedelta(minutes=20)
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE chat_handoffs SET last_active_at=%s WHERE session_id=%s", (old, sid))
        assert handoff_check(sid, pending_timeout=900)["status"] == "expired"
        queue = client.get("/api/handoff/queue", headers=agent_headers).json()["handoffs"]
        assert sid not in {h["session_id"] for h in queue}, "过期会话应移出待接入队列"
    finally:
        _cleanup(sid)


def test_user_message_recover(client: TestClient, user_headers: dict[str, str], agent_headers: dict[str, str]):
    sid = _session_id()
    client.post(f"/api/handoff/{sid}", json={"reason": "投诉"}, headers=user_headers)
    client.post(f"/api/handoff/{sid}/resolve", json={"mode": "ai"}, headers=agent_headers)
    try:
        assert handoff_check(sid)["status"] == "resolved"
        chat_append(sid, "user", "我回来了，还在吗")
        # 修复（0f1bf51）：resolved 记录保留——已解决统计不丢，AI 正常对话
        assert handoff_check(sid)["status"] == "resolved", "已解决记录应保留（统计不丢）"
    finally:
        _cleanup(sid)


def test_handoff_status_api(client: TestClient, user_headers: dict[str, str], agent_headers: dict[str, str]):
    """状态查询接口：无记录 idle；触发后 pending；转回后 idle。"""
    sid = _session_id()
    try:
        r = client.get(f"/api/handoff/{sid}")
        assert r.status_code == 200 and r.json()["status"] == "idle"
        client.post(f"/api/handoff/{sid}", json={"reason": "测试"}, headers=user_headers)
        r2 = client.get(f"/api/handoff/{sid}")
        assert r2.json()["status"] == "pending"
        client.post(f"/api/handoff/{sid}/resolve", json={"mode": "ai"}, headers=agent_headers)
        r3 = client.get(f"/api/handoff/{sid}")
        assert r3.json()["status"] == "resolved"
    finally:
        _cleanup(sid)


def test_supervisor_handoff_active(client: TestClient, user_headers: dict[str, str], agent_headers: dict[str, str], runtime_supervisor):
    """人工接管中（active）：编排应返回转人工提示，不执行检索。"""
    from agent_base.agents.graph_supervisor import run_supervisor_graph

    sid = _session_id()
    try:
        r = client.post(f"/api/handoff/{sid}", json={}, headers=user_headers)
        assert r.status_code == 200
        r2 = client.post(f"/api/handoff/{sid}/reply", json={"content": "收到"}, headers=agent_headers)
        assert r2.status_code == 200

        plan = run_supervisor_graph("玻尿酸精华适合敏感肌吗", runtime_supervisor, {"top_k": 6}, session_id=sid)
        assert plan.get("mode") == "handoff"
        assert plan.get("handoff", {}).get("status") == "active"
        assert not plan.get("sources"), "接管中不应检索"
    finally:
        _cleanup(sid)


def test_classic_chain_handoff_block(client: TestClient, user_headers: dict[str, str], agent_headers: dict[str, str]):
    """经典链路（supervisor 默认关闭）下：人工接管中发问应被拦截，返回人工提示。"""
    sid = _session_id()
    client.post(f"/api/handoff/{sid}", json={"reason": "用户要求人工"}, headers=user_headers)
    client.post(f"/api/handoff/{sid}/reply", json={"content": "收到，我是客服"}, headers=agent_headers)
    try:
        with client.stream(
            "POST",
            "/api/ask/stream",
            json={"question": "玻尿酸精华适合敏感肌吗", "session_id": sid},
            headers=agent_headers,
        ) as resp:
            body = "".join(resp.iter_text())
        assert resp.status_code == 200
        assert "人工客服正在处理" in body, "经典链路下人工接管中应拦截并返回人工提示"
        assert "玻尿酸" not in body, "不应生成 AI 回答"
    finally:
        _cleanup(sid)


def test_admin_cannot_access_agent_api(client: TestClient, headers: dict[str, str]):
    """权限：管理员（admin）不能访问人工端接口（队列/回复），聊天内容保密。"""
    r = client.get("/api/handoff/queue", headers=headers)
    assert r.status_code == 403
    sid = _session_id()
    client.post(f"/api/handoff/{sid}", json={})
    try:
        r2 = client.post(f"/api/handoff/{sid}/reply", json={"content": "越权"}, headers=headers)
        assert r2.status_code == 403
        # v0.51: messages 需登录 token（买家轮询用，防未登录枚举）
        r3 = client.get(f"/api/sessions/{sid}/messages", headers=headers)
        assert r3.status_code == 200
    finally:
        _cleanup(sid)


def test_messages_requires_login(client: TestClient):
    """SEC-1：会话消息接口必须登录（无 token 枚举会话 → 403）。"""
    r = client.get("/api/sessions/no_such_session/messages")
    assert r.status_code == 403, "无 token 不应能拉取任意会话消息"


def test_handoff_trigger_requires_login(client: TestClient):
    """BUG-9：匿名不能触发转人工（防未登录刷队列）。"""
    r = client.post("/api/handoff/anonymous_sid", json={"reason": "x"})
    assert r.status_code == 403, "无 token 不应能触发转人工"


def test_session_messages_ownership(client: TestClient, user_headers: dict[str, str], agent_headers: dict[str, str]):
    """BUG-8：买家只能读自己（owner）的会话；客服可读任意会话。"""
    mine = _session_id()
    others = _session_id()
    chat_append(mine, "user", "我的会话", owner="user")
    chat_append(others, "user", "他人会话", owner="agent")
    try:
        r = client.get(f"/api/sessions/{mine}/messages", headers=user_headers)
        assert r.status_code == 200, r.text
        r2 = client.get(f"/api/sessions/{others}/messages", headers=user_headers)
        assert r2.status_code == 403, "买家不应读到他人会话"
        r3 = client.get(f"/api/sessions/{mine}/messages", headers=agent_headers)
        assert r3.status_code == 200, "客服应可读任意会话"
    finally:
        _cleanup(mine)
        _cleanup(others)


def test_handoff_stats_admin(client: TestClient, headers: dict[str, str]):
    """管理端状态统计：管理员可看（无聊天内容）。"""
    r = client.get("/api/handoff/stats", headers=headers)
    assert r.status_code == 200, r.text
    stats = r.json()["stats"]
    for key in ("total", "pending", "active", "expired", "closed", "resolved", "avg_waiting_secs"):
        assert key in stats


def test_chat_append_stores_sources(client: TestClient):
    """客服端 AI 召回依据：chat_append 落库 sources JSONB，chat_history 原样返回。"""
    sid = _session_id()
    sources = [
        {
            "rank": 1,
            "doc_id": "doc_001",
            "doc_name": "玻尿酸精华商品详情",
            "section": "功效",
            "score": 0.92,
            "preview": "双重分子量玻尿酸，大分子表层锁水、小分子深层渗透。",
        }
    ]
    try:
        chat_append(sid, "user", "这个精华适合干皮吗", owner="user")
        chat_append(sid, "assistant", "适合的，玻尿酸补水保湿。", owner="user", sources=sources)
        history = chat_history(sid, limit=10)
        assistant = [m for m in history if m["role"] == "assistant"]
        assert assistant and assistant[0]["sources"] == sources, f"sources 未原样返回: {assistant}"
        assert "sources" in history[0], "chat_history 每条消息都应带 sources 字段"
    finally:
        _cleanup(sid)


def test_session_messages_returns_sources(client: TestClient, user_headers: dict[str, str], agent_headers: dict[str, str]):
    """消息接口：assistant 消息返回 sources（客服端洞察面板数据源）。"""
    sid = _session_id()
    sources = [{"rank": 1, "doc_name": "FAQ", "section": "售后", "score": 0.8, "preview": "签收后 7 天内未拆封可退"}]
    try:
        chat_append(sid, "user", "能退吗", owner="user")
        chat_append(sid, "assistant", "支持七天无理由退货。", owner="user", sources=sources)
        r = client.get(f"/api/sessions/{sid}/messages", headers=user_headers)
        assert r.status_code == 200, r.text
        msgs = r.json()["messages"]
        assistant = [m for m in msgs if m["role"] == "assistant"]
        assert assistant and assistant[0].get("sources") == sources, f"接口未返回 sources: {assistant}"
        # 客服端同样可读
        r2 = client.get(f"/api/sessions/{sid}/messages", headers=agent_headers)
        assert r2.status_code == 200, r2.text
        assistant2 = [m for m in r2.json()["messages"] if m["role"] == "assistant"]
        assert assistant2 and assistant2[0].get("sources") == sources
    finally:
        _cleanup(sid)


def test_handoff_polish_endpoint(client: TestClient, agent_headers: dict[str, str]):
    """润色接口：合法入参返回 ok:true 或降级 ok:false（LLM 未配置时），不崩、不改输入。"""
    r = client.post("/api/handoff/polish", json={"text": "", "style": "polite"}, headers=agent_headers)
    assert r.status_code == 200 and r.json()["ok"] is False, r.text
    r2 = client.post("/api/handoff/polish", json={"text": "这个我处理不了", "style": "polite"}, headers=agent_headers)
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert "ok" in body
    if body["ok"]:
        assert isinstance(body["text"], str) and body["text"].strip()
        assert "这个我处理不了" in body["text"] or "处理" in body["text"], f"润色应保留原意: {body}"
    else:
        assert "error" in body, "降级时应返回 error 说明"
    # 非法 style 兜底 polite，不 500
    r3 = client.post("/api/handoff/polish", json={"text": "你好", "style": "weird"}, headers=agent_headers)
    assert r3.status_code == 200 and "ok" in r3.json()


def test_chat_history_order_same_timestamp(client: TestClient):
    """BUG-21：同时间戳批量落库消息按 id 兜底排序，客服端轮次标注/提问回查不串位。"""
    sid = _session_id()
    try:
        for i in range(5):
            chat_append(sid, "user", f"问题{i}", owner="user")
            chat_append(sid, "assistant", f"回答{i}")
        # 把所有消息时间戳设为相同，模拟批量落库（排序只能依赖 id 兜底）
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE chat_messages SET created_at = NOW() WHERE session_id=%s", (sid,))
        history = chat_history(sid, limit=100)
        contents = [m["content"] for m in history]
        roles = [m["role"] for m in history]
        assert contents[0] == "问题0" and contents[1] == "回答0", f"头部顺序错乱: {contents[:2]}"
        assert contents[-2] == "问题4" and contents[-1] == "回答4", f"尾部顺序错乱: {contents[-2:]}"
        # user/assistant 应严格交替
        for i, r in enumerate(roles):
            expect = "user" if i % 2 == 0 else "assistant"
            assert r == expect, f"第 {i} 条角色应为 {expect}，实际 {r}（顺序不稳定）"
    finally:
        _cleanup(sid)
