"""契约测试：PG 持久化异步任务队列（入队 / 原子认领 / 完成 / 回收 / handler）。

依赖真实 PG（与运行环境一致）。不创建 TestClient（避免应用 lifespan
worker 与本测试争抢任务），直接测 PG 原语 + execute_task 执行路径。
"""

from __future__ import annotations

from unittest.mock import patch

from agent_base.async_tasks import TASK_HANDLERS, execute_task, register_task_handler
from agent_base.storage.pg import (
    task_claim_next,
    task_enqueue,
    task_finish,
    task_get,
    task_list,
    task_reap_stale,
)


def _cleanup(task_ids: list[int]) -> None:
    from agent_base.storage.pg import _conn

    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM task_queue WHERE id = ANY(%s)", (list(task_ids),))
        conn.commit()


def test_task_queue_roundtrip():
    """入队 → 原子认领 → 完成 → 状态与结果可查。"""
    task_id = task_enqueue("echo", {"k": "v"}, owner="tester")
    assert task_id > 0
    try:
        claimed = task_claim_next("test-worker", task_types=["echo"])
        assert claimed is not None and int(claimed["id"]) == task_id
        assert claimed["status"] == "running"
        assert claimed["picked_by"] == "test-worker"
        assert claimed["payload"] == {"k": "v"}
        assert task_finish(task_id, status="done", result={"echo": {"k": "v"}})
        done = task_get(task_id)
        assert done is not None and done["status"] == "done"
        assert done["result"] == {"echo": {"k": "v"}}
        assert done["finished_at"] is not None
    finally:
        _cleanup([task_id])


def test_task_claim_skips_running_and_wrong_type():
    """认领只取 pending 且类型匹配的任务（多 worker 安全语义）。"""
    running_id = task_enqueue("echo", {"n": 1})
    image_id = task_enqueue("image_gen", {"prompt": "x"})
    try:
        claim1 = task_claim_next("w1", task_types=["echo"])
        assert claim1 is not None and int(claim1["id"]) == running_id
        # running 中的 echo 不可再被认领；类型限定下 image_gen 不可被认领
        assert task_claim_next("w2", task_types=["echo"]) is None
        claim2 = task_claim_next("w2", task_types=["image_gen"])
        assert claim2 is not None and int(claim2["id"]) == image_id
    finally:
        _cleanup([running_id, image_id])


def test_task_reap_stale():
    """僵死任务回收：running 超时重新置 pending。"""
    task_id = task_enqueue("echo", {"n": 2})
    try:
        task_claim_next("crash-worker", task_types=["echo"])
        # 直接把 started_at 拨回 10 分钟前，模拟 worker 崩溃
        from agent_base.storage.pg import _conn

        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE task_queue SET started_at = NOW() - INTERVAL '10 minutes' WHERE id=%s",
                (task_id,),
            )
            conn.commit()
        reaped = task_reap_stale(older_than_seconds=300)
        assert reaped >= 1
        re_claimed = task_claim_next("recovery-worker", task_types=["echo"])
        assert re_claimed is not None and int(re_claimed["id"]) == task_id
        assert re_claimed["attempts"] == 2  # 认领次数递增
    finally:
        _cleanup([task_id])


def test_execute_task_echo_and_unknown():
    """execute_task：echo 回显；未知类型返回 failed（不抛异常）。"""
    ok = execute_task({"task_type": "echo", "payload": {"a": 1}})
    assert ok["ok"] is True and ok["result"]["echo"] == {"a": 1}
    unknown = execute_task({"task_type": "no_such_type", "payload": {}})
    assert unknown["ok"] is False and "未注册" in unknown["error"]


def test_task_list_and_registry():
    """任务列表与可插拔注册表。"""
    task_id = task_enqueue("echo", {"list": True})
    try:
        items = task_list(status="pending", limit=10)
        assert any(int(t["id"]) == task_id for t in items)
    finally:
        _cleanup([task_id])
    assert "echo" in TASK_HANDLERS
    register_task_handler("custom_echo", lambda p: {"ok": True, "got": p})
    assert "custom_echo" in TASK_HANDLERS
    out = execute_task({"task_type": "custom_echo", "payload": {"x": 1}})
    assert out["ok"] is True and out["result"]["got"] == {"x": 1}


def test_claim_loop_survives_transient_errors():
    """BUG-18：瞬时错误（DB 抖动）不得杀死 worker 主循环。"""
    import asyncio

    from agent_base.async_tasks import _run_claim_loop

    calls = {"n": 0}

    def fake_claim(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("db blip")
        return None

    async def runner():
        stop = asyncio.Event()
        loop_task = asyncio.create_task(_run_claim_loop(stop, "t", 0.1, 1, 5))
        await asyncio.sleep(0.6)
        alive = not loop_task.done()
        stop.set()
        await asyncio.gather(loop_task, return_exceptions=True)
        return alive, calls["n"]

    with patch("agent_base.storage.pg.task_claim_next", side_effect=fake_claim):
        alive, n = asyncio.run(runner())
    assert alive is True
    assert n >= 2
