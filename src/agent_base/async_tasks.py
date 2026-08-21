"""PG 持久化异步任务队列 worker（知识流水线 / 商品图生成等耗时操作）。

设计：
- 队列真相源在 PG（task_queue 表，随主库备份、可审计），worker 通过
  FOR UPDATE SKIP LOCKED 原子认领，多 worker 可水平扩展；
- TASK_HANDLERS 为可插拔注册表：task_type -> 同步执行函数，新增任务
  类型只需注册一个 handler；
- worker 只认领注册过的任务类型（未知类型留给其他服务，避免吞任务）；
- 执行带超时护栏，失败落 error 不崩 worker；周期性回收僵死任务自愈。

启动方式：FastAPI lifespan 里 start_task_worker()（configs/app.yaml
tasks.worker_enabled 默认开启），测试环境可直接调用 execute_task 单测。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable


def _h_echo(payload: dict[str, Any]) -> dict[str, Any]:
    """测试用 echo 任务：原样回显 payload。"""
    return {"echo": payload}


def _h_knowledge_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    """知识生产流水线任务（采集 → 解析抽取 → 质检 → 发布）。"""
    from agent_base.knowledge_pipeline import run_knowledge_pipeline

    filename = str(payload.get("filename") or "async-upload.md")
    content = str(payload.get("content") or "")
    category = str(payload.get("category") or "异步任务")
    return run_knowledge_pipeline(filename=filename, content=content, category=category)


def _h_image_gen(payload: dict[str, Any]) -> dict[str, Any]:
    """文生图任务（Seedream，无密钥 mock 降级）。"""
    from agent_base.multimodal import generate_product_image

    prompt = str(payload.get("prompt") or "")
    if not prompt:
        return {"ok": False, "error": "prompt 不能为空"}
    return generate_product_image(prompt)


def _h_image_edit(payload: dict[str, Any]) -> dict[str, Any]:
    """图生图任务（Seedream images/edits，无密钥 mock 降级）。"""
    from agent_base.multimodal import edit_product_image

    prompt = str(payload.get("prompt") or "")
    image = payload.get("image") or None
    image_url = payload.get("image_url") or None
    if not prompt:
        return {"ok": False, "error": "prompt 不能为空"}
    if not image and not image_url:
        return {"ok": False, "error": "必须提供参考图（image 或 image_url）"}
    return edit_product_image(prompt, image=image, image_url=image_url)


def _index_media_document(row: dict[str, Any], description: str, ocr_text: str) -> bool:
    """图片解析文本入库（Phase 3）：单 chunk 直入向量库，metadata 带 image_urls。

    复用 ingest_document_from_chunks 的内部导入豁免（skip_tag_check=True）；
    doc_id 固定 media-<id>，重复解析幂等覆盖。失败静默返回 False（不阻塞任务）。
    """
    try:
        import hashlib

        from agent_base.storage.documents import ingest_document_from_chunks

        media_id = int(row.get("id") or 0)
        url = str(row.get("url") or "")
        doc_id = f"media-{media_id}"
        text = "\n".join([p for p in [description, ocr_text] if p]).strip()
        if not text:
            return False
        chunk_id = hashlib.sha256((doc_id + ":chunk0").encode("utf-8")).hexdigest()[:16]
        metadata = {
            "doc_id": doc_id,
            "media_id": media_id,
            "image_urls": [url] if url else [],
            "product_id": str(row.get("product_id") or ""),
            "source_type": str(row.get("source_type") or "upload"),
            "category": "图片知识库",
        }
        chunk = {"chunk_id": chunk_id, "text": text, "metadata": metadata}
        from agent_base.api.main import get_runtime

        runtime = get_runtime()
        ingest_document_from_chunks(
            doc_id=doc_id,
            content=text,
            chunks=[chunk],
            vector_store=runtime["vector_store"],
            category="图片知识库",
            metadata=metadata,
            skip_tag_check=True,
        )
        return True
    except Exception:
        return False


def _h_media_parse(payload: dict[str, Any]) -> dict[str, Any]:
    """图片解析任务（OCR/视觉理解 + 文本入库，Phase 2/3）。"""
    from agent_base.media_library import media_file_path
    from agent_base.multimodal import analyze_media_image
    from agent_base.storage.pg import media_document_get, media_document_update_parse

    media_id = int(payload.get("media_id") or 0)
    if not media_id:
        return {"ok": False, "error": "media_id 不能为空"}
    row = media_document_get(media_id)
    if not row:
        return {"ok": False, "error": "图片记录不存在"}
    path = media_file_path(str(row.get("url") or ""))
    result = analyze_media_image(
        image_path=str(path) if path else None,
        image_url=None if path else (str(row.get("url") or "") or None),
    )
    description = str(result.get("description") or "")
    ocr_text = str(result.get("ocr_text") or "")
    if not description:
        # mock/失败降级时不覆盖人工描述
        description = str(row.get("description") or "")
    media_document_update_parse(media_id, description=description, ocr_text=ocr_text)
    indexed = _index_media_document(row, description, ocr_text) if ocr_text.strip() else False
    return {
        "ok": True,
        "engine": result.get("engine", "mock"),
        "media_id": media_id,
        "description": description,
        "ocr_text_len": len(ocr_text),
        "indexed": indexed,
    }


def _h_media_parse_video(payload: dict[str, Any]) -> dict[str, Any]:
    """视频解析任务（ffmpeg 抽帧 + 视觉理解 + 文本入库，Phase 3 视频版）。"""
    from agent_base.media_library import media_file_path
    from agent_base.multimodal.video import analyze_video, index_video_document
    from agent_base.storage.pg import media_document_get, media_document_update_parse

    media_id = int(payload.get("media_id") or 0)
    if not media_id:
        return {"ok": False, "error": "media_id 不能为空"}
    row = media_document_get(media_id)
    if not row:
        return {"ok": False, "error": "媒体记录不存在"}
    path = media_file_path(str(row.get("url") or ""))
    result = analyze_video(
        video_path=str(path) if path else None,
        video_url=None if path else (str(row.get("url") or "") or None),
    )
    description = str(result.get("description") or "")
    scenes_text = str(result.get("scenes_text") or "")
    if not description:
        # mock/失败降级时不覆盖人工描述
        description = str(row.get("description") or "")
    url = str(row.get("url") or "")
    duration = int(result.get("duration_sec") or 0)
    media_document_update_parse(
        media_id,
        description=description,
        ocr_text=scenes_text,
        video_urls=[url] if url else None,
        poster_url=str(row.get("poster_url") or ""),
        duration_sec=duration,
        parse_type="video",
    )
    indexed = index_video_document(row, description, scenes_text, duration) if scenes_text.strip() else False
    return {
        "ok": True,
        "engine": result.get("engine", "mock"),
        "media_id": media_id,
        "description": description,
        "scenes_len": len(scenes_text),
        "frame_count": result.get("frame_count", 0),
        "duration_sec": duration,
        "indexed": indexed,
        "note": str(result.get("note") or ""),
    }


TASK_HANDLERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "echo": _h_echo,
    "knowledge_pipeline": _h_knowledge_pipeline,
    "image_gen": _h_image_gen,
    "image_edit": _h_image_edit,
    "media_parse": _h_media_parse,
    "media_parse_video": _h_media_parse_video,
}


def register_task_handler(task_type: str, handler: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    """可插拔注册：注册新的任务类型 handler（供业务扩展）。"""
    TASK_HANDLERS[task_type] = handler


def execute_task(task: dict[str, Any], timeout_s: float = 300.0) -> dict[str, Any]:
    """同步执行单个任务（测试与 worker 共用入口）。

    Args:
        task: task_get/task_claim_next 返回的任务 dict。
        timeout_s: 执行超时（秒），超时返回 failed 结果而非抛异常。

    Returns:
        {"ok": bool, "result": dict, "error": str}。
    """
    task_type = str(task.get("task_type") or "")
    payload = dict(task.get("payload") or {})
    handler = TASK_HANDLERS.get(task_type)
    if handler is None:
        return {"ok": False, "result": {}, "error": f"未注册的任务类型: {task_type}"}
    started = time.perf_counter()
    try:
        result = handler(payload)
        latency_ms = int((time.perf_counter() - started) * 1000)
        if not isinstance(result, dict):
            result = {"output": result}
        result.setdefault("latency_ms", latency_ms)
        return {"ok": bool(result.get("ok", True)), "result": result, "error": ""}
    except Exception as exc:
        return {"ok": False, "result": {}, "error": f"{type(exc).__name__}: {exc}"}


async def _run_claim_loop(
    stop_event: asyncio.Event,
    worker_name: str,
    poll_interval: float,
    concurrency: int,
    timeout_s: float,
) -> None:
    """认领 → 执行 → 收尾 主循环（带并发护栏与僵死回收）。"""
    from agent_base.storage.pg import task_claim_next, task_finish, task_reap_stale

    semaphore = asyncio.Semaphore(concurrency)
    last_reap = time.monotonic()

    async def _process(task: dict[str, Any]) -> None:
        async with semaphore:
            try:
                outcome = await asyncio.wait_for(
                    asyncio.to_thread(execute_task, task, timeout_s),
                    timeout=timeout_s + 10,
                )
                status = "done" if outcome["ok"] else "failed"
                task_finish(int(task["id"]), status=status, result=outcome["result"], error=outcome["error"])
            except (TimeoutError, asyncio.TimeoutError):
                task_finish(int(task["id"]), status="failed", error=f"任务执行超时（>{timeout_s}s）")
            except Exception as exc:
                task_finish(int(task["id"]), status="failed", error=f"{type(exc).__name__}: {exc}"[:500])

    pending: set[asyncio.Task] = set()
    try:
        while not stop_event.is_set():
            # BUG-18 修复：瞬时错误（DB 抖动等）不得杀死 worker——
            # 记日志后进入下一轮轮询，保证队列长期自愈
            try:
                # 每 60s 回收一次僵死任务（worker 崩溃自愈）
                if time.monotonic() - last_reap > 60:
                    try:
                        task_reap_stale(older_than_seconds=max(60, int(timeout_s * 2)))
                    except Exception:
                        pass
                    last_reap = time.monotonic()

                claimed = 0
                while len(pending) < concurrency * 4:
                    task = await asyncio.to_thread(task_claim_next, worker_name, list(TASK_HANDLERS.keys()))
                    if task is None:
                        break
                    claimed += 1
                    pending.add(asyncio.create_task(_process(task)))
                # 已完成任务回收句柄，避免集合无限增长
                if pending:
                    done, pending = await asyncio.wait(
                        pending, timeout=poll_interval, return_when=asyncio.FIRST_COMPLETED
                    )
                    for finished in done:
                        exc = finished.exception()
                        if exc is not None:
                            try:
                                from agent_base.monitoring.logger import log_event
                                log_event("ERROR", "task_worker", "task_runner_crashed", {"error": str(exc)[:200]})
                            except Exception:
                                pass
                else:
                    # BUG-18 根因：wait_for 超时会抛 TimeoutError 杀死主循环，
                    # 空闲等待必须用 sleep
                    await asyncio.sleep(poll_interval)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                try:
                    from agent_base.monitoring.logger import log_event
                    log_event("ERROR", "task_worker", "claim_loop_error", {"error": str(exc)[:200]})
                except Exception:
                    pass
                await asyncio.sleep(poll_interval)
    except asyncio.CancelledError:
        pass
    finally:
        for t in list(pending):
            t.cancel()


def start_task_worker(
    stop_event: asyncio.Event | None = None,
    worker_name: str = "worker-1",
    poll_interval: float = 2.0,
    concurrency: int = 2,
    timeout_s: float = 300.0,
) -> tuple[asyncio.Task, asyncio.Event]:
    """启动后台任务 worker，返回 (task, stop_event)。

    Args:
        stop_event: 外部停止信号；None 时新建。
        worker_name: worker 标识（写入 task_queue.picked_by）。
        poll_interval: 空闲轮询间隔（秒）。
        concurrency: 并发执行上限。
        timeout_s: 单任务执行超时（秒）。

    Returns:
        (asyncio.Task, asyncio.Event)：取消 task 或 set event 均可优雅停止。
    """
    if stop_event is None:
        stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    worker_task = loop.create_task(
        _run_claim_loop(stop_event, worker_name, poll_interval, concurrency, timeout_s),
        name=f"task-worker-{worker_name}",
    )
    return worker_task, stop_event
