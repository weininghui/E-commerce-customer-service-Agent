"""观测底座：Token 用量 + 工具调用统一埋点。

- ``wrap_chat_model`` 包装 LLM 对象，拦截 invoke/ainvoke/stream/astream，抽取 usage 落库；
- ``record_tool_call`` 在工具执行层统一记录成功/失败/耗时。
所有埋点失败静默，不阻塞主流程。
"""

from __future__ import annotations

import time
from typing import Any, Callable

from langchain_core.runnables.base import Runnable

from agent_base.monitoring.logger import request_id_var


def _request_id() -> str:
    try:
        return request_id_var.get("-")
    except Exception:
        return "-"


def record_model_usage(
    *,
    agent: str = "",
    model: str = "",
    source: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    latency_ms: int = 0,
    ok: bool = True,
    error: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    """记录一次模型调用 Token 用量（失败静默）。

    ok=False 时附带失败原因（超时/重试耗尽/解析失败等），支撑失败事件明细面板。
    """
    try:
        from agent_base.storage.pg import record_token_usage

        record_token_usage(
            request_id=_request_id(),
            agent=agent,
            model=model,
            source=source,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            ok=ok,
            error=error,
            data=extra or {},
        )
    except Exception:
        pass


def record_tool_call(
    *,
    agent: str = "",
    tool_name: str,
    params: dict[str, Any] | None = None,
    ok: bool = True,
    error: str = "",
    latency_ms: int = 0,
    result_preview: str = "",
    session_id: str = "",
) -> None:
    """记录一次工具调用（失败静默）。"""
    try:
        from agent_base.storage.pg import record_tool_call as _pg_record

        _pg_record(
            request_id=_request_id(),
            session_id=session_id,
            agent=agent,
            tool_name=tool_name,
            params=params or {},
            ok=ok,
            error=error,
            latency_ms=latency_ms,
            result_preview=result_preview,
        )
    except Exception:
        pass


def _extract_usage_meta(response: Any) -> tuple[int, int, int]:
    """从 LangChain 响应中抽取 prompt/completion/total tokens。"""
    prompt = completion = total = 0
    try:
        meta = {}
        if hasattr(response, "response_metadata") and response.response_metadata:
            meta = response.response_metadata
        elif isinstance(response, dict):
            meta = response.get("response_metadata") or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", 0) or 0)
        if not total:
            total = prompt + completion
    except Exception:
        pass
    return prompt, completion, total


class _TrackingModel(Runnable):
    """包装 LangChain 模型，拦截同步/异步/流式调用并埋点。

    继承 Runnable（而非普通包装类）：保证 prompt | model | parser 的
    LCEL 管道组合可用（coerce_to_runnable 只接受 Runnable/callable/dict），
    否则受控生成 / LCEL 判官会因 TypeError 静默降级为模板兜底。
    """

    def __init__(self, inner: Any, *, agent: str = "", source: str = ""):
        super().__init__()
        self._inner = inner
        self._agent = agent
        self._source = source or "chat"
        self._model = ""
        try:
            self._model = str(getattr(inner, "model_name", "") or getattr(inner, "model", "") or "")
        except Exception:
            pass

    def _record_failure(self, start: float, exc: Exception) -> None:
        """LLM 调用失败留痕：原因 + 耗时 + 链路 ID（埋点失败静默，绝不掩盖原异常）。"""
        try:
            record_model_usage(
                agent=self._agent,
                model=self._model,
                source=self._source,
                ok=False,
                error=f"{type(exc).__name__}: {exc}"[:500],
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
        except Exception:
            pass

    # ---------- 同步 ----------
    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            resp = self._inner.invoke(*args, **kwargs)
        except Exception as exc:
            self._record_failure(start, exc)
            raise
        p, c, t = _extract_usage_meta(resp)
        record_model_usage(
            agent=self._agent,
            model=self._model,
            source=self._source,
            prompt_tokens=p,
            completion_tokens=c,
            total_tokens=t,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
        return resp

    def stream(self, *args: Any, **kwargs: Any):
        start = time.perf_counter()
        last: Any = None
        try:
            for chunk in self._inner.stream(*args, **kwargs):
                last = chunk
                yield chunk
        except Exception as exc:
            self._record_failure(start, exc)
            raise
        p, c, t = _extract_usage_meta(last)
        record_model_usage(
            agent=self._agent,
            model=self._model,
            source=self._source,
            prompt_tokens=p,
            completion_tokens=c,
            total_tokens=t,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    # ---------- 异步 ----------
    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            resp = await self._inner.ainvoke(*args, **kwargs)
        except Exception as exc:
            self._record_failure(start, exc)
            raise
        p, c, t = _extract_usage_meta(resp)
        record_model_usage(
            agent=self._agent,
            model=self._model,
            source=self._source,
            prompt_tokens=p,
            completion_tokens=c,
            total_tokens=t,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
        return resp

    async def astream(self, *args: Any, **kwargs: Any):
        start = time.perf_counter()
        last: Any = None
        try:
            async for chunk in self._inner.astream(*args, **kwargs):
                last = chunk
                yield chunk
        except Exception as exc:
            self._record_failure(start, exc)
            raise
        p, c, t = _extract_usage_meta(last)
        record_model_usage(
            agent=self._agent,
            model=self._model,
            source=self._source,
            prompt_tokens=p,
            completion_tokens=c,
            total_tokens=t,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    def bind_tools(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.bind_tools(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def wrap_chat_model(model: Any, *, agent: str = "", source: str = "") -> Any:
    """包装模型对象；model 为 None 时原样返回。"""
    if model is None:
        return None
    try:
        return _TrackingModel(model, agent=agent, source=source)
    except Exception:
        return model


def tracked_tool(*, agent: str = ""):
    """装饰器：包装 LangChain @tool 函数，自动记录调用。"""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            tool_name = getattr(fn, "name", None) or getattr(fn, "__name__", "tool")
            try:
                result = fn(*args, **kwargs)
                record_tool_call(
                    agent=agent,
                    tool_name=tool_name,
                    params=kwargs or {},
                    ok=True,
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    result_preview=str(result)[:500],
                )
                return result
            except Exception as exc:
                record_tool_call(
                    agent=agent,
                    tool_name=tool_name,
                    params=kwargs or {},
                    ok=False,
                    error=str(exc)[:500],
                    latency_ms=int((time.perf_counter() - start) * 1000),
                )
                raise

        wrapper.__name__ = getattr(fn, "__name__", "tool")
        wrapper.__doc__ = getattr(fn, "__doc__", None)
        return wrapper

    return decorator


def wrap_tool(tool: Any, *, agent: str = "") -> Any:
    """包装 LangChain StructuredTool，统一记录调用（成功/失败/耗时）。

    在工具绑定处调用：``tools = [wrap_tool(t, agent="worker") for t in TOOLS]``。
    """
    if tool is None:
        return tool
    try:
        orig_func = getattr(tool, "func", None)
        if orig_func is None:
            return tool
        tool_name = getattr(tool, "name", "") or "tool"

        def _tracked(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = orig_func(*args, **kwargs)
                record_tool_call(
                    agent=agent,
                    tool_name=tool_name,
                    params=kwargs or {},
                    ok=True,
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    result_preview=str(result)[:500],
                )
                return result
            except Exception as exc:
                record_tool_call(
                    agent=agent,
                    tool_name=tool_name,
                    params=kwargs or {},
                    ok=False,
                    error=str(exc)[:500],
                    latency_ms=int((time.perf_counter() - start) * 1000),
                )
                raise

        tool.func = _tracked
        return tool
    except Exception:
        return tool
