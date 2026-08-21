"""契约测试：观测底座（token_usage / tool_calls / eval_feedback）+ 埋点 + 统计。"""

from __future__ import annotations

from unittest.mock import patch

from agent_base.monitoring.usage import record_model_usage, record_tool_call, wrap_chat_model


class _FakeResp:
    response_metadata = {
        "token_usage": {
            "prompt_tokens": 120,
            "completion_tokens": 80,
            "total_tokens": 200,
        }
    }


class _FakeModel:
    model_name = "fake-model"

    def invoke(self, *args, **kwargs):
        return _FakeResp()

    def bind_tools(self, *args, **kwargs):
        return self


def test_llms_factory_wraps_model():
    from agent_base.llms import build_chat_model

    with patch("langchain_openai.ChatOpenAI") as mock_cls:
        mock_cls.return_value = _FakeModel()
        model = build_chat_model(provider="langchain", model="fake", tracking_agent="t", tracking_source="s")
    assert model is not None
    # 包装后 invoke 应触发埋点（落库失败静默，不抛错）
    resp = model.invoke("hi")
    assert resp is not None


def test_record_model_usage_swallows_errors():
    with patch("agent_base.storage.pg.record_token_usage", side_effect=Exception("db down")):
        record_model_usage(prompt_tokens=1)  # 不应抛异常


def test_record_tool_call_swallows_errors():
    with patch("agent_base.storage.pg.record_tool_call", side_effect=Exception("db down")):
        record_tool_call(tool_name="x")


def test_wrap_tool_records_call():
    from agent_base.monitoring.usage import wrap_tool

    class FakeTool:
        name = "fake_tool"

        def __init__(self):
            self.func = lambda *a, **k: "result"

    tool = FakeTool()
    wrapped = wrap_tool(tool, agent="test")
    with patch("agent_base.monitoring.usage.record_tool_call") as mock_record:
        assert wrapped.func(x=1) == "result"
    mock_record.assert_called_once()
    assert mock_record.call_args.kwargs["tool_name"] == "fake_tool"
    assert mock_record.call_args.kwargs["ok"] is True


def test_pg_stats_functions_swallow_db_errors():
    from agent_base.storage.pg import failure_stats, token_usage_stats, tool_calls_stats

    with patch("agent_base.storage.pg._conn", side_effect=Exception("no db")):
        assert token_usage_stats()["rows"] == []
        assert tool_calls_stats()["rows"] == []
        assert failure_stats()["by_module"] == []


def test_eval_feedback_functions_swallow_db_errors():
    from agent_base.storage.pg import eval_feedback_list, update_eval_feedback_status, upsert_eval_feedback

    with patch("agent_base.storage.pg._conn", side_effect=Exception("no db")):
        assert upsert_eval_feedback(failure_type="knowledge_gap") == 0
        assert eval_feedback_list() == []
        assert update_eval_feedback_status(1, "regressed") is False


def test_tracking_wrapper_delegates_bind_tools():
    model = wrap_chat_model(_FakeModel(), agent="a")
    bound = model.bind_tools(["tool"])
    assert bound is model._inner


def test_tracking_wrapper_returns_none_for_none():
    assert wrap_chat_model(None) is None


def test_tracking_wrapper_records_llm_failure():
    """LLM 调用失败必须留痕原因（面板要能看到"每次失败的原因"）。"""

    class _BoomModel:
        model_name = "boom-model"

        def invoke(self, *args, **kwargs):
            raise RuntimeError("request timeout")

    model = wrap_chat_model(_BoomModel(), agent="qa", source="chat")
    with patch("agent_base.monitoring.usage.record_model_usage") as mock_record:
        try:
            model.invoke("hi")
        except RuntimeError:
            pass
    mock_record.assert_called_once()
    assert mock_record.call_args.kwargs["ok"] is False
    assert "timeout" in mock_record.call_args.kwargs["error"]
    assert mock_record.call_args.kwargs["latency_ms"] >= 0


def test_recent_failure_events_swallows_db_errors():
    from agent_base.storage.pg import recent_failure_events

    with patch("agent_base.storage.pg._conn", side_effect=Exception("no db")):
        body = recent_failure_events()
    assert body["events"] == []
    assert body["total"] == 0
