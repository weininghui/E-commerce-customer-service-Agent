"""官方 LangChain Agent 构建模块（面试主线：create_agent + 中间件）。

边界（用户定稿）：
- 意图识别（intent_agent）/ 改写（enrich_agent）保留自研——项目核心，不动。
- 其余子 Agent 全部用官方 LangChain/LangGraph API 创建：
  - ``create_agent``（LangChain 官方，LLM + 工具循环）
  - ``middleware``（官方中间件：PII 脱敏 / 摘要压缩 / HITL 转人工）
  - 现有 ``@tool`` 工具（tools_ecommerce / tools_memory）直接复用

这样项目里能看到完整的官方用法，同时不破坏已验收的自研 RAG 逻辑。
"""

from __future__ import annotations

from typing import Any

from agent_base.agents.tools_ecommerce import check_stock, get_product_info
from agent_base.agents.tools_memory import (
    make_delete_memory_tool,
    make_retrieve_memory_tool,
    make_save_memory_tool,
    make_update_memory_tool,
)


def _build_llm(llm_cfg: dict[str, Any] | None) -> Any:
    """构建官方 ChatModel（复用统一工厂，DeepSeek OpenAI 兼容）。"""
    from agent_base.llms import build_chat_model

    cfg = llm_cfg or {}
    return build_chat_model(
        provider=cfg.get("provider", "langchain"),
        model=cfg.get("model", "deepseek-v4-pro"),
        base_url=cfg.get("base_url"),
        api_key_env=cfg.get("api_key_env", "ANTHROPIC_AUTH_TOKEN"),
        temperature=float(cfg.get("temperature", 0.1)),
    )


def build_official_worker_agent(llm_cfg: dict[str, Any] | None = None) -> Any:
    """官方 create_react_agent：LLM 自主决定调用哪个工具（ReAct 循环）。

    工具：库存 / 商品信息 + 长期记忆 4 件套（复用现有 @tool）。
    中间件：PII 脱敏（邮箱/手机号不出现在回复）、摘要压缩。
    订单/物流等敏感信息不提供自服务查询，引导转人工核实。

    Returns:
        编译后的官方 agent（CompiledStateGraph）；LLM 不可用返回 None。
    """
    from langchain.agents import create_agent
    from langchain.agents.middleware import (
        PIIMiddleware,
        SummarizationMiddleware,
    )
    from agent_base.prompts import get_prompt

    model = _build_llm(llm_cfg)
    if model is None:
        return None

    # P26 单一口径：人设/铁律从 prompts_ecommerce.yaml 的 qa.system 加载
    # （唯一真相源），代码只保留 YAML 缺失时的兜底。
    base_prompt = get_prompt(
        "qa",
        "system",
        default="你是鹿屿好物电商客服小满——知性、有情绪价值的购物顾问。",
    )

    tools = [
        check_stock,
        get_product_info,
        make_save_memory_tool(),
        make_retrieve_memory_tool(),
        make_update_memory_tool(),
        make_delete_memory_tool(),
    ]
    from agent_base.monitoring.usage import wrap_tool

    tools = [wrap_tool(t, agent="official_worker") for t in tools]
    middleware = [
        # 官方中间件 1：PII 脱敏——回复里不暴露邮箱/手机号（apply_to_output=True 兜底）
        PIIMiddleware(pii_type="email", strategy="redact", apply_to_output=True),
        # 官方中间件 2：摘要压缩——长对话自动压缩，控制 token 成本
        # DeepSeek 无内置 model profile，用绝对 token 阈值（64K 窗口的 80%）
        SummarizationMiddleware(model=model, trigger=("tokens", 52000), keep=("messages", 20)),
    ]
    # 官方长期记忆：BaseStore 适配器包装 user_memories 表，
    # create_agent(store=...) 让 agent 自动读写用户画像（跨会话）。
    from agent_base.storage.memory_store import UserMemoryStore

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            f"{base_prompt}\n\n"
            "【工具使用】库存/商品信息用工具查真实数据；涉及记忆先保存用户偏好。"
            "【隐私红线】订单/物流/退款等涉及个人敏感信息的内容，不要编造状态，"
            "请引导用户点击界面上的『转人工』由人工客服核实身份后处理。"
        ),
        middleware=middleware,
        store=UserMemoryStore(),
        name="official_worker_agent",
    )
def build_official_generate_agent(llm_cfg: dict[str, Any] | None = None) -> Any:
    """官方 create_agent：受控生成（证据 + 记忆 + 人设，零工具 + PII 中间件）。

    对应原 generate_agent 的 LLM 路径；意图/改写仍走自研。
    """
    from langchain.agents import create_agent
    from langchain.agents.middleware import PIIMiddleware
    from agent_base.prompts import get_prompt

    model = _build_llm(llm_cfg)
    if model is None:
        return None
    base_prompt = get_prompt(
        "qa",
        "system",
        default="你是鹿屿好物电商客服小满——知性、有情绪价值的购物顾问。",
    )
    # 官方中间件：PII 脱敏——回复不暴露邮箱/卡号/URL（apply_to_output=True 兜底输出层）
    middleware = [
        PIIMiddleware(pii_type="email", strategy="redact", apply_to_output=True),
        PIIMiddleware(pii_type="credit_card", strategy="redact", apply_to_output=True),
        PIIMiddleware(pii_type="url", strategy="redact", apply_to_output=True),
    ]
    return create_agent(
        model=model,
        tools=None,
        system_prompt=base_prompt,
        middleware=middleware,
        name="official_generate_agent",
    )
