"""Agent 体系（主项目电商化）。

routing.py       — Agent 触发路由
tools_ecommerce.py — 电商工具链（商品/FAQ/合规/库存/订单/转人工）
"""

from agent_base.agents.routing import should_use_agent

__all__ = ["should_use_agent"]
