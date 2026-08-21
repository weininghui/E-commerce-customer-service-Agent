"""契约测试：会话级销售接线层（graph 节点 / 状态字段 / 主链路上下文）。"""

from __future__ import annotations

from agent_base.graphs.state import RagState


def test_rag_state_declares_sales_field():
    annotations = RagState.__annotations__
    assert "sales" in annotations


def test_sales_node_writes_context():
    from agent_base.graphs.nodes import make_sales_node

    node = make_sales_node()
    out = node(
        {
            "question": "这个精华适合我吗，有点想买",
            "route": {
                "intent": "product_query",
                "sub_intent": "",
                "buying_signal": "buying",
                "objection_type": "none",
                "missing_info": ["skin_type"],
                "emotion": "neutral",
            },
            "conversation": {},
        }
    )
    assert out["sales"]["action"] == "clarify_requirements"
    assert out["sales"]["stage"] == "consult"


def test_sales_node_graceful_when_route_missing():
    from agent_base.graphs.nodes import make_sales_node

    out = make_sales_node()({"question": "你好", "conversation": {}})
    assert out["sales"]["action"] in {"answer", "handoff"}


def test_graph_builds_with_sales_node():
    from agent_base.graphs.graph import build_rag_graph

    graph = build_rag_graph(object(), llm_cfg={"provider": "none"})
    nodes = graph.get_graph().nodes
    assert "sales" in nodes
    assert "route" in nodes
