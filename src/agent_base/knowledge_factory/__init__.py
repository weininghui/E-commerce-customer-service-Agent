"""Knowledge factory — document tagging and review state machine (P14/P19).

P19 两级审核：
  agent 预审（启发式 + LLM，只建议不生效）→ 人工精审（approve / 打回）
  → approved 才允许入库（硬约束，见 storage.documents）。

标准来源（D5）：``configs/tagging.yaml``（doc_type 定义/信号/示例）；
预审 LLM 配置在 ``configs/app.yaml`` pre_review_llm 段，评审 prompt 在
``configs/prompts_ecommerce.yaml`` pre_review 段。YAML 是真相源，PG 是运行时状态。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

# 默认值（配置缺失时的兜底；优先读 configs/tagging.yaml 的 doc_types）
# 18 种，覆盖全电商知识库入库场景；合并了原 category_guide/guide/fashion_guide → guide
DOC_TYPES: tuple[str, ...] = (
    # 商品类
    "product_detail", "product_longdoc", "metadata_doc",
    # 指南类（合并原 category_guide / guide / fashion_guide）
    "guide",
    # 售后客服类
    "faq", "agent_script", "aftersale_flow",
    # 规则政策类
    "policy", "logistics", "membership",
    # 溯源科普类
    "origin_cert", "ingredient", "material",
    # 对比品牌类
    "comparison", "brand_intro",
    # 垂直类目类
    "nutrition_facts", "tech_spec", "install_guide",
)

STRATEGY_MAP: dict[str, list[str]] = {
    # 商品类
    "product_detail": ["parent_child", "summary_index"],
    "product_longdoc": ["summary_index", "parent_child"],
    "metadata_doc": ["self_query"],
    # 指南类
    "guide": ["summary_index"],
    # 售后客服类
    "faq": ["hypothetical_variants"],
    "agent_script": ["summary_index"],
    "aftersale_flow": ["default_vector"],
    # 规则政策类
    "policy": ["default_vector"],
    "logistics": ["default_vector"],
    "membership": ["default_vector"],
    # 溯源科普类
    "origin_cert": ["default_vector"],
    "ingredient": ["default_vector"],
    "material": ["default_vector"],
    # 对比品牌类
    "comparison": ["summary_index"],
    "brand_intro": ["default_vector"],
    # 垂直类目类
    "nutrition_facts": ["self_query"],
    "tech_spec": ["self_query"],
    "install_guide": ["summary_index"],
}

_HEURISTIC_SIGNALS: dict[str, list[str]] = {
    "faq": ["q:", "a:", "问题", "回答", "faq", "常见问题", "咨询"],
    "metadata_doc": ["参数", "规格", "成分表", "尺码表", "spf", "upf"],
    "guide": ["搭配", "穿搭", "指南", "攻略", "推荐", "怎么穿", "怎么选", "尺码", "色彩"],
    "origin_cert": ["产地", "溯源", "原料来源", "生产车间", "资质", "检测报告", "iso"],
    "product_longdoc": ["使用方法", "用户反馈", "种草", "卖点", "测评"],
    "ingredient": ["成分", "浓度", "耐受", "禁忌", "烟酰胺", "玻尿酸"],
    "policy": ["广告法", "合规", "宣称", "医疗用语", "话术", "红线"],
    "material": ["面料", "材质", "支数", "洗涤", "保养", "起球"],
    "agent_script": ["话术模板", "sop", "标准答复", "安抚话术", "场景话术"],
    "aftersale_flow": ["处理流程", "工单", "升级", "赔付", "sla", "退货政策"],
    "logistics": ["发货时效", "运费", "偏远地区", "跨境", "配送范围"],
    "membership": ["会员", "积分", "等级", "权益", "vip", "兑换"],
    "comparison": ["对比", "vs", "区别", "哪个好", "新款", "升级版"],
    "brand_intro": ["品牌故事", "品牌理念", "品牌历史", "创始人", "品牌定位"],
    "nutrition_facts": ["营养成分表", "配料表", "保质期", "热量", "蛋白质"],
    "tech_spec": ["技术参数", "认证", "3c认证", "接口", "功率", "处理器"],
    "install_guide": ["安装说明", "组装", "步骤", "使用说明", "安全注意事项"],
}


@dataclass
class DocTag:
    """Tag assigned to a document after pre-review and human audit.

    Attributes:
        doc_id: Source document identifier.
        doc_type: One of DOC_TYPES.
        strategy: List of applicable indexing strategies.
        reviewer: Who approved the tag.
        status: "pending_fine_review" | "approved" | "returned".
        review_round: 打回重审轮次（1 起）。
        first_review: agent 初审快照 {type, confidence, reasoning}。
        confidence: 初审置信度（兼容字段，冗余于 first_review）。
        reasoning: 初审理由（兼容字段）。
        reject_reason: 打回理由。
        reviewed_at: ISO-8601 精审时间。
    """

    doc_id: str
    doc_type: str = ""
    strategy: list[str] = field(default_factory=list)
    reviewer: str = ""
    status: str = "pending_fine_review"
    review_round: int = 1
    first_review: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reasoning: str = ""
    reject_reason: str = ""
    reviewed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转为 dict，便于 JSON 序列化与前端展示。"""
        return asdict(self)


# ── 配置 ────────────────────────────────────────────────────────────────────


def _load_tagging_cfg() -> dict[str, Any]:
    """加载 tagging.yaml（失败时安全返回空字典）。"""
    try:
        from agent_base.config import load_yaml
        return load_yaml("configs/tagging.yaml") or {}
    except Exception:
        return {}


def _config_doc_types(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """合并配置 doc_types 与内置默认值。"""
    types: dict[str, dict[str, Any]] = {}
    for t in DOC_TYPES:
        types[t] = {
            "description": "",
            "strategy": list(STRATEGY_MAP.get(t, [])),
            "signals": list(_HEURISTIC_SIGNALS.get(t, [])),
            "examples": [],
        }
    for t, spec in (cfg.get("doc_types") or {}).items():
        if t in types:
            types[t].update(spec)
        else:
            types[t] = dict(spec)
    return types


def _llm_cfg_from_config() -> dict[str, Any] | None:
    """预审 LLM 配置（D7：flash 低成本）。"""
    try:
        from agent_base.config import load_yaml

        cfg = (load_yaml("configs/app.yaml") or {}).get("pre_review_llm") or {}
    except Exception:
        cfg = {}
    if not cfg.get("provider"):
        return None
    return {
        "provider": cfg.get("provider", "langchain"),
        "model": cfg.get("model", "deepseek-v4-flash"),
        "base_url": cfg.get("base_url"),
        "api_key_env": cfg.get("api_key_env", "ANTHROPIC_AUTH_TOKEN"),
        "temperature": float(cfg.get("temperature", 0.0)),
        "timeout": cfg.get("timeout"),
    }


# ── 预审（agent 只建议，不生效） ─────────────────────────────────────────────


def pre_review_document(
    content_snippet: str,
    filename: str = "",
    llm_cfg: dict[str, Any] | None = None,
    prev_reject_reason: str = "",
) -> DocTag:
    """运行预审 Agent（启发式 + LLM）给出文档类型建议。

    LLM 只在 confidence >= 0.5 时采用，否则回退启发式。产出永远是
    ``pending_fine_review``，需要人工精审才生效。

    Args:
        content_snippet: 文档前约 2000 字符。
        filename: 原始文件名（启发式兜底用）。
        llm_cfg: LLM config；None 时读 ``configs/app.yaml`` 的 pre_review_llm 段。
        prev_reject_reason: 上次打回理由（重提时注入短期记忆）。

    Returns:
        含建议 doc_type / strategy / first_review 快照的 DocTag。
    """
    heuristic_type, heuristic_hits, heuristic_total = _heuristic_classify_with_hits(content_snippet, filename)
    heuristic_confidence = _heuristic_confidence(heuristic_hits, heuristic_total)

    cfg = llm_cfg if llm_cfg is not None else _llm_cfg_from_config()
    if cfg and cfg.get("provider", "none") not in {"none", "off", "false"}:
        try:
            llm_type, confidence, reasoning, suggest_action, reject_hint, risk_flags = _llm_classify(
                content_snippet, cfg, prev_reject_reason=prev_reject_reason
            )
            if llm_type and confidence >= 0.5:
                return DocTag(
                    doc_id="",
                    doc_type=llm_type,
                    strategy=list(STRATEGY_MAP.get(llm_type, ["default_vector"])),
                    status="pending_fine_review",
                    first_review={
                        "type": llm_type,
                        "strategy": list(STRATEGY_MAP.get(llm_type, ["default_vector"])),
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "suggest_action": suggest_action,
                        "reject_hint": reject_hint,
                        "risk_flags": risk_flags,
                        "source": "llm",
                    },
                    confidence=confidence,
                    reasoning=reasoning,
                )
        except Exception:
            pass

    return DocTag(
        doc_id="",
        doc_type=heuristic_type,
        strategy=list(STRATEGY_MAP.get(heuristic_type, ["default_vector"])),
        status="pending_fine_review",
        first_review={
            "type": heuristic_type,
            "strategy": list(STRATEGY_MAP.get(heuristic_type, ["default_vector"])),
            "confidence": heuristic_confidence,
            "reasoning": f"规则关键词分类（命中 {heuristic_hits} 个信号，LLM 不可用或置信度低，走规则兜底）",
            "suggest_action": "review",
            "reject_hint": "",
            "risk_flags": [],
            "source": "heuristic",
        },
        confidence=heuristic_confidence,
        reasoning=f"规则关键词分类（命中 {heuristic_hits} 个信号，LLM 不可用或置信度低，走规则兜底）",
    )


# ── 状态机：精审 / 打回 / 重提 ────────────────────────────────────────────────


def apply_tag(
    doc_id: str,
    tag: DocTag,
    reviewer: str = "admin",
  ) -> DocTag:
      """人工精审通过：status → approved，可入库。"""
      tag.doc_id = doc_id
      tag.reviewer = reviewer
      tag.status = "approved"
      tag.reviewed_at = datetime.now(timezone.utc).isoformat()
      _append_audit(tag, "approve", reviewer, {})
      return tag


def reject_tag(
    doc_id: str,
    tag: DocTag,
    reviewer: str = "admin",
    reason: str = "",
  ) -> DocTag:
      """人工打回：status → returned（原路打回，不入库；round 不变，重提时 +1）。"""
      tag.doc_id = doc_id
      tag.reviewer = reviewer
      tag.status = "returned"
      tag.reject_reason = reason or tag.reject_reason
      tag.reviewed_at = datetime.now(timezone.utc).isoformat()
      _append_audit(tag, "reject", reviewer, {"reason": reason})
      return tag


def _append_audit(tag: DocTag, action: str, reviewer: str, extra: dict[str, Any]) -> None:
    """P27e：在 first_review.audit.history 追加一条审核记录（不覆盖历史）。"""
    try:
        first = dict(tag.first_review or {})
        audit = dict(first.get("audit") or {})
        history = list(audit.get("history") or [])
        history.append({
            "action": action,
            "reviewer": reviewer,
            "round": tag.review_round,
            "ai": {
                "doc_type": first.get("type", ""),
                "confidence": float(first.get("confidence", 0.0)),
            },
            "at": datetime.now(timezone.utc).isoformat(),
            **extra,
        })
        audit["history"] = history
        first["audit"] = audit
        tag.first_review = first
    except Exception:
        pass


def submit_document_for_review(
    doc_id: str,
    doc_type: str = "",
    strategy: list[str] | None = None,
    reviewer: str = "",
    content: str = "",
) -> DocTag:
    """提交/重提文档进入待精审队列（数据中台 re-submit 契约入口）。

    returned 文档重提 → review_round + 1，保留打回理由供短期记忆；
    新文档 → round=1。P27：重提时读上轮记忆，注入新的 first_review。

    Args:
        doc_id: 文档标识。
        doc_type: 可选，预填 doc_type。
        strategy: 可选，预填 strategy。
        reviewer: 提交人（数据中台标识）。
        content: 文档全文（精审记忆注入用）。

    Returns:
        DocTag in pending_fine_review state.
    """
    existing: dict[str, Any] | None = None
    try:
        from agent_base.storage.pg import strategy_get
        existing = strategy_get(doc_id)
    except Exception:
        pass

    review_round = 1
    if existing and existing.get("status") == "returned":
        review_round = int(existing.get("review_round") or 1) + 1

    tag = DocTag(
        doc_id=doc_id,
        doc_type=doc_type or (existing or {}).get("doc_type", ""),
        strategy=strategy or (existing or {}).get("strategy") or [],
        reviewer=reviewer,
        status="pending_fine_review",
        review_round=review_round,
        first_review=(existing or {}).get("first_review") or {},
    )
    # P27：重提带短期记忆（上轮打回原因）注入 first_review，供评审复核
    try:
        from agent_base.storage.review_memory import load_memory
        prev_round = max(1, review_round - 1)
        memory = load_memory(doc_id, prev_round)
        if memory and memory.get("reject_reason"):
            tag.first_review = dict(tag.first_review or {})
            tag.first_review["prev_reject_reason"] = memory["reject_reason"]
            tag.first_review["memory_round"] = prev_round
            tag.first_review["content_preview"] = content[:2000]
    except Exception:
        pass
    persist_tag(tag)
    return tag


# ── 持久化 ───────────────────────────────────────────────────────────────────


def persist_tag(tag: DocTag) -> None:
    """将审核通过的标签持久化到 PG document_strategy 表。"""
    try:
        from agent_base.storage.pg import strategy_upsert
        strategy_upsert(
            doc_id=tag.doc_id,
            doc_type=tag.doc_type,
            strategy=tag.strategy,
            reviewer=tag.reviewer,
            status=tag.status,
            review_round=tag.review_round,
            first_review=tag.first_review,
            reject_reason=tag.reject_reason,
            reviewed_at=tag.reviewed_at or None,
        )
    except Exception:
        pass


def load_tags(doc_id: str | None = None) -> list[DocTag]:
    """从 PG 加载文档标签。"""
    try:
        from agent_base.storage.pg import strategy_list
        rows = strategy_list(doc_id=doc_id)
        return [
            DocTag(
                doc_id=r["doc_id"],
                doc_type=r.get("doc_type", ""),
                strategy=r.get("strategy", []),
                reviewer=r.get("reviewer", ""),
                status=r.get("status", "pending_fine_review"),
                review_round=int(r.get("review_round") or 1),
                first_review=r.get("first_review") or {},
                reject_reason=r.get("reject_reason", ""),
                reviewed_at=r.get("reviewed_at", ""),
            )
            for r in rows
        ]
    except Exception:
        return []


def get_tag(doc_id: str) -> DocTag | None:
    """获取单个标签（不存在返回 None）。"""
    tags = load_tags(doc_id=doc_id)
    return tags[0] if tags else None


def seed_legacy_tags() -> int:
    """D2: 存量文档批量 approved 迁移（启发式预填 doc_type，reviewer=migration）。

    Returns:
        Number of seeded tags.
    """
    seeded = 0
    try:
        from agent_base.storage.pg import list_document_ids, strategy_get, strategy_upsert
        for doc_id in list_document_ids():
            if strategy_get(doc_id) is not None:
                continue
            doc_type = _heuristic_classify(doc_id, "")
            if doc_type not in DOC_TYPES:
                doc_type = "product_detail"
            strategy_upsert(
                doc_id=doc_id,
                doc_type=doc_type,
                strategy=list(STRATEGY_MAP.get(doc_type, ["default_vector"])),
                reviewer="migration",
                status="approved",
                review_round=1,
                first_review={
                    "type": doc_type,
                    "confidence": 0.0,
                    "reasoning": "legacy auto-approve migration (D2)",
                    "source": "migration",
                },
            )
            seeded += 1
    except Exception:
        pass
    return seeded


# ── 内部辅助函数 ────────────────────────────────────────────────────────────


def _heuristic_classify(content: str, filename: str = "") -> str:
    """基于可配置信号的快速关键词分类（D5）。"""
    return _heuristic_classify_with_hits(content, filename)[0]


def _heuristic_classify_with_hits(content: str, filename: str = "") -> tuple[str, int, int]:
    """启发式分类，返回 (doc_type, 命中信号数, 该类型信号总数)。

    动态遍历 YAML 配置的所有 doc_type，按命中率最高者胜出；
    命中率用于计算置信度（0 命中 → 低置信默认分类）。
    """
    cfg = _load_tagging_cfg()
    types = _config_doc_types(cfg)
    text = (content + " " + filename).lower()

    candidates: list[tuple[str, int, int]] = []
    for t in types:
        signals = types[t].get("signals") or _HEURISTIC_SIGNALS.get(t, [])
        if not signals:
            continue
        hits = sum(1 for s in signals if s.lower() in text)
        candidates.append((t, hits, len(signals)))

    # 命中率最高者胜出（命中 ≥2 才有资格）；无任何特征命中 → 默认 product_detail
    best = max(candidates, key=lambda c: (c[1] / max(1, c[2]), c[1]))
    if best[1] >= 2:
        return best
    return "product_detail", 0, len(types.get("product_detail", {}).get("signals") or _HEURISTIC_SIGNALS.get("product_detail", []))


def _heuristic_confidence(hits: int, total_signals: int) -> float:
    """按命中率计算启发式置信度。

    命中率 0 → 0.0（无信号不显示误导性百分比）；命中率越高置信度越高，
    上限 0.8（规则兜底不应超过 LLM 下限）。
    """
    if hits <= 0 or total_signals <= 0:
        return 0.0
    ratio = hits / total_signals
    return round(min(0.8, 0.3 + 0.5 * ratio), 2)



def _llm_classify(
    content: str,
    cfg: dict[str, Any],
    prev_reject_reason: str = "",
) -> tuple[str, float, str, str, str, list[str]]:
    """LLM 评审：返回完整决策包（P27）。

    Returns:
        (doc_type, confidence, reasoning, suggest_action, reject_hint, risk_flags)。
        LLM 不可用/输出非法时返回 ("", 0.0, "", "review", "", [])。
    """
    from agent_base.llms import build_chat_model

    model = build_chat_model(
        provider=cfg.get("provider", "langchain"),
        model=cfg.get("model", "deepseek-v4-flash"),
        base_url=cfg.get("base_url"),
        api_key_env=cfg.get("api_key_env", "ANTHROPIC_AUTH_TOKEN"),
        temperature=float(cfg.get("temperature", 0.0)),
        timeout=cfg.get("timeout"),
    )
    if model is None:
        return ("", 0.0, "", "review", "", [])

    types = _config_doc_types(_load_tagging_cfg())
    lines: list[str] = []
    for t, spec in types.items():
        lines.append(f"- {t}: {spec.get('description', '')}")
        for ex in spec.get("examples", [])[:2]:
            lines.append(f"  例: {ex[:200]}")

    snippet = content[:1500]
    from agent_base.prompts import get_prompt
    system_prompt = get_prompt("pre_review", "system")
    user_template = get_prompt("pre_review", "user_template")
    prompt = user_template.format(snippet=snippet)
    if prev_reject_reason:
        prompt += f"\n\n上一轮打回原因（请重点核查是否已修复）：{prev_reject_reason}"
    prompt += "\n\n判定标准（类型定义与示例）：\n" + "\n".join(lines)

    # P26b + LCEL：直接以消息列表调用（不用 ChatPromptTemplate 解析 system 提示词，
    # 避免提示词里的 JSON 花括号 {\"type\"} 被当成模板变量导致 KeyError）
    from langchain_core.messages import HumanMessage, SystemMessage
    from agent_base.structured import PreReviewResult, parse_json_or_none

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
    result: dict[str, Any] | None = None
    try:
        structured = model.with_structured_output(PreReviewResult).invoke(messages)
        if structured is not None:
            result = structured.model_dump()
    except Exception:
        pass
    if result is None:
        try:
            resp = model.invoke(messages)
            text = str(getattr(resp, "content", "") or resp or "")
        except Exception:
            text = ""
        parsed = parse_json_or_none(text.strip())
        if isinstance(parsed, dict):
            result = parsed

    if not result:
        return ("", 0.0, "", "review", "", [])
    doc_type = str(result.get("type", ""))
    if doc_type not in DOC_TYPES:
        return ("", 0.0, "", "review", "", [])
    return (
        doc_type,
        float(result.get("confidence", 0.5)),
        str(result.get("reasoning", "")),
        str(result.get("suggest_action", "review")),
        str(result.get("reject_hint", "")),
        list(result.get("risk_flags") or []),
    )
