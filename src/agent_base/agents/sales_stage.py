"""会话级导购状态机（v2）。

把意图识别从"单轮分类"升级为"会话级理解"：每个会话维护购买阶段，
按（意图, 子意图, 购买信号, 异议类型, 缺失需求）决定下一阶段与导购动作。

阶段流转：none → consult → evaluate → hesitate → close → after。

触发边界（强导购但不打扰）：
- 首轮不主动推销：普通商品咨询只回答，不切导购；
- 只有商品/价格/推荐类意图且透出购买信号或异议（或用户主动求推荐）才进入；
- 闲聊、售后、促销咨询永不推销；情绪激动直接转人工。

阶段持久化：Redis（``sales:stage:{session_id}``，TTL 24h），
Redis 不可用时静默降级到进程内内存，不阻塞对话链路。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

STAGE_NONE = "none"
STAGE_CONSULT = "consult"
STAGE_EVALUATE = "evaluate"
STAGE_HESITATE = "hesitate"
STAGE_CLOSE = "close"
STAGE_AFTER = "after"

STAGES = (STAGE_NONE, STAGE_CONSULT, STAGE_EVALUATE, STAGE_HESITATE, STAGE_CLOSE, STAGE_AFTER)

# 商品类意图：只有这类问题才可能进入导购状态机
SALES_INTENTS = {
    "product_query",
    "fashion_query",
    "price_query",
    "recommendation",
    "comparison",
    "size_recommendation",
}

NON_SALES_INTENTS = {"aftersale"}

ACTIONS = (
    "answer",
    "clarify_requirements",
    "recommend",
    "objection_handle",
    "close_attempt",
    "cross_sell",
    "handoff",
)

# 销售动作：只有这些动作才注入导购策略/话术（其余一律纯回答，守住不推销红线）
SALES_ACTIONS = {
    "clarify_requirements",
    "recommend",
    "objection_handle",
    "close_attempt",
    "cross_sell",
}

_CONFIRM_PATTERNS = (
    "好吧", "就它", "就这个", "就要", "下单", "要了", "可以", "行吧",
    "买了", "拍下", "来一瓶", "来一件", "入手", "确定要",
)

_GIVE_UP_PATTERNS = (
    "不要了", "算了", "先不", "不买", "再看看别的", "不用了", "不需要",
)

_CROSS_SELL_PATTERNS = ("搭配", "配什么", "还缺", "需要配", "组合", "一套", "怎么配")

_SKIN_KEYWORDS = ("肤质", "干皮", "油皮", "敏感肌", "混合皮", "中性皮", "痘痘肌", "混油", "混干")
_BUDGET_KEYWORDS = ("预算", "价位", "多少钱", "价格", "入门", "中端", "高端", "三百", "百元", "千元", "以内")
_SCENE_KEYWORDS = ("通勤", "约会", "日常", "秋冬", "夏天", "送礼", "送人", "场合", "场景", "上班")
_SIZE_KEYWORDS = ("尺码", "身高", "体重", "腰围", "胸围", "肩宽")
_PRODUCT_KEYWORDS = (
    "精华", "面霜", "眼霜", "洁面", "防晒", "面膜", "水乳", "润肤油", "身体乳",
    "T恤", "裤子", "裙子", "衬衫", "外套", "卫衣", "针织", "毛衣", "帽子",
)

_STAGE_TTL = int(os.getenv("SALES_STAGE_TTL", "86400"))
_REDIS_KEY = "sales:stage:{session_id}"
_OFFER_REDIS_KEY = "sales:media_offer:{session_id}"
_memory_store: dict[str, str] = {}
_offer_memory: dict[str, bool] = {}
_redis_client: Any | None = None
_redis_checked = False


@dataclass(slots=True)
class SalesDecision:
    """一次会话理解的输出：目标阶段 + 导购动作 + 原因。"""

    stage: str
    action: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化字典。"""
        return {"stage": self.stage, "action": self.action, "reason": self.reason}


def decide_sales_step(
    route: dict[str, Any],
    current_stage: str = STAGE_NONE,
    *,
    question: str = "",
    turn_count: int = 1,
    clarify_count: int = 0,
    profile: dict[str, Any] | None = None,
) -> SalesDecision:
    """按规则表决定下一阶段与导购动作（确定性，不依赖 LLM）。

    Args:
        route: 会话理解结果（intent/sub_intent/buying_signal/objection_type/
            missing_info/emotion）。
        current_stage: 会话当前购买阶段。
        question: 用户当前问题（促单/放弃模式识别用）。
        turn_count: 会话内商品类轮次数（保留参数，供后续成本控制扩展）。
        clarify_count: 已主动问过需求的次数（最多 2 次，避免堆问题）。
        profile: 用户画像字典（skin_type/price_band/size 等），用于消解缺失需求，避免重复挖需。

    Returns:
        SalesDecision：目标阶段 + 动作 + 原因。
    """
    intent = str(route.get("intent") or "general_qa")
    sub_intent = str(route.get("sub_intent") or "")
    signal = str(route.get("buying_signal") or "normal")
    missing = _resolve_missing_with_profile(
        list(route.get("missing_info") or []),
        profile or {},
    )
    emotion = str(route.get("emotion") or "neutral")

    # 0. 情绪激动：安抚 + 转人工（售前售后一律优先）
    if emotion == "anger":
        return SalesDecision(current_stage or STAGE_NONE, "handoff", "情绪激动，优先安抚并转人工")

    # 0.5 购买确认/放弃信号：与具体意图无关（避免"好吧，就它了"落 general_qa 被当闲聊）
    if current_stage in (STAGE_CONSULT, STAGE_EVALUATE, STAGE_HESITATE, STAGE_CLOSE):
        if _has_give_up(question):
            return SalesDecision(STAGE_NONE, "answer", "用户明确放弃，尊重并留口子")
        if _has_confirm(question):
            if current_stage == STAGE_CLOSE:
                return SalesDecision(STAGE_AFTER, "answer", "已确认购买，转入售后/关怀")
            return SalesDecision(STAGE_CLOSE, "close_attempt", "购买确认，进入促单/收尾")

    # 1. 闲聊 / 泛问答：永不推销
    if sub_intent == "chat" or intent == "general_qa":
        return SalesDecision(current_stage or STAGE_NONE, "answer", "闲聊/泛问答不推销")

    # 2. 售后：处理问题不是卖货；若正处于销售阶段则切到售后阶段
    if intent in NON_SALES_INTENTS:
        stage = STAGE_AFTER if current_stage != STAGE_NONE else STAGE_NONE
        return SalesDecision(stage, "answer", "售后/物流类问题不推销")

    # 3. 促销咨询（活动/优惠/满减）：信息型，直接回答不催单
    if intent == "promotion":
        return SalesDecision(current_stage or STAGE_NONE, "answer", "促销信息咨询直接回答")

    # 4. 非商品类意图（其余 general_qa 等）：不切导购
    if intent not in SALES_INTENTS:
        return SalesDecision(current_stage or STAGE_NONE, "answer", "非商品类意图不切导购")

    # 4.5 售后阶段后用户又透出购买意向 → 重新进入导购（新品旅程开始）
    if current_stage == STAGE_AFTER and signal == "buying":
        current_stage = STAGE_NONE

    # 5. 首轮（阶段 none）：只有购买信号/异议/求推荐才进入
    if current_stage == STAGE_NONE:
        if signal == "objection":
            return SalesDecision(STAGE_CONSULT, "objection_handle", "首轮即异议，先处理顾虑")
        if signal == "buying":
            if sub_intent == "price_inquiry":
                return SalesDecision(STAGE_CONSULT, "answer", "询价先答价格，价格本身就是购买信号")
            if missing and clarify_count < 2:
                return SalesDecision(STAGE_CONSULT, "clarify_requirements", "有购买意向，先确认需求")
            return SalesDecision(STAGE_CONSULT, "recommend", "购买意向明确，直接给方案")
        if intent == "recommendation" or sub_intent == "recommend_request":
            if missing and clarify_count < 2:
                return SalesDecision(STAGE_CONSULT, "clarify_requirements", "用户求推荐，先问需求")
            return SalesDecision(STAGE_CONSULT, "recommend", "用户求推荐，直接给方案")
        return SalesDecision(STAGE_NONE, "answer", "普通商品咨询不主动推销")

    # 6. 咨询阶段：确认需求 → 给方案；出现异议则进入犹豫
    if current_stage == STAGE_CONSULT:
        if signal == "objection":
            return SalesDecision(STAGE_HESITATE, "objection_handle", "咨询中出现异议")
        effective_missing = _resolved_missing(missing, question)
        if signal == "buying" or _provides_requirements(question):
            if effective_missing and clarify_count < 2:
                return SalesDecision(STAGE_CONSULT, "clarify_requirements", "继续确认需求")
            return SalesDecision(STAGE_EVALUATE, "recommend", "需求已确认，进入方案推荐")
        return SalesDecision(STAGE_CONSULT, "answer", "继续回答商品细节")

    # 7. 评估/比价阶段：无异议继续答疑；购买信号明确则促单
    if current_stage == STAGE_EVALUATE:
        if signal == "objection":
            return SalesDecision(STAGE_HESITATE, "objection_handle", "评估中出现异议")
        if sub_intent == "recommend_request":
            return SalesDecision(STAGE_EVALUATE, "recommend", "用户再次求推荐，直接给方案")
        if any(p in (question or "") for p in _CROSS_SELL_PATTERNS):
            return SalesDecision(STAGE_EVALUATE, "cross_sell", "搭配/组合咨询，直接给搭配建议")
        if signal == "buying" or _has_confirm(question):
            return SalesDecision(STAGE_CLOSE, "close_attempt", "购买信号明确，进入促单")
        return SalesDecision(STAGE_EVALUATE, "answer", "继续对比/答疑")

    # 8. 犹豫/异议阶段：异议未消继续处理；顾虑解除则促单；明确放弃则尊重
    if current_stage == STAGE_HESITATE:
        if _has_give_up(question):
            return SalesDecision(STAGE_NONE, "answer", "用户明确放弃，尊重并留口子")
        if signal == "objection":
            return SalesDecision(STAGE_HESITATE, "objection_handle", "异议未消除，继续处理")
        if _has_confirm(question) or signal == "buying":
            return SalesDecision(STAGE_CLOSE, "close_attempt", "顾虑解除，进入促单")
        return SalesDecision(STAGE_HESITATE, "objection_handle", "继续追问顾虑，给明确建议")

    # 9. 促单阶段：确认购买 → 售后阶段；临门异议 → 回犹豫；否则自然连带推荐
    if current_stage == STAGE_CLOSE:
        if _has_confirm(question):
            return SalesDecision(STAGE_AFTER, "answer", "已确认购买，转入售后/关怀")
        if signal == "objection":
            return SalesDecision(STAGE_HESITATE, "objection_handle", "临门一脚出现异议")
        return SalesDecision(STAGE_CLOSE, "cross_sell", "需求已匹配，自然带出搭配")

    # 10. 售后/已购阶段：正常处理，不硬推销
    return SalesDecision(STAGE_AFTER, "answer", "售后/已购阶段正常处理")


def _has_confirm(question: str) -> bool:
    q = (question or "").strip().lower()
    return any(p in q for p in _CONFIRM_PATTERNS)


def _has_give_up(question: str) -> bool:
    q = (question or "").strip().lower()
    return any(p in q for p in _GIVE_UP_PATTERNS)


def _provides_requirements(question: str) -> bool:
    """用户消息是否补充了需求信息（肤质/预算/场景/尺码/商品）。"""
    q = (question or "").strip()
    return any(
        marker in q
        for marker in (
            *_SKIN_KEYWORDS,
            *_BUDGET_KEYWORDS,
            *_SCENE_KEYWORDS,
            *_SIZE_KEYWORDS,
            *_PRODUCT_KEYWORDS,
        )
    )


def _resolved_missing(missing: list[str], question: str) -> list[str]:
    """去掉用户本轮已补充的缺失需求字段。"""
    q = (question or "").strip()
    coverage = {
        "skin_type": _SKIN_KEYWORDS,
        "budget": _BUDGET_KEYWORDS,
        "scene": _SCENE_KEYWORDS,
        "size": _SIZE_KEYWORDS,
        "product": _PRODUCT_KEYWORDS,
    }
    return [key for key in missing if not any(k in q for k in coverage.get(key, ()))]


_PROFILE_MISSING_MAP: dict[str, tuple[str, ...]] = {
    "skin_type": ("skin_type",),
    "budget": ("price_band", "budget"),
    "size": ("size",),
    "scene": ("style", "scene"),
}


def _resolve_missing_with_profile(
    missing: list[str],
    profile: dict[str, Any],
) -> list[str]:
    """画像已覆盖的需求字段从缺失列表中移除（避免重复追问）。"""
    if not missing or not profile:
        return missing
    return [
        key
        for key in missing
        if not any(
            profile.get(candidate) not in (None, "", [], {})
            for candidate in _PROFILE_MISSING_MAP.get(key, ())
        )
    ]


# ── 话术指导（注入受控生成层） ──

STAGE_GUIDES: dict[str, str] = {
    "clarify_requirements": (
        "【需求确认】最多问 1-2 个问题（肤质/预算/尺码/使用场景），"
        "用一句话轻问，不要堆问题、不要客套；问完结合资料给一句建议。"
    ),
    "recommend": (
        "【方案推荐】用标准 Markdown 组织：## 推荐 小标题 + - 要点，给 2-3 个选项，"
        "每个一句理由（≤3 行/块），讲清适合场景与卖点；不贬低竞品、不承诺功效，"
        "数值以资料原文为准，不写废话。讲解后若本会话尚未提过，可轻提一句"
        "“要不要我发几张实拍图/视频给你参考？”——只提一次，用户要看再发。"
    ),
    "objection_handle": (
        "【异议处理】先共情再回应：嫌贵→重述价值（面料/配方/使用时长）；"
        "犹豫→帮 TA 理清真实需求并给明确建议；怕不合适→售后保障+尺码/肤质确认。"
        "用 1-2 句讲清，不写客套废话。"
    ),
    "close_attempt": (
        "【促单】需求匹配后给明确行动建议（如“今天就可以安排”），"
        "一句话讲清，自然不催促；用户说再考虑时尊重，留一句可回头找我的话。"
        "若本会话尚未提过，可顺带一句“需要发实拍图/视频给你参考吗？”，只看不回传媒体。"
    ),
    "cross_sell": (
        "【连带推荐】用 - 要点自然带出搭配（洁面+精华+防晒 / 上衣+下装），"
        "每个搭配一句理由，优先引用搭配方案资料，不硬推、不写废话。"
    ),
}


def build_stage_guide(decision: SalesDecision) -> str:
    """按动作返回话术指导块（注入系统提示词）；无对应动作返回空串。"""
    return STAGE_GUIDES.get(decision.action, "")


# ── 阶段持久化（Redis 优先，内存兜底） ──


def _get_redis() -> Any | None:
    """懒加载 Redis 客户端；连接失败后静默降级内存（只探测一次）。"""
    global _redis_client, _redis_checked
    if not _redis_checked:
        _redis_checked = True
        try:
            import redis as _redis_mod

            client = _redis_mod.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "0")),
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            client.ping()
            _redis_client = client
        except Exception:
            _redis_client = None
    return _redis_client


def load_stage(session_id: str | None) -> str:
    """读取会话当前购买阶段；无会话/失败返回 none。"""
    if not session_id:
        return STAGE_NONE
    client = _get_redis()
    if client is not None:
        try:
            value = client.get(_REDIS_KEY.format(session_id=session_id))
            if value:
                return value.decode("utf-8", errors="ignore")
            return STAGE_NONE
        except Exception:
            pass
    return _memory_store.get(session_id, STAGE_NONE)


def save_stage(session_id: str | None, stage: str) -> None:
    """保存会话购买阶段；无会话/失败静默跳过。"""
    if not session_id:
        return
    client = _get_redis()
    if client is not None:
        try:
            client.set(_REDIS_KEY.format(session_id=session_id), stage, ex=_STAGE_TTL)
            return
        except Exception:
            pass
    _memory_store[session_id] = stage


def reset_stage(session_id: str | None) -> None:
    """清空会话购买阶段。"""
    if not session_id:
        return
    client = _get_redis()
    if client is not None:
        try:
            client.delete(_REDIS_KEY.format(session_id=session_id))
            return
        except Exception:
            pass
    _memory_store.pop(session_id, None)


def load_media_offer(session_id: str | None) -> bool:
    """是否已主动提过“看实拍/视频”引导（每会话至多一次）。"""
    if not session_id:
        return False
    client = _get_redis()
    if client is not None:
        try:
            return bool(client.get(_OFFER_REDIS_KEY.format(session_id=session_id)))
        except Exception:
            pass
    return _offer_memory.get(session_id, False)


def save_media_offer(session_id: str | None) -> None:
    """标记本会话已提过媒体引导。"""
    if not session_id:
        return
    client = _get_redis()
    if client is not None:
        try:
            client.set(_OFFER_REDIS_KEY.format(session_id=session_id), "1", ex=_STAGE_TTL)
            return
        except Exception:
            pass
    _offer_memory[session_id] = True


def reset_media_offer(session_id: str | None) -> None:
    """清空媒体引导标记（测试/会话重置用）。"""
    if not session_id:
        return
    client = _get_redis()
    if client is not None:
        try:
            client.delete(_OFFER_REDIS_KEY.format(session_id=session_id))
            return
        except Exception:
            pass
    _offer_memory.pop(session_id, None)


def _count_clarify_questions(history: list[dict[str, Any]] | None) -> int:
    """统计历史中助手主动问过的需求确认轮次（防止反复挖需、堆问题）。"""
    if not history:
        return 0
    markers = ("肤质", "预算", "尺码", "场景", "平时")
    count = 0
    for msg in history:
        if msg.get("role") != "assistant":
            continue
        text = str(msg.get("content") or "")
        if any(marker in text for marker in markers):
            count += 1
    return count


def build_sales_context(
    route: dict[str, Any],
    session_id: str | None,
    question: str,
    *,
    history: list[dict[str, Any]] | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """接线层：读取会话阶段 → 决策 → 持久化 → 组装销售策略与话术指导。

    streaming / qa_chain 在检索拿到 route 后调用本函数，把结果注入
    受控生成层并写入 trace，实现"会话级销售"闭环。

    Returns:
        {
            intent, sub_intent, buying_signal, objection_type, missing_info,
            prev_stage, stage, action, reason, sales_strategy, guide,
        }
    """
    prev_stage = load_stage(session_id) if session_id else STAGE_NONE
    clarify = _count_clarify_questions(history)
    decision = decide_sales_step(
        route,
        prev_stage,
        question=question,
        clarify_count=clarify,
        profile=profile,
    )
    if session_id:
        save_stage(session_id, decision.stage)
    from agent_base.agents.sales import build_sales_strategy

    offer_media = False
    if (
        decision.action in {"recommend", "close_attempt"}
        and str(route.get("sub_intent") or "") != "media_request"
        and not load_media_offer(session_id)
    ):
        save_media_offer(session_id)
        offer_media = True
    sales_strategy = (
        build_sales_strategy(question, str(route.get("intent") or ""))
        if decision.action in SALES_ACTIONS
        else ""
    )
    return {
        "intent": route.get("intent", ""),
        "sub_intent": route.get("sub_intent", ""),
        "buying_signal": route.get("buying_signal", "normal"),
        "objection_type": route.get("objection_type", "none"),
        "missing_info": list(route.get("missing_info") or []),
        "prev_stage": prev_stage,
        "stage": decision.stage,
        "action": decision.action,
        "reason": decision.reason,
        "sales_strategy": sales_strategy,
        "guide": build_stage_guide(decision),
        "profile": dict(profile or {}),
        "offer_media": offer_media,
    }
