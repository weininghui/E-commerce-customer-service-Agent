"""情绪规则路由（P6-04，规则通道不调 LLM）。

情绪是路由信号不是标签——只影响话术和转人工决策。
"""

from __future__ import annotations

from typing import Any

EMOTION_RULES: dict[str, dict[str, Any]] = {
    "anger": {
        "keywords": [
            "投诉", "太差", "垃圾", "气死", "骗子", "差评", "坑人",
            "退货", "退款", "无语", "服了", "气人", "推卸", "敷衍",
            "踢皮球", "心累", "拉黑", "再也不买", "这态度", "什么态度",
            "态度差", "怎么办事的", "是不是故意", "骗人",
            "居然说是我弄的", "说是人为的", "霸王条款", "过分", "失望",
            "答非所问", "没下文", "没天理", "我真的会谢", "让我自己承担",
            "骚扰", "投诉无门", "315", "一次比一次失望", "到底想不想做生意",
            "让我自己找", "让我自己查", "你们还怪", "怪我们", "怪我",
        ],
        "action": "安抚优先 + 转人工高优先级",
        "tone": "先致歉再解决问题，语气诚恳不急躁",
    },
    "anxiety": {
        "keywords": [
            "担心", "害怕", "会不会", "安全吗", "怎么办", "靠谱吗",
            "怕踩雷", "心里没底", "没底", "有点慌", "忐忑", "万一",
            "有保障吗", "保障", "不安", "咋办", "纠结", "犹豫",
            "怕买到假货", "怕过敏", "会过敏", "怕效果不好", "踩雷",
            "怕烂脸", "怕买错", "怕被税", "怕混卖", "怕不值", "怕收到",
            "怕临期", "怕跑路", "怕买贵", "担心买错", "担心正品",
            "能保证正品吗", "保证正品", "退换麻烦吗", "退款会不会", "用得完吗",
            "会不会很慢", "是不是临期", "是不是假的", "是不是买到假的",
            "心里特别没底", "特别犹豫", "纠结要不要", "复不复杂", "流程复不复杂",
            "怕这单出问题", "怕混卖", "怕不合", "怕不合适", "怕有问题",
            "说明书没说清楚", "说明没写", "没说清楚", "不太懂", "流程我不太懂",
            "怕买到混卖", "特别怕", "有点不安", "不安", "骗人的吧", "心里没底",
            "退换麻烦", "退换方不方便", "有点担心", "担心适不适合",
        ],
        "action": "确定性信息优先，少用'可能''应该'",
        "tone": "给明确、具体的信息，减少不确定性用语",
    },
    "positive": {
        "keywords": [
            "喜欢", "满意", "太好了", "不错", "好评", "推荐",
            "太棒", "超出预期", "回购", "满分",
            "相见恨晚", "宝藏店铺", "收藏了", "国货之光", "点赞", "安利",
            "太划算", "没话说", "三好", "不后悔", "强烈推荐", "物超所值",
            "效果很赞", "肉眼可见", "体验很好", "体验好", "支持一下",
            "性价比超高", "最好用", "很好用", "很赞", "很舒服", "很专业",
            "态度超棒", "人很好", "态度很好", "秒退款", "体验好",
        ],
        "action": "促单 + 好评引导",
        "tone": "亲切热情，可主动推荐搭配产品或引导好评",
    },
}

_HANDOFF_KEYWORDS = ["转人工", "人工客服", "找人工", "客服电话", "投诉"]

# 强情绪词在"咨询/疑问"语境下不视为情绪触发（避免"请问退货地址"误判愤怒）
_QUESTION_PREFIXES = ["请问", "想问", "咨询", "了解", "问下", "问一下", "想问问"]
_SOFT_STRONG_WORDS = {"退货", "退款", "投诉", "骗人"}
_CONSULT_MARKERS = ["不太懂", "流程", "麻烦不", "会不会", "能不能", "是不是", "吗", "怎么", "哪"]


def detect_emotion(text: str) -> dict[str, Any]:
    """规则通道情绪检测。

    Args:
        text: 用户输入文本。

    Returns:
        {"label": str, "intensity": float, "matched": list[str], "should_handoff": bool}
    """
    text_lower = text.lower()
    best_label = "neutral"
    best_intensity = 0.0
    best_matched: list[str] = []

    is_question = any(text_lower.startswith(p) or f"{p}" in text_lower[:8] for p in _QUESTION_PREFIXES) or any(
        m in text_lower for m in _CONSULT_MARKERS
    )
    for label, rule in EMOTION_RULES.items():
        matched = [kw for kw in rule["keywords"] if kw in text_lower]
        # 疑问语境下的"退货/退款/投诉"是正常咨询，不是情绪触发
        if is_question and label in ("anger",):
            matched = [kw for kw in matched if kw not in _SOFT_STRONG_WORDS]
        if matched:
            intensity = min(1.0, len(matched) * 0.3)
            if intensity > best_intensity:
                best_label = label
                best_intensity = intensity
                best_matched = matched

    # 转人工检测
    should_handoff = any(kw in text_lower for kw in _HANDOFF_KEYWORDS) or (
        best_label == "anger" and best_intensity >= 0.6
    )

    return {
        "label": best_label,
        "intensity": round(best_intensity, 2),
        "matched": best_matched,
        "should_handoff": should_handoff,
    }


def emotion_tone_guide(label: str) -> str:
    """根据情绪标签返回话术指导。

    Args:
        label: 情绪标签。

    Returns:
        可注入 agent prompt 的话术指导文本。
    """
    if label == "neutral" or label not in EMOTION_RULES:
        return ""
    rule = EMOTION_RULES[label]
    return f"[用户情绪: {label}] {rule['action']}。话术风格: {rule['tone']}。"


_CHAT_PATTERNS = (
    "你好", "您好", "hello", "hi", "嗨", "谢谢", "感谢", "再见", "拜拜",
    "辛苦了", "在吗", "好的", "嗯", "哈哈", "加油", "晚安", "早安", "没事",
)


def looks_like_chat(question: str) -> bool:
    """闲聊/寒暄检测（确定性规则，纯对话直答用）。

    极短问题且命中寒暄词表 → 跳过检索直接生成，零成本。
    """
    q = question.strip().lower()
    return len(q) <= 12 and any(p in q for p in _CHAT_PATTERNS)
