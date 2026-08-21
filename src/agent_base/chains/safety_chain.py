"""电商客服安全/合规评估（safety 门禁）。

规则化风险识别 + 门禁永远在图里（不交给模型自觉）。
电商场景覆盖：违禁功效承诺、敏感人群（孕妇/儿童）、过敏/皮肤刺激、
紧急健康场景（呼吸困难/休克/严重过敏）→ 提示就医并转人工。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class SafetyFinding:
    """一条安全发现：标签、风险等级、命中的关键词与提示文案。"""

    label: str
    level: str
    keywords: list[str]
    message: str

    def to_dict(self) -> dict[str, Any]:
        """转为 dict，便于 JSON 序列化。"""
        return asdict(self)


@dataclass(slots=True)
class SafetyAssessment:
    """整轮安全评估结果：风险等级 + 发现列表 + 就医/转人工标记。"""

    risk_level: str
    findings: list[SafetyFinding]
    must_consult: bool
    emergency: bool
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典，findings 逐条展开。"""
        return {
            "risk_level": self.risk_level,
            "must_consult": self.must_consult,
            "emergency": self.emergency,
            "warnings": self.warnings,
            "findings": [finding.to_dict() for finding in self.findings],
        }


RISK_RULES = [
    {
        "label": "pregnancy",
        "level": "medium",
        "keywords": ["孕妇", "妊娠", "怀孕", "备孕"],
        "message": "涉及孕妇/备孕人群使用护肤品，应优先选择成分温和的产品，建议先咨询医生或专业人士。",
    },
    {
        "label": "child",
        "level": "medium",
        "keywords": ["儿童", "小孩", "孩子", "婴儿", "宝宝", "未成年"],
        "message": "涉及儿童/婴幼儿护肤，需确认产品适用年龄段，建议咨询儿科或皮肤科专业人士。",
    },
    {
        "label": "allergy_skin",
        "level": "high",
        "keywords": ["过敏", "泛红", "刺痛", "发痒", "皮疹", "荨麻疹", "红肿"],
        "message": "涉及过敏或皮肤刺激风险，建议先停用并进行局部皮肤测试；症状明显时及时就医。",
    },
    {
        "label": "acute_allergy",
        "level": "high",
        "keywords": ["呼吸困难", "休克", "严重过敏", "昏迷"],
        "message": "涉及严重过敏或紧急健康场景，请立即停止使用并尽快就医，同时转接人工客服协助处理。",
    },
    {
        "label": "efficacy_claim",
        "level": "medium",
        "keywords": ["根治", "祛斑", "美白", "抗皱", "减肥", "瘦身", "生发", "永久"],
        "message": "涉及功效承诺类表述，化妆品/服饰客服不得承诺医疗级功效，应如实引用商品资料并提示个体差异。",
    },
]

CONTEXT_HIGH_RISK_KEYWORDS = ["禁止", "停用", "立即就医", "严重过敏", "慎用"]
EMERGENCY_KEYWORDS = ["呼吸困难", "休克", "昏迷", "严重过敏", "胸痛"]
LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}


def assess_safety(question: str, context: str = "") -> SafetyAssessment:
    """电商安全/合规评估。

    Args:
        question: 用户问题。
        context: 检索到的证据文本（用于识别来源中的高风险表述）。

    Returns:
        SafetyAssessment。
    """
    findings = [_finding_from_rule(rule, question) for rule in RISK_RULES if _matched_keywords(rule, question)]
    if any(keyword in context for keyword in CONTEXT_HIGH_RISK_KEYWORDS):
        findings.append(
            SafetyFinding(
                label="source_high_risk",
                level="high",
                keywords=[keyword for keyword in CONTEXT_HIGH_RISK_KEYWORDS if keyword in context],
                message="检索到的商品资料包含停用/就医等高风险表述，回答必须突出该风险。",
            )
        )

    emergency = any(keyword in question for keyword in EMERGENCY_KEYWORDS)
    risk_level = _max_level([finding.level for finding in findings] + (["high"] if emergency else []))
    must_consult = risk_level in {"medium", "high"} or emergency
    warnings = _build_warnings(risk_level, findings, emergency)
    return SafetyAssessment(
        risk_level=risk_level,
        findings=findings,
        must_consult=must_consult,
        emergency=emergency,
        warnings=warnings,
    )


def _finding_from_rule(rule: dict[str, Any], text: str) -> SafetyFinding:
    return SafetyFinding(
        label=rule["label"],
        level=rule["level"],
        keywords=_matched_keywords(rule, text),
        message=rule["message"],
    )


def _matched_keywords(rule: dict[str, Any], text: str) -> list[str]:
    return [keyword for keyword in rule["keywords"] if keyword in text]


def _max_level(levels: list[str]) -> str:
    if not levels:
        return "low"
    return max(levels, key=lambda level: LEVEL_ORDER[level])


def _build_warnings(risk_level: str, findings: list[SafetyFinding], emergency: bool) -> list[str]:
    warnings: list[str] = []
    if emergency:
        warnings.append("如出现呼吸困难、休克、昏迷或严重过敏等紧急情况，请立即停止使用并就医，同时联系人工客服。")
    if risk_level == "high":
        warnings.append("该问题涉及高风险健康/过敏场景，请不要仅凭自动回答使用产品，务必咨询专业医生或客服。")
    elif risk_level == "medium":
        warnings.append("该问题涉及需要谨慎判断的使用场景，建议咨询专业医生或在线客服后再购买/使用。")
    else:
        warnings.append("以上信息仅供参考，具体效果因人而异；如有皮肤不适请及时停止使用并咨询专业人士。")
    for finding in findings:
        if finding.message not in warnings:
            warnings.append(finding.message)
    return warnings
