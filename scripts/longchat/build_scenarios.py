"""生成程序化长对话场景（S1 超长混合 / S5 上下文压缩）。

运行一次把生成的 YAML 落到 scenarios/ 目录，便于直接跑模拟器。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"


def _turn(q: str, **asserts: Any) -> dict[str, Any]:
    return {"q": q, "assert": asserts or {"answer_nonempty": True, "event_complete": True}}


def build_s1() -> dict[str, Any]:
    """S1：100 轮超长混合（闲聊/咨询/价格/售后/购买信号/推荐/媒体穿插）。"""
    pool = [
        "你好",
        "在吗",
        "谢谢",
        "好的",
        "玻尿酸精华适合油皮吗",
        "这款面霜的成分是什么",
        "保质期多久",
        "含酒精吗",
        "白色纯棉T恤怎么样",
        "它会不会透",
        "高腰阔腿裤显矮吗",
        "多少钱",
        "有优惠吗",
        "能便宜点吗",
        "什么时候发货",
        "能退货吗",
        "这个精华适合我吗，有点想买",
        "有点纠结",
        "好吧，就它了",
        "帮我推荐一款面霜",
        "敏感肌用什么",
        "有图片吗",
        "看看这件T恤",
        "再见",
    ]
    turns = [_turn(q) for i in range(100) for q in [pool[i % len(pool)]]]
    return {"id": "s1", "name": "超长混合100轮（稳定性基线）", "turns": turns}


def build_s5() -> dict[str, Any]:
    """S5：60 轮长消息（≤950 字/轮，API 限制 1000）触发四级上下文压缩。"""
    bases = [
        "请详细介绍玻尿酸保湿精华液的成分、功效、质地和使用方法",
        "白色纯棉T恤的面料克重、版型、尺码和洗涤注意事项有哪些",
        "神经酰胺面霜适合什么肤质，换季泛红怎么用",
    ]
    turns: list[dict[str, Any]] = []
    for i in range(80):
        base = bases[i % len(bases)]
        filler = "（补充：希望了解成分表、使用步骤、注意事项、适用人群、搭配建议、售后政策等细节信息，尽量完整）" * 16
        q = f"{base}，{filler}"[:950]
        turns.append(_turn(q))
    return {"id": "s5", "name": "上下文压缩80轮（长消息触发四级降级）", "turns": turns}


def main() -> None:
    """生成 S1/S5 程序化场景 YAML。"""
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    for data in (build_s1(), build_s5()):
        # 清理同 id 的旧版本文件，避免 --scenario 把多版本都跑一遍
        for old in SCENARIOS_DIR.glob(f"{data['id']}_*.yaml"):
            old.unlink()
        path = SCENARIOS_DIR / f"{data['id']}_{data['name'].split('（')[0]}.yaml"
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
        print(f"{path}: {len(data['turns'])} 轮")


if __name__ == "__main__":
    main()
