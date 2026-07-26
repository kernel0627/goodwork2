"""Agent 配置与消融阶梯。

每个靶子是一个开关，消融表每行一个配置、累积开启。这样「哪个改动值多少个点」
是可读出来的，而不是笼统地说「优化后提升了 N%」。

阶梯的对照组是 v1（全部关闭），也就是阶段 2 报告里那一列。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    name: str

    # 靶子 1：明确差错边界。治 scope creep ——
    # agent 会把同一订单上看到的所有问题都并进当前这一条差错。
    # 该缺陷同时造成复合漏报/乱报、过度转人工、越权。
    scope_boundary: bool = False

    # 靶子 2：把 diff_sop 内联进 system prompt。
    # v1 里它被 read_policy 调了 55/60 次、每次约 7KB，是输入 token 的主要来源。
    # 「会不会主动读政策」这个问题已经用 55/60 回答完了，继续每条付 7KB 是纯浪费，
    # 而且内联后能吃上 prompt 缓存。
    inline_sop: bool = False

    # 靶子 3：收结论前强制过一遍四个维度。
    # v1 的复合漏报形态非常一致（D01,D05→D01 / D03,D09→D03 / D04,D09→D04），
    # 都是「找到一个足够的解释就停」。
    dimension_checklist: bool = False

    max_steps: int = 14

    def label(self) -> str:
        on = [k for k in ("scope_boundary", "inline_sop", "dimension_checklist")
              if getattr(self, k)]
        return f"{self.name}" + (f" [{'+'.join(on)}]" if on else " [v1]")


# 累积阶梯：每一级只比上一级多开一个开关
def ablation_ladder(model: str) -> list[AgentConfig]:
    return [
        AgentConfig(name=f"{model}·v1"),
        AgentConfig(name=f"{model}·+边界", scope_boundary=True),
        AgentConfig(name=f"{model}·+内联SOP", scope_boundary=True, inline_sop=True),
        AgentConfig(name=f"{model}·+维度表", scope_boundary=True, inline_sop=True,
                    dimension_checklist=True),
    ]


V1 = AgentConfig(name="v1")
