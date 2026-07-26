"""求解方的输出格式 —— 规则基线和 agent 输出同一个结构，才能同表对比。"""
from __future__ import annotations

from dataclasses import dataclass, field

UNKNOWN = "UNKNOWN"


@dataclass
class Solution:
    task_id: str

    # ---- 结论 ----
    root_causes: list[str] = field(default_factory=list)   # 归因，可多个（复合差错）
    actions: list[str] = field(default_factory=list)       # 处置动作
    expected_status: str = "closed"                        # 处置完成后差错应处的终态
    confidence: float = 0.0                                # 0-1
    notes: str = ""

    # ---- 过程（用于成本-效果对比）----
    evidence_refs: list[str] = field(default_factory=list)  # "表:id"
    reads: int = 0            # 取证次数
    rows_read: int = 0
    chars_read: int = 0       # 取证读到的字符数 —— agent 的 context 压力代理指标
    steps: int = 0            # 决策轮数（基线恒为 1）
    tokens_in: int = 0
    tokens_out: int = 0
    cached_in: int = 0        # 输入里命中 prompt 缓存的部分
    cost_micro_cny: int = 0   # 微元（1e-6 元），避免浮点
    latency_ms: int = 0

    @property
    def is_unknown(self) -> bool:
        return not self.root_causes or self.root_causes == [UNKNOWN]

    def as_row(self) -> dict:
        return {
            "task_id": self.task_id,
            "root_causes": list(self.root_causes),
            "actions": list(self.actions),
            "expected_status": self.expected_status,
            "confidence": self.confidence,
            "reads": self.reads,
            "rows_read": self.rows_read,
            "chars_read": self.chars_read,
            "steps": self.steps,
            "tokens_in": self.tokens_in,
            "cached_in": self.cached_in,
            "tokens_out": self.tokens_out,
            "cost_micro_cny": self.cost_micro_cny,
            "latency_ms": self.latency_ms,
            "notes": self.notes,
        }
