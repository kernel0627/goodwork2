"""金额与手续费。

两条铁律：
1. 金额一律用「整数分」表示，绝不出现 float。
2. 舍入模式必须显式声明。渠道之间舍入模式不同，是 D03 类差错的真实来源。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from typing import Literal

Cents = int

RoundingMode = Literal["half_up", "half_even"]


def yuan(amount: str | int | float) -> Cents:
    """把「元」转成「分」。字符串优先，避免 float 误差。"""
    return int((Decimal(str(amount)) * 100).to_integral_value(rounding=ROUND_HALF_UP))


def fmt(cents: Cents) -> str:
    """分 -> 便于人读的元字符串。"""
    sign = "-" if cents < 0 else ""
    c = abs(int(cents))
    return f"{sign}{c // 100}.{c % 100:02d}"


def _round(value: Decimal, mode: RoundingMode) -> Cents:
    rounding = ROUND_HALF_UP if mode == "half_up" else ROUND_HALF_EVEN
    return int(value.to_integral_value(rounding=rounding))


# --------------------------------------------------------------------------
# 手续费规则
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FeeRule:
    """一条渠道手续费规则。

    kind:
      rate   —— 按费率
      fixed  —— 每笔固定
      tiered —— 阶梯费率，tiers 为 (金额上限含, 费率) 升序，最后一档上限用 None 表示无穷
      mixed  —— 费率 + 固定（PayPal 那种）
    """

    kind: Literal["rate", "fixed", "tiered", "mixed"]
    rate: Decimal = Decimal("0")
    fixed_cents: Cents = 0
    tiers: tuple[tuple[Cents | None, Decimal], ...] = field(default_factory=tuple)
    min_cents: Cents = 0
    max_cents: Cents | None = None
    rounding: RoundingMode = "half_up"

    def compute(self, amount_cents: Cents, *, rounding: RoundingMode | None = None) -> Cents:
        mode = rounding or self.rounding
        amt = Decimal(amount_cents)

        if self.kind == "rate":
            raw = amt * self.rate
        elif self.kind == "fixed":
            raw = Decimal(self.fixed_cents)
        elif self.kind == "mixed":
            raw = amt * self.rate + Decimal(self.fixed_cents)
        elif self.kind == "tiered":
            rate = self.tiers[-1][1]
            for upper, tier_rate in self.tiers:
                if upper is None or amount_cents <= upper:
                    rate = tier_rate
                    break
            raw = amt * rate
        else:  # pragma: no cover - dataclass 已限定字面量
            raise ValueError(f"unknown fee kind: {self.kind}")

        fee = _round(raw, mode)
        fee = max(fee, self.min_cents)
        if self.max_cents is not None:
            fee = min(fee, self.max_cents)
        return fee

    def describe(self) -> str:
        if self.kind == "rate":
            body = f"费率 {self.rate * 100:.2f}%"
        elif self.kind == "fixed":
            body = f"每笔固定 {fmt(self.fixed_cents)} 元"
        elif self.kind == "mixed":
            body = f"费率 {self.rate * 100:.2f}% + 每笔 {fmt(self.fixed_cents)} 元"
        else:
            parts = []
            for upper, rate in self.tiers:
                bound = "以上" if upper is None else f"≤{fmt(upper)}元"
                parts.append(f"{bound} {rate * 100:.2f}%")
            body = "阶梯（" + " / ".join(parts) + "）"
        extra = []
        if self.min_cents:
            extra.append(f"最低 {fmt(self.min_cents)} 元")
        if self.max_cents is not None:
            extra.append(f"最高 {fmt(self.max_cents)} 元")
        extra.append("四舍五入" if self.rounding == "half_up" else "银行家舍入")
        return f"{body}（{'，'.join(extra)}）"


# --------------------------------------------------------------------------
# 口径归一：把渠道账单金额换算回「交易额（gross）」口径
# --------------------------------------------------------------------------

def to_gross(record_amount_cents: Cents, fee_cents: Cents, basis: Literal["gross", "net"]) -> Cents:
    """渠道账单金额 -> 我方交易额口径。

    basis="gross" 渠道报交易额，手续费单列
    basis="net"   渠道报扣费后净额 —— 直接和我方比会差出一个手续费，这就是 D04
    """
    return record_amount_cents if basis == "gross" else record_amount_cents + fee_cents
