"""业务世界的规则配置。

这个文件是「业务复杂度」的真正来源：四个渠道的手续费规则、日切时间、账单口径、
结算周期、退款方式、舍入模式全都不一样。差错不是随机噪声，是这些规则差异的必然产物。

⚠️ 这里的每一条规则都必须和 recon/policies/ 下的政策文档严格一致。
   如果两边对不上，agent 就算检索到了政策也算不出正确答案 —— 任务集就成了废的。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from .money import Cents, FeeRule, RoundingMode, yuan

# --------------------------------------------------------------------------
# 渠道
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Channel:
    id: str
    name: str
    fee_rule: FeeRule
    cutoff_minutes: int          # 日切：当日 0 点起算的分钟数；>= 该值的交易落入次日账单
    bill_basis: Literal["gross", "net"]
    settle_cycle: Literal["T+1", "T+3", "weekly"]
    refund_mode: Literal["original", "balance"]   # 原路退回 / 退到余额
    currency: str = "CNY"

    @property
    def rounding(self) -> RoundingMode:
        """舍入模式只在 FeeRule 上声明一处。

        ⚠️ 这里曾经是一个独立字段，结果银联只在 FeeRule 里设了 half_even、
           Channel 上还留着默认的 half_up，导致「银行家舍入渠道」集合为空，
           D03 一条都注入不出来。同一事实声明两遍必然会不一致 —— 改成派生属性。
        """
        return self.fee_rule.rounding


CHANNELS: dict[str, Channel] = {
    "alipay": Channel(
        id="alipay",
        name="支付宝",
        fee_rule=FeeRule(kind="rate", rate=Decimal("0.006"), min_cents=1, rounding="half_up"),
        cutoff_minutes=0,                 # 0 点日切
        bill_basis="gross",               # 报交易额
        settle_cycle="T+1",
        refund_mode="original",
    ),
    "wxpay": Channel(
        id="wxpay",
        name="微信支付",
        fee_rule=FeeRule(kind="rate", rate=Decimal("0.006"), min_cents=1, rounding="half_up"),
        cutoff_minutes=23 * 60 + 30,      # 23:30 日切 —— 跨日归属差错的来源
        bill_basis="net",                 # ⭐ 报扣费后净额 —— D04 口径差异的来源
        settle_cycle="T+1",
        refund_mode="original",
    ),
    "unionpay": Channel(
        id="unionpay",
        name="银联",
        fee_rule=FeeRule(
            kind="tiered",
            tiers=((yuan(100), Decimal("0.005")), (yuan(1000), Decimal("0.0045")), (None, Decimal("0.0038"))),
            min_cents=5,
            rounding="half_even",         # ⭐ 银行家舍入 —— D03 精度差错的来源
        ),
        cutoff_minutes=2 * 60,            # 凌晨 2 点日切
        bill_basis="gross",
        settle_cycle="T+3",
        refund_mode="original",
    ),
    "paypal": Channel(
        id="paypal",
        name="PayPal",
        fee_rule=FeeRule(
            kind="mixed", rate=Decimal("0.034"), fixed_cents=yuan(2), min_cents=0, rounding="half_up"
        ),
        cutoff_minutes=8 * 60,            # UTC 0 点 = 本地 8 点
        bill_basis="net",
        settle_cycle="weekly",
        refund_mode="balance",            # ⭐ 退到余额 —— D15 退款口径差错的来源
        currency="USD",
    ),
}

# 汇率（USD -> CNY），固定值，用于 D12 币种错配
USD_CNY_RATE = Decimal("7.20")


# --------------------------------------------------------------------------
# 商户合约
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Merchant:
    id: str
    name: str
    channels: tuple[str, ...]
    fee_override: dict[str, Decimal] = field(default_factory=dict)  # channel_id -> 协议费率
    settle_cycle_override: str | None = None
    allow_advance: bool = False          # 是否允许有未平差错时垫资结算
    split_receivers: tuple[tuple[str, Decimal], ...] = field(default_factory=tuple)  # 分账


MERCHANTS: dict[str, Merchant] = {
    "M001": Merchant("M001", "云图科技", ("alipay", "wxpay"), allow_advance=True),
    "M002": Merchant("M002", "海角零售", ("alipay", "wxpay", "unionpay"),
                     fee_override={"alipay": Decimal("0.0055")}),
    "M003": Merchant("M003", "长风教育", ("wxpay", "unionpay"), settle_cycle_override="T+3"),
    "M004": Merchant("M004", "北岸出海", ("paypal", "alipay")),
    "M005": Merchant("M005", "同舟平台", ("alipay", "wxpay"),
                     split_receivers=(("S01", Decimal("0.70")), ("S02", Decimal("0.30")))),
}


# --------------------------------------------------------------------------
# 容差分档（金额越大，允许的绝对容差越大）
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ToleranceBand:
    upper_cents: Cents | None      # 交易额上限（含），None 表示以上
    abs_cents: Cents               # 允许的绝对差额
    note: str


TOLERANCE_BANDS: tuple[ToleranceBand, ...] = (
    ToleranceBand(yuan(100), 1, "≤100 元，容差 1 分"),
    ToleranceBand(yuan(1000), 5, "≤1000 元，容差 5 分"),
    ToleranceBand(yuan(10000), 10, "≤10000 元，容差 1 角"),
    ToleranceBand(None, 100, "10000 元以上，容差 1 元"),
)


def tolerance_for(amount_cents: Cents) -> Cents:
    for band in TOLERANCE_BANDS:
        if band.upper_cents is None or amount_cents <= band.upper_cents:
            return band.abs_cents
    return TOLERANCE_BANDS[-1].abs_cents


# --------------------------------------------------------------------------
# 冲正/调账权限矩阵（金额阈值 -> 所需角色）
# --------------------------------------------------------------------------

ROLES = ("operator", "finance", "finance_manager", "risk")

AUTH_THRESHOLDS: tuple[tuple[Cents | None, str], ...] = (
    (yuan(100), "operator"),          # ≤100 元，运营可自行核销
    (yuan(5000), "finance"),          # ≤5000 元，财务审批
    (yuan(50000), "finance_manager"), # ≤50000 元，财务主管审批
    (None, "risk"),                   # 50000 元以上，风控审批
)


def required_role(amount_cents: Cents) -> str:
    amt = abs(amount_cents)
    for upper, role in AUTH_THRESHOLDS:
        if upper is None or amt <= upper:
            return role
    return "risk"


# --------------------------------------------------------------------------
# 生成参数默认值
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerateConfig:
    seed: int = 42
    start_date: str = "2026-07-01"
    days: int = 3
    orders_per_day: int = 200
    refund_ratio: float = 0.18          # 多少比例的成功订单会产生退款
    partial_refund_ratio: float = 0.45  # 退款里多少是部分退款
    fail_ratio: float = 0.06            # 支付失败比例
    inject_count_per_day: int = 120     # 每天注入多少条差错
    composite_ratio: float = 0.25       # 注入里多少是复合差错
