"""金额与手续费 —— 单元测试。金额系统的每一条舍入规则都要钉住。"""
from __future__ import annotations

from decimal import Decimal

import pytest

from recon.config import CHANNELS, required_role, tolerance_for
from recon.money import FeeRule, fmt, to_gross, yuan


def test_yuan_avoids_float_error():
    assert yuan("0.07") == 7
    assert yuan("1234.56") == 123456
    assert yuan(0.07) == 7          # float 也要能正确进位


def test_fmt_roundtrip():
    for v in (0, 1, 7, 99, 100, 123456, -1, -123456):
        assert yuan(fmt(v)) == v


@pytest.mark.parametrize("amount,expected", [
    (yuan("100.00"), 60),           # 100 * 0.6% = 0.60
    (yuan("1.00"), 1),              # 0.006 元 -> 进位到最低收费 1 分
    (yuan("0.50"), 1),              # 最低收费兜底
])
def test_rate_fee(amount, expected):
    assert CHANNELS["alipay"].fee_rule.compute(amount) == expected


def test_rounding_modes_actually_differ():
    """银行家舍入必须和四舍五入在 .5 上给出不同结果 —— D03 的地基。"""
    rule_up = FeeRule(kind="rate", rate=Decimal("0.005"), rounding="half_up")
    rule_even = FeeRule(kind="rate", rate=Decimal("0.005"), rounding="half_even")
    amount = 100          # 100 * 0.005 = 0.5 分，正好在中点
    assert rule_up.compute(amount) == 1
    assert rule_even.compute(amount) == 0
    assert rule_up.compute(amount) != rule_even.compute(amount)


def test_channel_rounding_is_single_sourced():
    """回归测试：Channel.rounding 曾经是独立字段，和 FeeRule.rounding 对不上，
    导致「银行家舍入渠道」集合为空、D03 一条也注不出来。"""
    for c in CHANNELS.values():
        assert c.rounding == c.fee_rule.rounding, f"{c.id} 两处舍入模式声明不一致"


def test_at_least_one_banker_rounding_channel_exists():
    """D03（舍入模式差异）的存在前提。没有这种渠道，D03 就是不可解的标注。"""
    banker = [c.id for c in CHANNELS.values() if c.rounding == "half_even"]
    assert banker, "没有任何银行家舍入渠道，D03 无处可注入"


def test_tiered_fee_picks_one_bracket_not_cumulative():
    rule = CHANNELS["unionpay"].fee_rule
    assert rule.compute(yuan("100.00")) == 50        # 100 * 0.5%
    assert rule.compute(yuan("1000.00")) == 450      # 1000 * 0.45%
    assert rule.compute(yuan("2000.00")) == 760      # 2000 * 0.38%


def test_mixed_fee():
    rule = CHANNELS["paypal"].fee_rule
    # 100 * 3.4% + 2.00 = 3.40 + 2.00 = 5.40
    assert rule.compute(yuan("100.00")) == yuan("5.40")


def test_to_gross_normalisation():
    assert to_gross(1000, 60, "gross") == 1000
    assert to_gross(940, 60, "net") == 1000          # 净额 + 手续费 = 交易额


def test_tolerance_bands_monotonic():
    bands = [tolerance_for(yuan(x)) for x in ("50", "500", "5000", "50000")]
    assert bands == [1, 5, 10, 100]
    assert bands == sorted(bands)


def test_required_role_escalates_with_amount():
    assert required_role(yuan("50")) == "operator"
    assert required_role(yuan("500")) == "finance"
    assert required_role(yuan("20000")) == "finance_manager"
    assert required_role(yuan("100000")) == "risk"
    assert required_role(-yuan("100000")) == "risk"   # 取绝对值
