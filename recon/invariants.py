"""金额不变量校验器 —— 这个项目的技术脊梁。

金额系统的正确性只能靠不变量证明，不能靠「看起来对」。
每条不变量都要能定位到具体单据，不能只报一个 bool。

INV1 单笔守恒   订单与其支付/退款金额自洽，累计退款不得超过原额
INV2 全量守恒   我方合计 - 渠道合计（归一到 gross）== 差错池残留金额
INV3 幂等       同一 idempotency_key 的资金动作执行次数不超过 1
INV4 借贷平衡   每张凭证 sum(借) == sum(贷)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import db
from .config import CHANNELS
from .money import fmt, to_gross


@dataclass
class CheckResult:
    name: str
    code: str
    passed: bool
    checked: int = 0
    violations: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def n_violations(self) -> int:
        return len(self.violations)


MAX_SAMPLES = 12


# --------------------------------------------------------------------------

def inv1_order_integrity(conn) -> CheckResult:
    rows = db.q(conn, """
        SELECT o.id, o.amount_cents, o.status,
               COALESCE(p.paid_total, 0)  AS paid_total,
               COALESCE(p.paid_count, 0)  AS paid_count,
               COALESCE(f.refund_total, 0) AS refund_total
        FROM orders o
        LEFT JOIN (SELECT order_id, SUM(amount_cents) AS paid_total, COUNT(*) AS paid_count
                   FROM payments WHERE status='success' GROUP BY order_id) p
               ON p.order_id = o.id
        LEFT JOIN (SELECT order_id, SUM(amount_cents) AS refund_total
                   FROM refunds WHERE status='success' GROUP BY order_id) f
               ON f.order_id = o.id
    """)
    bad: list[str] = []
    for r in rows:
        if r["paid_count"] > 1:
            bad.append(f"订单 {r['id']} 有 {r['paid_count']} 笔成功支付（应为 1）")
        elif r["paid_count"] == 1 and r["paid_total"] != r["amount_cents"]:
            bad.append(f"订单 {r['id']} 订单额 {fmt(r['amount_cents'])} != "
                       f"支付额 {fmt(r['paid_total'])}")
        if r["refund_total"] > r["amount_cents"]:
            bad.append(f"订单 {r['id']} 累计退款 {fmt(r['refund_total'])} > "
                       f"原额 {fmt(r['amount_cents'])}")
    return CheckResult("单笔守恒", "INV1", not bad, len(rows), bad[:MAX_SAMPLES],
                       note="累计退款超原额即 D10；违反是预期的，说明注入生效")


def inv2_total_conservation(conn) -> CheckResult:
    """我方合计 - 渠道合计（归一 gross）== 该对账任务差错池残留。

    两侧都用带符号口径（退款为负）。这条必须精确成立到分，
    不成立就是匹配逻辑或 diff_cents 约定有 bug，不是注入效果。
    """
    bad: list[str] = []
    tasks = db.q(conn, "SELECT * FROM recon_tasks")
    for t in tasks:
        channel = CHANNELS[t["channel_id"]]
        ch_gross = 0
        for r in db.q(conn, """
            SELECT r.rec_type, r.amount_cents, r.fee_cents
            FROM channel_bill_records r JOIN channel_bills b ON b.id = r.bill_id
            WHERE r.channel_id=? AND b.bill_date=?
        """, (t["channel_id"], t["bill_date"])):
            if r["rec_type"] == "refund":
                ch_gross -= r["amount_cents"]
            else:
                ch_gross += to_gross(r["amount_cents"], r["fee_cents"], channel.bill_basis)

        # diff_cents 恒等于「我方带符号贡献 - 渠道带符号贡献」，直接求和即可，
        # 不做任何形态判断。只排除 post_match 造的结算合规差错（不来自流水匹配）。
        residual = int(db.scalar(conn, """
            SELECT COALESCE(SUM(diff_cents), 0) FROM recon_diffs
            WHERE recon_task_id = ? AND source = 'match'
        """, (t["id"],)))

        lhs = t["our_total_cents"] - ch_gross
        if lhs != residual:
            bad.append(f"任务 {t['id']}：我方 {fmt(t['our_total_cents'])} - 渠道(gross) "
                       f"{fmt(ch_gross)} = {fmt(lhs)}，差错池残留 {fmt(residual)}，"
                       f"缺口 {fmt(lhs - residual)}")
    return CheckResult("全量守恒", "INV2", not bad, len(tasks), bad[:MAX_SAMPLES],
                       note="必须恒成立。不成立说明匹配逻辑或 diff_cents 约定有 bug")


def inv3_idempotency(conn) -> CheckResult:
    bad: list[str] = []
    for r in db.q(conn, """
        SELECT id, idempotency_key, executed_count FROM adjustments WHERE executed_count > 1
    """):
        bad.append(f"调账 {r['id']} (key={r['idempotency_key']}) 执行了 "
                   f"{r['executed_count']} 次")
    dup = db.q(conn, """
        SELECT idempotency_key, COUNT(*) AS n FROM payments
        WHERE idempotency_key IS NOT NULL GROUP BY idempotency_key HAVING n > 1
    """)
    for r in dup:
        bad.append(f"支付幂等键重复：{r['idempotency_key']} 出现 {r['n']} 次")
    dup_r = db.q(conn, """
        SELECT idempotency_key, COUNT(*) AS n FROM refunds
        WHERE idempotency_key IS NOT NULL GROUP BY idempotency_key HAVING n > 1
    """)
    for r in dup_r:
        bad.append(f"退款幂等键重复：{r['idempotency_key']} 出现 {r['n']} 次")
    total = int(db.scalar(conn, "SELECT COUNT(*) FROM adjustments")) + \
        int(db.scalar(conn, "SELECT COUNT(*) FROM payments")) + \
        int(db.scalar(conn, "SELECT COUNT(*) FROM refunds"))
    return CheckResult("幂等", "INV3", not bad, total, bad[:MAX_SAMPLES],
                       note="阶段 5 执行冲正后是主战场，阶段 0 应恒成立")


def inv4_ledger_balance(conn) -> CheckResult:
    rows = db.q(conn, """
        SELECT ref_type, ref_id,
               SUM(CASE WHEN direction='D' THEN amount_cents ELSE 0 END) AS d,
               SUM(CASE WHEN direction='C' THEN amount_cents ELSE 0 END) AS c
        FROM ledger_entries GROUP BY ref_type, ref_id
    """)
    bad = [f"凭证 {r['ref_type']}:{r['ref_id']} 借 {fmt(r['d'])} != 贷 {fmt(r['c'])}"
           for r in rows if r["d"] != r["c"]]
    return CheckResult("借贷平衡", "INV4", not bad, len(rows), bad[:MAX_SAMPLES],
                       note="必须恒成立")


ALL_CHECKS = (inv1_order_integrity, inv2_total_conservation,
              inv3_idempotency, inv4_ledger_balance)

# 阶段 0 里允许违反的不变量（注入本身就是为了制造这些违反）
EXPECTED_VIOLATIONS = {"INV1"}


def run_all(conn) -> list[CheckResult]:
    return [check(conn) for check in ALL_CHECKS]


def hard_failures(results: list[CheckResult]) -> list[CheckResult]:
    """必须恒成立却没成立的 —— 这些是真 bug，不是注入效果。"""
    return [r for r in results if not r.passed and r.code not in EXPECTED_VIOLATIONS]
