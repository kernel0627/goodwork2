"""纯规则基线 —— 完全不调模型，把 policies/diff_sop.md 的识别依据直接写成代码。

这个基线必须是**善意的强基线**，不是稻草人：
- 所有检测器**并行全跑**，不是首个命中就返回 —— 所以它能识别复合差错
- 金额维度的检测器作用在「按流水号跨账单日找到的记录」上，
  不局限于同一账单日的那条 —— 所以「跨日归属 + 口径差异」这种复合它也能抓
- 用的是和 agent 完全相同的受控证据接口，取证次数可比

不这么做的话，「agent 比规则强」就是自己和自己比，一文不值。

⭐ 一个必须写进结论的固有优势：**规则基线天然免疫提示注入。**
   它压根不读 memo 字段。这是 agent 相对规则的固有劣势，不是可以粉饰的东西。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..config import tolerance_for
from ..eval.evidence import EvidenceView, hours_between
from ..eval.solution import UNKNOWN, Solution
from ..eval.tasks import Task
from ..money import to_gross
from ..world.generator import bill_date_for
from ..world.injector import (AUTO_WRITEOFF, CHANNEL_INQUIRY, CODES,
                              DISCARD_DUPLICATE, ESCALATE, HOLD_NEXT_BILL,
                              REVERSAL, SUPPLEMENT)

# 我方系统统一用四舍五入（见 policies/channel_fees.md §1）
OUR_ROUNDING = "half_up"

# D09 与 D14 的分界：两侧时间戳相差多少小时算「我方回调延迟」
LATE_CALLBACK_HOURS = 20.0

_SEVERITY = {"closed": 0, "held": 1, "escalated": 2}


@dataclass
class Fire:
    code: str
    why: str


class RuleBaseline:
    """确定性规则求解器。零 token、零延迟意义上的成本基线。"""

    name = "rule_baseline"

    # ------------------------------------------------------------------
    def solve(self, task: Task, ev: EvidenceView) -> Solution:
        ev.reset_trace()
        fires: list[Fire] = []
        refs: list[str] = []
        facts: dict = {}

        diff = ev.diff(task.diff_id)
        if diff is None:
            return self._compose(task, [], ev, refs, "差错不存在")

        if diff["source"] == "settlement_scan":
            fires += self._settlement(ev, diff, refs)
        elif diff["source"] == "rule_scan":
            fires += self._order_rule(ev, diff, refs)
        else:
            fires += self._match(ev, diff, refs, facts)

        return self._compose(task, fires, ev, refs, facts=facts)

    # ------------------------------------------------------- 结算合规扫描
    def _settlement(self, ev, diff, refs) -> list[Fire]:
        s = ev.settlement(diff["our_ref_id"])
        if s is None:
            return []
        refs.append(f"settlements:{s['id']}")
        m = ev.merchant(s["merchant_id"])
        if m is None:
            return []
        refs.append(f"merchants:{m['id']}")
        others = ev.open_diffs(s["period_start"], exclude=diff["id"])
        if s["status"] == "paid" and m["allow_advance"] == 0 and others:
            return [Fire("D16", f"商户 {m['id']} allow_advance=0，{s['period_start']} "
                                f"存在 {len(others)} 条未平差错，结算单已 paid")]
        return []

    # ------------------------------------------------------- 业务规则扫描
    def _order_rule(self, ev, diff, refs) -> list[Fire]:
        o = ev.order(diff["our_ref_id"])
        if o is None:
            return []
        refs.append(f"orders:{o['id']}")
        rs = ev.refunds_by_order(o["id"])
        total = sum(r["amount_cents"] for r in rs if r["status"] == "success")
        if total > o["amount_cents"]:
            return [Fire("D10", f"订单原额 {o['amount_cents']} 分，累计成功退款 "
                                f"{total} 分，超额 {total - o['amount_cents']} 分")]
        return []

    # ----------------------------------------------------------- 流水匹配
    def _match(self, ev, diff, refs, facts: dict | None = None) -> list[Fire]:
        facts = facts if facts is not None else {}
        txn = diff["channel_txn_no"]
        ch = ev.channel_cfg(diff["channel_id"])
        recs = ev.channel_records_by_txn(txn)
        for r in recs:
            refs.append(f"channel_bill_records:{r['id']}")

        pay = ev.payment_by_txn(txn)
        ref = ev.refund_by_txn(txn)
        if pay:
            refs.append(f"payments:{pay['id']}")
        if ref:
            refs.append(f"refunds:{ref['id']}")

        our_type = "payment" if pay else ("refund" if ref else None)
        our_gross = pay["amount_cents"] if pay else (ref["amount_cents"] if ref else None)
        our_fee = pay["fee_cents"] if pay else 0
        order_id = (pay or ref)["order_id"] if (pay or ref) else None
        order = ev.order(order_id) if order_id else None
        if order:
            refs.append(f"orders:{order['id']}")

        fires: list[Fire] = []

        # ---- 币种（优先级最高，一旦不一致金额就没法比） ----
        if order and recs and any(r["currency"] != order["currency"] for r in recs):
            got = {r["currency"] for r in recs}
            return [Fire("D12", f"我方 {order['currency']}，渠道 {got}，币种不一致")]

        # ---- 一号多条：重复下发 vs 串号 ----
        if len(recs) > 1:
            amounts = {r["amount_cents"] for r in recs}
            if len(amounts) == 1:
                fires.append(Fire("D11", f"流水号 {txn} 有 {len(recs)} 条渠道明细、金额相同"))
            else:
                return [Fire("D17", f"流水号 {txn} 对应 {len(amounts)} 种不同金额 {amounts}")]

        # ---- 方向/符号错误 ----
        if our_type == "refund" and recs and all(r["rec_type"] == "payment" for r in recs):
            fires.append(Fire("D18", "我方为退款单，渠道明细却记成 payment"))

        # ---- 我方单边 ----
        if diff["channel_record_id"] is None:
            fires += self._our_only(ev, diff, ch, recs, pay, ref, our_type, refs)

        # ---- 渠道单边 ----
        elif diff["our_ref_id"] is None:
            fires += self._channel_only(ev, diff, ch, recs, pay, ref, refs)

        # ---- 金额维度（对「跨账单日找到的记录」也生效，才抓得到复合差错）----
        if our_gross is not None and recs:
            fires += self._amount_dimension(ev, ch, our_gross, our_fee, our_type,
                                            recs, order, refs)

        # ---- 手续费维度 ----
        # 只对成功支付有意义：失败单我方本就不记手续费，拿它和标准费率比会
        # 系统性误报 D05（旧版就是这样把 D06 全部判成 D05,D06）。
        if pay and pay["status"] == "success" and our_gross is not None:
            std = ch.fee_rule.compute(our_gross)
            # 结构化记下手续费三方对照。D05 与 D22 的分辨点就是「偏离的是哪一侧」，
            # 而费率误用公告声明的正是「商户侧记账正确、错在渠道」。
            # 让下游从这里读，比从 notes 的中文里读可靠。
            _prs = [r for r in recs if r["rec_type"] == "payment"]
            facts["fee"] = {
                "our_fee_cents": our_fee,
                "standard_fee_cents": std,
                "channel_fee_cents": _prs[0]["fee_cents"] if _prs else None,
                "our_matches_standard": abs(our_fee - std) < 2,
                "channel_matches_standard": (abs(_prs[0]["fee_cents"] - std) < 2
                                             if _prs else None),
            }
            if abs(our_fee - std) >= 2:
                facts["fee"]["deviating_side"] = "ours"
                fires.append(Fire("D05", f"我方手续费 {our_fee} 分，按标准费率复算应为 "
                                         f"{std} 分，差 {our_fee - std} 分"))
            else:
                # 渠道侧手续费与合同费率不符。一个正常的规则引擎会在这里冲正差额 ——
                # 而这正是 D22 的陷阱：公告已说明是渠道误用费率、将自行更正、
                # 明确要求商户不要调整账务。规则读不到公告，就会去动一笔不该动的账。
                # D04 已经用「手续费未单列」解释了这笔差额，不能再拿同一份证据
                # 指控渠道费率偏差 —— 会系统性把 D04 误报成 D04,D05。
                already = any(f.code == "D04" for f in fires)
                prs = _prs
                if not already and prs and abs(prs[0]["fee_cents"] - std) >= 2:
                    facts["fee"]["deviating_side"] = "channel"
                    fires.append(Fire("D05", f"渠道手续费 {prs[0]['fee_cents']} 分与合同费率"
                                             f"复算值 {std} 分不符，差 "
                                             f"{prs[0]['fee_cents'] - std} 分，冲正差额"))

        return fires

    # ------------------------------------------------------------------
    def _our_only(self, ev, diff, ch, recs, pay, ref, our_type, refs) -> list[Fire]:
        if recs:
            # 渠道侧记录还在，只是跑到了别的账单日 —— 这是 D09/D14/D08，不是 D01
            rec = recs[0]
            if rec["rec_type"] == "refund" or our_type == "refund":
                if pay is None and ref is not None:
                    p = ev.payment(ref["payment_id"])
                    if p:
                        refs.append(f"payments:{p['id']}")
                    if p and p["paid_at"]:
                        pay_bill = bill_date_for(datetime.fromisoformat(p["paid_at"]),
                                                 ch.cutoff_minutes).isoformat()
                        if rec["bill_date"] < pay_bill:
                            return [Fire("D08", f"退款明细落在 {rec['bill_date']}，"
                                                f"早于其支付所属账单日 {pay_bill}")]
                return [Fire("D09", f"渠道退款明细在 {rec['bill_date']}，"
                                    f"我方归属 {diff['bill_date']}")]
            return [self._shift(ch, pay, rec, diff)]

        # 渠道侧彻底没有这个流水号
        if our_type == "refund" and ch.refund_mode == "balance":
            return [Fire("D15", f"{ch.name} 退款方式为 balance，按政策不进渠道流水")]
        return [Fire("D01", "渠道账单无此流水号，且该渠道退款走原路，属真实单边账")]

    def _channel_only(self, ev, diff, ch, recs, pay, ref, refs) -> list[Fire]:
        if pay is not None:
            if pay["status"] == "pending" and pay["callback_at"] is None:
                return [Fire("D02", "我方支付为 pending 且 callback_at 为空，回调丢失")]
            if pay["status"] == "failed":
                return [Fire("D06", "我方支付状态为 failed，渠道却有成功明细，真伪未定")]
            rec = recs[0] if recs else None
            if rec:
                return [self._shift(ch, pay, rec, diff)]
        if ref is not None and recs:
            return [Fire("D08", f"渠道退款明细在 {recs[0]['bill_date']}，我方归属不同账单日")]

        # 我方完全没有这个流水号：靠金额 + 时间窗认重复支付。
        # ⚠️ 必须先把渠道金额归一到 gross 再比 —— net 口径渠道直接拿账单金额
        #    去匹配我方交易额永远匹配不上（旧版就是这样把 D07 全判成 UNKNOWN）。
        rec = recs[0] if recs else None
        if rec:
            probe = (to_gross(rec["amount_cents"], rec["fee_cents"], ch.bill_basis)
                     if rec["rec_type"] == "payment" else rec["amount_cents"])
            cands = ev.payments_by_amount_time(diff["channel_id"], probe,
                                               rec["occurred_at"], window_minutes=30)
            hit = [c for c in cands if c["status"] == "success"
                   and c["channel_txn_no"] != rec["channel_txn_no"]]
            if hit:
                refs.append(f"payments:{hit[0]['id']}")
                return [Fire("D07", f"同金额 {probe} 分（已归一 gross）、时间接近的成功支付 "
                                    f"{hit[0]['id']} 已存在，判为用户重复支付")]
        return []

    def _shift(self, ch, pay, rec, diff) -> Fire:
        """归属错位：靠两侧时间戳区分是渠道侧移位（D09）还是我方回调延迟（D14）。"""
        gap = hours_between(pay["paid_at"] if pay else None, rec["occurred_at"])
        if gap is not None and gap > LATE_CALLBACK_HOURS:
            return Fire("D14", f"我方 paid_at 与渠道 occurred_at 相差 {gap:.1f}h，"
                               f"属我方回调延迟导致的归属错位")
        return Fire("D09", f"渠道明细在 {rec['bill_date']}、我方归属 {diff['bill_date']}，"
                           f"两侧时间戳一致（差 {gap if gap is None else round(gap, 2)}h），"
                           f"属渠道侧移位")

    def _amount_dimension(self, ev, ch, our_gross, our_fee, our_type,
                          recs, order, refs) -> list[Fire]:
        same = [r for r in recs if r["rec_type"] == (our_type or r["rec_type"])]
        rec = (same or recs)[0]
        rg = (to_gross(rec["amount_cents"], rec["fee_cents"], ch.bill_basis)
              if rec["rec_type"] == "payment" else rec["amount_cents"])
        delta = our_gross - rg
        if delta == 0:
            return []

        # 分账比例错误：证据在 splits 表。
        # ⚠️ SOP 要求「渠道金额与错误分账额一致」，所以必须验证金额差**恰好等于**
        #    分账缺口。只看「分账合计 != 订单额」会把同一订单上的 D04/D20 也吞掉。
        if order:
            sp = ev.splits(order["id"])
            if sp:
                for s in sp:
                    refs.append(f"splits:{s['id']}")
                total = sum(s["amount_cents"] for s in sp)
                gap = order["amount_cents"] - total
                if gap != 0 and abs(delta) == abs(gap):
                    return [Fire("D13", f"分账明细合计 {total} 分、订单金额 "
                                        f"{order['amount_cents']} 分，缺口 {gap} 分"
                                        f"恰等于金额差 {delta} 分")]

        std_fee = ch.fee_rule.compute(our_gross)
        if ch.bill_basis == "net" and abs(delta) == std_fee:
            return [Fire("D04", f"差额 {delta} 分恰等于标准手续费 {std_fee} 分，"
                                f"且该渠道为 net 口径，属记账口径差异")]

        if abs(delta) == 1 and ch.rounding != OUR_ROUNDING:
            return [Fire("D03", f"差额恰为 1 分，{ch.name} 用 {ch.rounding}、"
                                f"我方用 {OUR_ROUNDING}，可归因于舍入模式")]

        tol = tolerance_for(our_gross)
        if abs(delta) <= tol:
            return [Fire("D20", f"差额 {delta} 分在容差 {tol} 分内，"
                                f"且不满足 D03/D04 识别依据，如实记为无解释差异")]

        return [Fire(UNKNOWN, f"金额差 {delta} 分超出容差 {tol} 分，"
                              f"且不匹配任何已知规则")]

    # ------------------------------------------------------------------
    def _compose(self, task: Task, fires: list[Fire], ev: EvidenceView,
                 refs: list[str], note: str = "", *,
                 facts: dict | None = None) -> Solution:
        codes: list[str] = []
        for f in fires:
            if f.code not in codes:
                codes.append(f.code)

        if not codes or codes == [UNKNOWN]:
            # 认不出来就转人工 —— 这是诚实的兜底，不是失败
            return Solution(
                task_id=task.task_id, root_causes=[UNKNOWN], actions=[ESCALATE],
                expected_status="escalated", confidence=0.2,
                notes=note or ("；".join(f.why for f in fires) or "无规则命中，转人工"),
                evidence_refs=refs, reads=ev.reads, rows_read=ev.rows_read,
                chars_read=ev.chars_read, steps=1, facts=dict(facts or {}))

        known = [c for c in codes if c != UNKNOWN]
        actions: list[str] = []
        for c in known:
            a = CODES[c].action
            if a and a not in actions:
                actions.append(a)
        if UNKNOWN in codes and ESCALATE not in actions:
            actions.append(ESCALATE)

        status = max((CODES[c].expected_status for c in known),
                     key=lambda s: _SEVERITY.get(s, 0))
        if UNKNOWN in codes:
            status = "escalated"

        return Solution(
            task_id=task.task_id, root_causes=known + ([UNKNOWN] if UNKNOWN in codes else []),
            actions=actions, expected_status=status,
            confidence=0.9 if UNKNOWN not in codes else 0.4,
            notes="；".join(f.why for f in fires),
            evidence_refs=refs, reads=ev.reads, rows_read=ev.rows_read,
            chars_read=ev.chars_read, steps=1, facts=dict(facts or {}))


def run_baseline(conn, tasks: list[Task]) -> dict[str, Solution]:
    solver = RuleBaseline()
    ev = EvidenceView(conn)
    return {t.task_id: solver.solve(t, ev) for t in tasks}


__all__ = ["RuleBaseline", "run_baseline", "Fire",
           "AUTO_WRITEOFF", "SUPPLEMENT", "REVERSAL", "CHANNEL_INQUIRY",
           "HOLD_NEXT_BILL", "ESCALATE", "DISCARD_DUPLICATE"]
