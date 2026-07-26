"""工具层 —— 包在受控证据层外面，是 agent 唯一的取数入口。

设计要点：

1. **不另写取数逻辑。** 全部复用 `recon/eval/evidence.py`，所以 agent 和规则基线
   取的是同一份证据、走的是同一套白名单断言，对比才公平。

2. **返回值预算。** 一天几千条流水不可能整个塞进 context。每个工具做字段投影 +
   行数上限，超了显式告知「还有 N 条，可用 X 下钻」，而不是静默截断。

3. **模型不做算术。** 手续费核算、口径归一、容差查档、时间差计算全部是确定性工具，
   模型只负责判断。这一刀划错，模型算错一次小数就会污染整条归因。

4. **错误信息就是给模型的 prompt。** 参数错 / 查无此物 / 权限不足要分开说清楚，
   模型才知道该改参数、该换方向、还是该转人工。

阶段 2 只开只读工具。动账工具（提案、审批、执行冲正）在阶段 5 接。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from ..config import required_role, tolerance_for
from ..eval.evidence import EvidenceAccessError, EvidenceView, hours_between
from ..money import to_gross

MAX_ROWS = 20


class ToolError(Exception):
    """参数或用法错误 —— 模型该改参数重试。"""


class NotFound(Exception):
    """查无此物 —— 模型该换个方向，而不是重复问。"""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    params: dict[str, str]           # 参数名 -> 说明
    handler: Callable[..., Any]
    risk: str = "readonly"

    def spec(self) -> str:
        args = ", ".join(f"{k}: {v}" for k, v in self.params.items()) or "无参数"
        return f"- {self.name}({args})\n    {self.description}"


# --------------------------------------------------------------------------

def _proj_channel_record(r, basis: str) -> dict:
    gross = (to_gross(r["amount_cents"], r["fee_cents"], basis)
             if r["rec_type"] == "payment" else r["amount_cents"])
    return {
        "record_id": r["id"], "bill_date": r["bill_date"], "rec_type": r["rec_type"],
        "amount_cents": r["amount_cents"], "fee_cents": r["fee_cents"],
        "gross_normalized_cents": gross,      # 口径归一由代码做，不让模型算
        "currency": r["currency"], "occurred_at": r["occurred_at"],
        "memo": r["memo"],
    }


def _proj_payment(p) -> dict:
    return {"payment_id": p["id"], "order_id": p["order_id"],
            "amount_cents": p["amount_cents"], "fee_cents": p["fee_cents"],
            "status": p["status"], "paid_at": p["paid_at"],
            "callback_at": p["callback_at"], "channel_txn_no": p["channel_txn_no"]}


def _proj_refund(f) -> dict:
    return {"refund_id": f["id"], "order_id": f["order_id"], "payment_id": f["payment_id"],
            "amount_cents": f["amount_cents"], "kind": f["kind"], "status": f["status"],
            "mode": f["mode"], "requested_at": f["requested_at"],
            "refunded_at": f["refunded_at"], "channel_txn_no": f["channel_txn_no"]}


def _budget(rows: list, n: int = MAX_ROWS, hint: str = "") -> dict:
    out = {"rows": rows[:n], "returned": min(len(rows), n), "total": len(rows)}
    if len(rows) > n:
        out["truncated"] = True
        out["hint"] = hint or f"还有 {len(rows) - n} 条未返回，请用更窄的条件再查"
    return out


# --------------------------------------------------------------------------

class ToolBox:
    def __init__(self, ev: EvidenceView, *, strip_injection_policy: bool = False,
                 as_of: str | None = None):
        self.ev = ev
        # 决策时刻。公告查询按它过滤 published_at，避免读到未来才发布的公告。
        self.as_of = as_of
        # 剥离对照组必须在**所有**读到 SOP 的路径上生效，否则模型一个
        # read_policy 就把剥掉的章节读回来了，对照组照旧不成立。
        self.strip_injection_policy = strip_injection_policy
        self._tools: dict[str, Tool] = {}
        self._register_all()

    # ------------------------------------------------------------- registry
    def _add(self, name: str, desc: str, params: dict[str, str], fn) -> None:
        self._tools[name] = Tool(name, desc, params, fn)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def catalog(self) -> str:
        return "\n".join(t.spec() for t in self._tools.values())

    def call(self, name: str, arguments: dict | None) -> dict:
        """统一入口。返回值一定是 dict，错误也是结构化的 —— 它就是给模型的 prompt。"""
        args = arguments or {}
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error_kind": "unknown_tool",
                    "error": f"没有名为 {name} 的工具。可用工具：{', '.join(self.names)}"}
        if not isinstance(args, dict):
            return {"ok": False, "error_kind": "bad_arguments",
                    "error": "arguments 必须是对象"}
        unknown = set(args) - set(tool.params)
        if unknown:
            return {"ok": False, "error_kind": "bad_arguments",
                    "error": f"{name} 不接受参数 {sorted(unknown)}；"
                             f"它接受 {sorted(tool.params)}"}
        try:
            return {"ok": True, "data": tool.handler(**args)}
        except NotFound as e:
            return {"ok": False, "error_kind": "not_found", "error": str(e)}
        except (ToolError, TypeError) as e:
            return {"ok": False, "error_kind": "bad_arguments", "error": str(e)}
        except EvidenceAccessError as e:
            return {"ok": False, "error_kind": "forbidden", "error": str(e)}

    # =====================================================================
    def _register_all(self) -> None:
        self._add("get_diff", "取当前差错的结构化信息（含金额差、手续费差、检测来源）。",
                  {"diff_id": "差错 ID"}, self.get_diff)
        self._add("get_channel_rules",
                  "取渠道规则：手续费规则、舍入模式、日切分钟、账单口径(gross/net)、"
                  "结算周期、退款方式、币种。",
                  {"channel_id": "渠道 ID"}, self.get_channel_rules)
        self._add("get_channel_records_by_txn",
                  "按渠道流水号取渠道账单明细，**跨所有账单日**。"
                  "区分「明细被删」与「明细只是落在别的账单日」必须用这个。",
                  {"channel_txn_no": "渠道流水号"}, self.get_channel_records_by_txn)
        self._add("get_our_records_by_txn",
                  "按渠道流水号取我方支付单与退款单。",
                  {"channel_txn_no": "渠道流水号"}, self.get_our_records_by_txn)
        self._add("get_order",
                  "取订单，以及该订单下全部支付、退款、分账的汇总（含累计退款是否超原额）。",
                  {"order_id": "订单 ID"}, self.get_order)
        self._add("get_settlement", "取结算单，以及其商户是否允许垫资。",
                  {"settlement_id": "结算单 ID"}, self.get_settlement)
        self._add("search_payments_by_amount_time",
                  "按金额 + 时间窗搜我方支付单。重复支付没有流水号线索，只能这样认。"
                  "金额请传已归一到交易额(gross)口径的值。",
                  {"channel_id": "渠道 ID", "amount_cents": "金额（分，gross 口径）",
                   "around": "时间中心 ISO8601", "window_minutes": "时间窗（分钟，默认 30）"},
                  self.search_payments_by_amount_time)
        self._add("read_channel_notices",
                  "读该渠道在该账单日生效的公告全文。"
                  "⚠️ 处置前必须查。同一天可能有多条公告，其中不少并不改变任何处置。",
                  {"channel_id": "渠道 ID", "bill_date": "账单日 YYYY-MM-DD"},
                  self.read_channel_notices)
        self._add("list_policies", "列出可读的政策文档名。", {}, self.list_policies)
        self._add("read_policy", "读一份政策文档全文。",
                  {"name": "文档名，如 diff_sop"}, self.read_policy)
        self._add("compute_standard_fee",
                  "按政策核算标准手续费（确定性计算，不要自己算）。",
                  {"channel_id": "渠道 ID", "gross_cents": "交易额（分）"},
                  self.compute_standard_fee)
        self._add("get_tolerance_and_authority",
                  "查该金额对应的对账容差档位与所需审批角色（确定性计算）。",
                  {"gross_cents": "交易额（分）", "action_amount_cents": "拟调账金额（分）"},
                  self.get_tolerance_and_authority)
        self._add("compute_bill_date",
                  "按渠道日切规则算某个时刻属于哪一天的账单（确定性计算）。",
                  {"channel_id": "渠道 ID", "at": "时刻 ISO8601"}, self.compute_bill_date)
        self._add("hours_between_timestamps",
                  "算两个时刻相差多少小时（确定性计算）。区分渠道侧移位与我方回调延迟要用。",
                  {"a": "时刻 A ISO8601", "b": "时刻 B ISO8601"}, self.hours_between_timestamps)

    # ----------------------------------------------------------- handlers
    def get_diff(self, diff_id: str) -> dict:
        d = self.ev.diff(diff_id)
        if d is None:
            raise NotFound(f"差错 {diff_id} 不存在")
        return {
            "diff_id": d["id"], "channel_id": d["channel_id"], "bill_date": d["bill_date"],
            "detection_source": d["source"],
            "our_ref_type": d["our_ref_type"], "our_ref_id": d["our_ref_id"],
            "channel_record_id": d["channel_record_id"],
            "channel_txn_no": d["channel_txn_no"],
            "our_gross_cents": d["our_gross_cents"],
            "channel_gross_cents": d["channel_gross_cents"],
            "diff_cents": d["diff_cents"],
            "fee_delta_cents": d["fee_delta_cents"],
            "status": d["status"],
            "note": ("diff_cents = 我方带符号贡献 - 渠道带符号贡献（退款计负）。"
                     "仅手续费维度有差异时 diff_cents 为 0，看 fee_delta_cents。"),
        }

    def get_channel_rules(self, channel_id: str) -> dict:
        try:
            ch = self.ev.channel_cfg(channel_id)
        except KeyError:
            raise NotFound(f"没有渠道 {channel_id}")
        self.ev.channel(channel_id)          # 留一条取证轨迹
        return {"channel_id": ch.id, "name": ch.name,
                "fee_rule": ch.fee_rule.describe(),
                "rounding": ch.rounding,
                "cutoff_minutes": ch.cutoff_minutes,
                "bill_basis": ch.bill_basis,
                "settle_cycle": ch.settle_cycle,
                "refund_mode": ch.refund_mode,
                "currency": ch.currency}

    def get_channel_records_by_txn(self, channel_txn_no: str) -> dict:
        recs = self.ev.channel_records_by_txn(channel_txn_no)
        if not recs:
            return {"rows": [], "returned": 0, "total": 0,
                    "note": "渠道侧任何账单日都没有该流水号的明细"}
        basis = self.ev.channel_cfg(recs[0]["channel_id"]).bill_basis
        out = _budget([_proj_channel_record(r, basis) for r in recs])
        out["bill_basis"] = basis
        return out

    def get_our_records_by_txn(self, channel_txn_no: str) -> dict:
        p = self.ev.payment_by_txn(channel_txn_no)
        f = self.ev.refund_by_txn(channel_txn_no)
        if p is None and f is None:
            return {"payment": None, "refund": None,
                    "note": "我方没有该流水号的支付或退款记录"}
        return {"payment": _proj_payment(p) if p else None,
                "refund": _proj_refund(f) if f else None}

    def get_order(self, order_id: str) -> dict:
        o = self.ev.order(order_id)
        if o is None:
            raise NotFound(f"订单 {order_id} 不存在")
        pays = self.ev.payments_by_order(order_id)
        refs = self.ev.refunds_by_order(order_id)
        sp = self.ev.splits(order_id)
        refunded = sum(r["amount_cents"] for r in refs if r["status"] == "success")
        split_total = sum(s["amount_cents"] for s in sp)
        return {
            "order_id": o["id"], "merchant_id": o["merchant_id"],
            "channel_id": o["channel_id"], "amount_cents": o["amount_cents"],
            "currency": o["currency"], "status": o["status"], "created_at": o["created_at"],
            "payments": [_proj_payment(p) for p in pays[:MAX_ROWS]],
            "refunds": [_proj_refund(r) for r in refs[:MAX_ROWS]],
            "refund_total_success_cents": refunded,
            "refund_exceeds_order": refunded > o["amount_cents"],
            "splits": [{"split_id": s["id"], "receiver_id": s["receiver_id"],
                        "ratio": s["ratio"], "amount_cents": s["amount_cents"]} for s in sp],
            "split_total_cents": split_total if sp else None,
            "split_gap_cents": (o["amount_cents"] - split_total) if sp else None,
        }

    def get_settlement(self, settlement_id: str) -> dict:
        s = self.ev.settlement(settlement_id)
        if s is None:
            raise NotFound(f"结算单 {settlement_id} 不存在")
        m = self.ev.merchant(s["merchant_id"])
        others = self.ev.open_diffs(s["period_start"], exclude=None,
                                    until=s["period_end"])
        return {"settlement_id": s["id"], "merchant_id": s["merchant_id"],
                "period_start": s["period_start"], "period_end": s["period_end"],
                "amount_cents": s["amount_cents"], "status": s["status"],
                "frozen_reason": s["frozen_reason"],
                "merchant_allow_advance": bool(m["allow_advance"]) if m else None,
                "open_diffs_in_period": len(others)}

    def search_payments_by_amount_time(self, channel_id: str, amount_cents: int,
                                       around: str, window_minutes: int = 30) -> dict:
        try:
            amount_cents = int(amount_cents)
            window_minutes = int(window_minutes)
        except (TypeError, ValueError):
            raise ToolError("amount_cents 与 window_minutes 必须是整数")
        rows = self.ev.payments_by_amount_time(channel_id, amount_cents, around,
                                               window_minutes=window_minutes)
        return _budget([_proj_payment(p) for p in rows],
                       hint="命中太多，请缩小时间窗")

    def read_channel_notices(self, channel_id: str, bill_date: str) -> dict:
        rows = self.ev.channel_notices(channel_id, bill_date, as_of=self.as_of)
        return {"rows": [{"notice_id": r["id"], "title": r["title"],
                          "published_at": r["published_at"],
                          "effective_from": r["effective_from"],
                          "effective_to": r["effective_to"],
                          "body": r["body"]} for r in rows],
                "total": len(rows),
                "note": ("公告是自由文本，没有类型标签。必须逐条读正文判断它是否真的覆盖"
                         "当前这条差错 —— 同一天常有维护通知、后台升级、风控调整这类"
                         "不改变任何处置的公告。")}

    def list_policies(self) -> dict:
        return {"policies": self.ev.policy_list()}

    def read_policy(self, name: str) -> dict:
        try:
            content = self.ev.policy(name)
        except EvidenceAccessError as e:
            raise NotFound(str(e))
        if self.strip_injection_policy and name == "diff_sop":
            from .prompts import _sop_text
            content = _sop_text(strip_injection=True)
        return {"name": name, "content": content}

    def compute_standard_fee(self, channel_id: str, gross_cents: int) -> dict:
        try:
            ch = self.ev.channel_cfg(channel_id)
        except KeyError:
            raise NotFound(f"没有渠道 {channel_id}")
        try:
            gross_cents = int(gross_cents)
        except (TypeError, ValueError):
            raise ToolError("gross_cents 必须是整数（单位：分）")
        return {"channel_id": channel_id, "gross_cents": gross_cents,
                "standard_fee_cents": ch.fee_rule.compute(gross_cents),
                "fee_rule": ch.fee_rule.describe(), "rounding": ch.rounding}

    def get_tolerance_and_authority(self, gross_cents: int,
                                    action_amount_cents: int = 0) -> dict:
        try:
            gross_cents = int(gross_cents)
            action_amount_cents = int(action_amount_cents)
        except (TypeError, ValueError):
            raise ToolError("金额参数必须是整数（单位：分）")
        return {"gross_cents": gross_cents,
                "tolerance_cents": tolerance_for(gross_cents),
                "action_amount_cents": action_amount_cents,
                "required_approval_role": required_role(action_amount_cents)}

    def compute_bill_date(self, channel_id: str, at: str) -> dict:
        from ..world.generator import bill_date_for
        try:
            ch = self.ev.channel_cfg(channel_id)
        except KeyError:
            raise NotFound(f"没有渠道 {channel_id}")
        try:
            ts = datetime.fromisoformat(at)
        except ValueError:
            raise ToolError(f"at 不是合法 ISO8601 时间：{at!r}")
        return {"at": at, "channel_id": channel_id,
                "cutoff_minutes": ch.cutoff_minutes,
                "bill_date": bill_date_for(ts, ch.cutoff_minutes).isoformat()}

    def hours_between_timestamps(self, a: str, b: str) -> dict:
        try:
            gap = hours_between(a, b)
        except ValueError as e:
            raise ToolError(f"时间格式错误：{e}")
        if gap is None:
            raise ToolError("两个时刻都不能为空")
        return {"a": a, "b": b, "hours": round(gap, 3)}


def digest(result: dict, limit: int = 900) -> str:
    """工具结果摘要 —— 存 trace 用，也用于「无进展」检测。"""
    s = json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
    return s if len(s) <= limit else s[:limit] + f"…(+{len(s) - limit})"
