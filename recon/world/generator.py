"""可复现的交易流水生成器。

给定 seed，产出完全一致的一份业务世界：订单、支付、退款、分账、账务分录、结算单。
这一层生成的是「干净」的世界 —— 差错由 injector.py 之后注入，注入即标注。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal

from .. import db
from ..config import CHANNELS, MERCHANTS, USD_CNY_RATE, Channel, GenerateConfig, Merchant
from ..money import Cents, yuan

# 金额分布：按档抽样，保证阶梯费率的每一档都被走到
AMOUNT_BANDS: tuple[tuple[Cents, Cents, int], ...] = (
    (yuan("1.00"),    yuan("99.99"),    45),   # 小额，容差 1 分档
    (yuan("100.00"),  yuan("999.99"),   30),   # 银联阶梯第二档
    (yuan("1000.00"), yuan("9999.99"),  18),   # 银联阶梯第三档
    (yuan("10000.00"), yuan("50000.00"), 7),   # 大额，触发高审批阈值
)


def bill_date_for(occurred_at: datetime, cutoff_minutes: int) -> date:
    """按渠道日切规则，判断一笔交易属于哪一天的对账文件。

    日切在 23:30 意味着：T 日 23:30 之后的交易，落入 T+1 的账单。
    """
    minute_of_day = occurred_at.hour * 60 + occurred_at.minute
    if minute_of_day >= cutoff_minutes > 0:
        return occurred_at.date() + timedelta(days=1)
    return occurred_at.date()


@dataclass
class GenStats:
    orders: int = 0
    payments_success: int = 0
    payments_failed: int = 0
    refunds: int = 0
    partial_refunds: int = 0
    splits: int = 0
    ledger_entries: int = 0
    settlements: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


class WorldGenerator:
    def __init__(self, conn, cfg: GenerateConfig):
        self.conn = conn
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.stats = GenStats()
        self._seq = 0

    # ---------------------------------------------------------------- ids
    def _next(self) -> int:
        self._seq += 1
        return self._seq

    def _oid(self, d: date) -> str:
        return f"O{d:%Y%m%d}{self._next():06d}"

    # ------------------------------------------------------------ master
    def seed_master(self) -> None:
        for m in MERCHANTS.values():
            db.insert(self.conn, "merchants", {
                "id": m.id,
                "name": m.name,
                "settle_cycle": m.settle_cycle_override or "T+1",
                "allow_advance": int(m.allow_advance),
                "channels": db.jdump(list(m.channels)),
            })
        for c in CHANNELS.values():
            db.insert(self.conn, "channels", {
                "id": c.id,
                "name": c.name,
                "fee_desc": c.fee_rule.describe(),
                "cutoff_minutes": c.cutoff_minutes,
                "bill_basis": c.bill_basis,
                "settle_cycle": c.settle_cycle,
                "refund_mode": c.refund_mode,
                "currency": c.currency,
                "rounding": c.rounding,
            })
        self.conn.commit()

    # ------------------------------------------------------------ amount
    def _amount(self, channel: Channel) -> Cents:
        weights = [b[2] for b in AMOUNT_BANDS]
        low, high, _ = self.rng.choices(AMOUNT_BANDS, weights=weights, k=1)[0]
        cents = self.rng.randint(low, high)
        if channel.currency == "USD":
            # USD 单笔小一些，直接按汇率折回去再取整，制造非整数分的原始值
            cents = max(100, int(Decimal(cents) / USD_CNY_RATE))
        return cents

    def _fee_rate_for(self, merchant: Merchant, channel: Channel) -> Decimal | None:
        return merchant.fee_override.get(channel.id)

    def _our_fee(self, merchant: Merchant, channel: Channel, amount: Cents) -> Cents:
        """我方记账用的手续费 —— 干净世界里按渠道标准费率记，和渠道一致。

        商户协议费率（fee_override）是我方和商户之间的合约，不改变渠道实际扣费。
        「我方误按协议费率记账」是 D05 差错，由 injector 注入，不在这里系统性发生。
        """
        return channel.fee_rule.compute(amount)

    # -------------------------------------------------------------- time
    def _order_time(self, d: date) -> datetime:
        """一天内的下单时间。刻意在 23:00-23:59 和 01:00-02:59 加权，
        让「跨日切」成为自然高频区，而不是靠注入硬造。"""
        bucket = self.rng.choices(
            ["day", "late_night", "early_morning"], weights=[70, 20, 10], k=1
        )[0]
        if bucket == "late_night":
            hour = self.rng.randint(23, 23)
            minute = self.rng.randint(0, 59)
        elif bucket == "early_morning":
            hour = self.rng.randint(0, 2)
            minute = self.rng.randint(0, 59)
        else:
            hour = self.rng.randint(6, 22)
            minute = self.rng.randint(0, 59)
        return datetime(d.year, d.month, d.day, hour, minute, self.rng.randint(0, 59))

    # --------------------------------------------------------------- run
    def run(self) -> GenStats:
        self.seed_master()
        start = datetime.strptime(self.cfg.start_date, "%Y-%m-%d").date()
        merchants = list(MERCHANTS.values())

        for day_i in range(self.cfg.days):
            d = start + timedelta(days=day_i)
            for _ in range(self.cfg.orders_per_day):
                merchant = self.rng.choice(merchants)
                channel = CHANNELS[self.rng.choice(merchant.channels)]
                self._one_order(d, merchant, channel)
            self.conn.commit()

        self._make_settlements(start)
        self.conn.commit()
        return self.stats

    # ---------------------------------------------------------- one flow
    def _one_order(self, d: date, merchant: Merchant, channel: Channel) -> None:
        created = self._order_time(d)
        amount = self._amount(channel)
        oid = self._oid(d)
        failed = self.rng.random() < self.cfg.fail_ratio

        db.insert(self.conn, "orders", {
            "id": oid,
            "merchant_id": merchant.id,
            "channel_id": channel.id,
            "amount_cents": amount,
            "currency": channel.currency,
            "status": "failed" if failed else "paid",
            "created_at": created.isoformat(timespec="seconds"),
        })
        self.stats.orders += 1

        pid = "P" + oid[1:]
        if failed:
            db.insert(self.conn, "payments", {
                "id": pid, "order_id": oid, "channel_id": channel.id,
                "channel_txn_no": None, "amount_cents": amount, "fee_cents": 0,
                "status": "failed", "paid_at": None, "callback_at": None,
                "idempotency_key": f"pay:{oid}",
            })
            self.stats.payments_failed += 1
            return

        paid_at = created + timedelta(seconds=self.rng.randint(3, 180))
        txn_no = f"{channel.id.upper()}{paid_at:%Y%m%d%H%M%S}{self.rng.randint(1000, 9999)}"
        fee = self._our_fee(merchant, channel, amount)

        db.insert(self.conn, "payments", {
            "id": pid, "order_id": oid, "channel_id": channel.id,
            "channel_txn_no": txn_no, "amount_cents": amount, "fee_cents": fee,
            "status": "success",
            "paid_at": paid_at.isoformat(timespec="seconds"),
            "callback_at": (paid_at + timedelta(seconds=self.rng.randint(1, 30))).isoformat(timespec="seconds"),
            "idempotency_key": f"pay:{oid}",
        })
        self.stats.payments_success += 1
        self._ledger_payment(pid, amount, fee, paid_at)

        # 分账
        if merchant.split_receivers:
            for receiver, ratio in merchant.split_receivers:
                db.insert(self.conn, "splits", {
                    "id": f"SP{self._next():07d}", "order_id": oid,
                    "receiver_id": receiver, "ratio": str(ratio),
                    "amount_cents": int((Decimal(amount) * ratio).to_integral_value()),
                })
                self.stats.splits += 1

        # 退款
        if self.rng.random() < self.cfg.refund_ratio:
            self._one_refund(oid, pid, channel, amount, paid_at, txn_no)

    def _one_refund(self, oid: str, pid: str, channel: Channel,
                    amount: Cents, paid_at: datetime, txn_no: str) -> None:
        partial = self.rng.random() < self.cfg.partial_refund_ratio
        if partial:
            ratio = Decimal(self.rng.choice(["0.2", "0.3", "0.5", "0.7"]))
            r_amount = int((Decimal(amount) * ratio).to_integral_value())
            kind = "partial"
        else:
            r_amount = amount
            kind = "full"
        if r_amount <= 0:
            return

        requested = paid_at + timedelta(minutes=self.rng.randint(30, 48 * 60))
        refunded = requested + timedelta(minutes=self.rng.randint(1, 120))
        rid = "R" + oid[1:] + f"{self.rng.randint(1, 9)}"

        db.insert(self.conn, "refunds", {
            "id": rid, "order_id": oid, "payment_id": pid,
            "channel_txn_no": txn_no + "R", "amount_cents": r_amount,
            "kind": kind, "status": "success", "mode": channel.refund_mode,
            "requested_at": requested.isoformat(timespec="seconds"),
            "refunded_at": refunded.isoformat(timespec="seconds"),
            "idempotency_key": f"refund:{rid}",
        })
        self.conn.execute(
            "UPDATE orders SET status=? WHERE id=?",
            ("refunded" if kind == "full" else "partially_refunded", oid),
        )
        self.stats.refunds += 1
        if partial:
            self.stats.partial_refunds += 1
        self._ledger_refund(rid, r_amount, refunded)

    # ------------------------------------------------------------ ledger
    def _ledger_payment(self, pid: str, amount: Cents, fee: Cents, at: datetime) -> None:
        """借贷平衡：D(净额) + D(手续费) == C(交易收入)"""
        ts = at.isoformat(timespec="seconds")
        rows = [
            {"id": f"L{self._next():08d}", "ref_type": "payment", "ref_id": pid,
             "account": "channel_receivable", "direction": "D",
             "amount_cents": amount - fee, "occurred_at": ts},
            {"id": f"L{self._next():08d}", "ref_type": "payment", "ref_id": pid,
             "account": "fee_expense", "direction": "D",
             "amount_cents": fee, "occurred_at": ts},
            {"id": f"L{self._next():08d}", "ref_type": "payment", "ref_id": pid,
             "account": "revenue", "direction": "C",
             "amount_cents": amount, "occurred_at": ts},
        ]
        db.insert_many(self.conn, "ledger_entries", rows)
        self.stats.ledger_entries += len(rows)

    def _ledger_refund(self, rid: str, amount: Cents, at: datetime) -> None:
        ts = at.isoformat(timespec="seconds")
        rows = [
            {"id": f"L{self._next():08d}", "ref_type": "refund", "ref_id": rid,
             "account": "revenue", "direction": "D",
             "amount_cents": amount, "occurred_at": ts},
            {"id": f"L{self._next():08d}", "ref_type": "refund", "ref_id": rid,
             "account": "channel_receivable", "direction": "C",
             "amount_cents": amount, "occurred_at": ts},
        ]
        db.insert_many(self.conn, "ledger_entries", rows)
        self.stats.ledger_entries += len(rows)

    # ------------------------------------------------------- settlements
    def _make_settlements(self, start: date) -> None:
        """按商户结算周期造结算单（默认 pending）。D16 会把其中一张改成 paid。"""
        for m in MERCHANTS.values():
            for day_i in range(self.cfg.days):
                d = start + timedelta(days=day_i)
                net = db.scalar(self.conn, """
                    SELECT COALESCE(SUM(p.amount_cents - p.fee_cents), 0)
                    FROM payments p JOIN orders o ON o.id = p.order_id
                    WHERE o.merchant_id = ? AND p.status='success' AND date(p.paid_at) = ?
                """, (m.id, d.isoformat()))
                refunded = db.scalar(self.conn, """
                    SELECT COALESCE(SUM(r.amount_cents), 0)
                    FROM refunds r JOIN orders o ON o.id = r.order_id
                    WHERE o.merchant_id = ? AND r.status='success' AND date(r.refunded_at) = ?
                """, (m.id, d.isoformat()))
                amount = int(net) - int(refunded)
                if amount == 0:
                    continue
                db.insert(self.conn, "settlements", {
                    "id": f"ST{m.id}{d:%Y%m%d}",
                    "merchant_id": m.id,
                    "period_start": d.isoformat(),
                    "period_end": d.isoformat(),
                    "amount_cents": amount,
                    "status": "pending",
                    "frozen_reason": None,
                    "created_at": (datetime.combine(d, datetime.min.time())
                                   + timedelta(days=1, hours=10)).isoformat(timespec="seconds"),
                })
                self.stats.settlements += 1


def generate(conn, cfg: GenerateConfig) -> GenStats:
    return WorldGenerator(conn, cfg).run()
