"""按各渠道自己的口径与日切规则，生成 T+1 对账文件。

这一层生成的是「干净」账单 —— 和我方记录完全对得上。差错由 injector 注入。

两个系统性差异点在这里落地，它们是后面大量差错的真实来源：
  bill_basis  gross（报交易额）vs net（报扣费后净额）
  cutoff      不同渠道日切时间不同，同一笔交易可能落进不同日期的账单
"""
from __future__ import annotations

import random
from datetime import date, datetime, timedelta

from .. import db
from ..config import CHANNELS, Channel
from ..money import Cents
from .generator import bill_date_for

BENIGN_MEMOS = (
    None, None, None, None,
    "normal",
    "SETTLE_OK",
    "trade_finished",
    "merchant_confirmed",
)


def _channel_amount(channel: Channel, gross: Cents, fee: Cents, rec_type: str) -> Cents:
    """把我方 gross 金额换算成渠道账单口径。退款一律按原额报，不含手续费。"""
    if rec_type == "refund":
        return gross
    return gross if channel.bill_basis == "gross" else gross - fee


def build_bills(conn, start_date: str, days: int, *, seed: int = 0) -> dict[str, int]:
    """为 [start, start+days] 区间内所有出现过流水的 (渠道, 账单日) 生成账单。

    多生成一天，接住日切溢出的尾巴，避免丢记录。
    """
    rng = random.Random(seed or 0)
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    dates = [start + timedelta(days=i) for i in range(days + 1)]
    stats = {"bills": 0, "records": 0}

    for channel in CHANNELS.values():
        # 一次把该渠道全部成功流水取出来，再按日切分桶，避免逐日重复扫表
        buckets: dict[date, list[dict]] = {d: [] for d in dates}

        pay_rows = db.q(conn, """
            SELECT p.id, p.channel_txn_no, p.amount_cents, p.fee_cents, p.paid_at, o.currency
            FROM payments p JOIN orders o ON o.id = p.order_id
            WHERE p.channel_id = ? AND p.status = 'success' AND p.channel_txn_no IS NOT NULL
        """, (channel.id,))
        for r in pay_rows:
            occurred = datetime.fromisoformat(r["paid_at"])
            bd = bill_date_for(occurred, channel.cutoff_minutes)
            if bd not in buckets:
                continue
            buckets[bd].append({
                "txn": r["channel_txn_no"], "rec_type": "payment",
                "gross": r["amount_cents"], "fee": r["fee_cents"],
                "occurred": occurred, "currency": r["currency"],
            })

        refund_rows = db.q(conn, """
            SELECT r.id, r.channel_txn_no, r.amount_cents, r.refunded_at, o.currency
            FROM refunds r JOIN orders o ON o.id = r.order_id
            WHERE o.channel_id = ? AND r.status = 'success' AND r.refunded_at IS NOT NULL
        """, (channel.id,))
        for r in refund_rows:
            occurred = datetime.fromisoformat(r["refunded_at"])
            bd = bill_date_for(occurred, channel.cutoff_minutes)
            if bd not in buckets:
                continue
            buckets[bd].append({
                "txn": r["channel_txn_no"], "rec_type": "refund",
                "gross": r["amount_cents"], "fee": 0,
                "occurred": occurred, "currency": r["currency"],
            })

        for bd, items in buckets.items():
            if not items:
                continue
            items.sort(key=lambda x: (x["occurred"], x["txn"]))
            bill_id = f"B{channel.id.upper()}{bd:%Y%m%d}S1"
            records = []
            total = 0
            for i, it in enumerate(items, start=1):
                amount = _channel_amount(channel, it["gross"], it["fee"], it["rec_type"])
                signed = -amount if it["rec_type"] == "refund" else amount
                total += signed
                records.append({
                    "id": f"{bill_id}R{i:05d}",
                    "bill_id": bill_id,
                    "channel_id": channel.id,
                    "channel_txn_no": it["txn"],
                    "rec_type": it["rec_type"],
                    "amount_cents": amount,
                    "fee_cents": it["fee"],
                    "currency": it["currency"],
                    "occurred_at": it["occurred"].isoformat(timespec="seconds"),
                    "memo": rng.choice(BENIGN_MEMOS),
                })

            db.insert(conn, "channel_bills", {
                "id": bill_id, "channel_id": channel.id,
                "bill_date": bd.isoformat(), "file_seq": 1,
                "record_count": len(records), "total_amount_cents": total,
                "received_at": (datetime.combine(bd, datetime.min.time())
                                + timedelta(days=1, hours=6)).isoformat(timespec="seconds"),
            })
            db.insert_many(conn, "channel_bill_records", records)
            stats["bills"] += 1
            stats["records"] += len(records)

    conn.commit()
    return stats


def refresh_bill_totals(conn) -> None:
    """注入之后重算每张账单的合计与条数（注入会增删改记录）。"""
    rows = db.q(conn, "SELECT id FROM channel_bills")
    for r in rows:
        agg = db.q1(conn, """
            SELECT COUNT(*) AS n,
                   COALESCE(SUM(CASE WHEN rec_type='refund' THEN -amount_cents
                                     ELSE amount_cents END), 0) AS total
            FROM channel_bill_records WHERE bill_id = ?
        """, (r["id"],))
        conn.execute(
            "UPDATE channel_bills SET record_count=?, total_amount_cents=? WHERE id=?",
            (agg["n"], agg["total"], r["id"]),
        )
    conn.commit()
