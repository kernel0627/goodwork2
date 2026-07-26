"""对账匹配 —— 只匹配、只算差额，绝不做归因。

归因是规则基线和 agent 的活。这一层的产出是「原始差错池」：
结构上能机械看出来的东西（哪边缺、差多少、有没有重复），到此为止。

diff_cents 约定（唯一）：
    diff_cents = 我方带符号贡献 - 渠道带符号贡献       （退款计负）
恒等于该笔对全量守恒的贡献，所以 INV2 直接求和即可，不做任何形态判断。
仅手续费维度有差异时 diff_cents 自然为 0，改看 fee_delta_cents。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from . import db
from .config import CHANNELS
from .money import to_gross
from .world.generator import bill_date_for


def recon_ts(bill_date: str, hour: int = 7) -> str:
    """对账动作的时间戳 —— 从账单日派生，绝不用 datetime.now()。

    ⚠️ 这里曾经用 now()，导致同 seed 两次构建只要跨了秒边界，
       recon_diffs.created_at 就不同，整个任务集变成不可复现。
       可复现的数据集里不允许出现墙上时钟。
       语义上也更对：对账任务是 T+1 早上跑的，不是「现在」。
    """
    d = datetime.strptime(bill_date, "%Y-%m-%d").date() + timedelta(days=1)
    return datetime(d.year, d.month, d.day, hour, 0, 0).isoformat(timespec="seconds")


def diff_shape(row: Any) -> str:
    """差错的机械形态（从字段派生，不是归因）。"""
    if row["channel_record_id"] is None:
        return "OUR_ONLY"
    if row["our_ref_id"] is None:
        return "CHANNEL_ONLY"
    if row["our_gross_cents"] is not None and row["channel_gross_cents"] is not None:
        if row["our_gross_cents"] != row["channel_gross_cents"]:
            return "AMOUNT"
    return "OTHER"


# --------------------------------------------------------------------------

def _our_side(conn, channel_id: str, bill_date: str) -> dict[str, dict]:
    """我方在该账单日应有的流水，按渠道流水号索引。"""
    channel = CHANNELS[channel_id]
    out: dict[str, dict] = {}

    for r in db.q(conn, """
        SELECT p.id, p.channel_txn_no, p.amount_cents, p.fee_cents, p.paid_at,
               o.currency, o.merchant_id
        FROM payments p JOIN orders o ON o.id = p.order_id
        WHERE p.channel_id = ? AND p.status='success' AND p.channel_txn_no IS NOT NULL
    """, (channel_id,)):
        if bill_date_for(datetime.fromisoformat(r["paid_at"]), channel.cutoff_minutes).isoformat() != bill_date:
            continue
        out[r["channel_txn_no"]] = {
            "ref_type": "payment", "ref_id": r["id"], "rec_type": "payment",
            "gross": r["amount_cents"], "fee": r["fee_cents"],
            "currency": r["currency"], "merchant_id": r["merchant_id"],
        }

    for r in db.q(conn, """
        SELECT f.id, f.channel_txn_no, f.amount_cents, f.refunded_at,
               o.currency, o.merchant_id
        FROM refunds f JOIN orders o ON o.id = f.order_id
        WHERE o.channel_id = ? AND f.status='success' AND f.refunded_at IS NOT NULL
          AND f.channel_txn_no IS NOT NULL
    """, (channel_id,)):
        if bill_date_for(datetime.fromisoformat(r["refunded_at"]), channel.cutoff_minutes).isoformat() != bill_date:
            continue
        out[r["channel_txn_no"]] = {
            "ref_type": "refund", "ref_id": r["id"], "rec_type": "refund",
            "gross": r["amount_cents"], "fee": 0,
            "currency": r["currency"], "merchant_id": r["merchant_id"],
        }
    return out


def _channel_side(conn, channel_id: str, bill_date: str) -> dict[str, list]:
    rows = db.q(conn, """
        SELECT r.id, r.channel_txn_no, r.rec_type, r.amount_cents, r.fee_cents,
               r.currency, r.occurred_at
        FROM channel_bill_records r JOIN channel_bills b ON b.id = r.bill_id
        WHERE r.channel_id = ? AND b.bill_date = ?
    """, (channel_id, bill_date))
    grouped: dict[str, list] = {}
    for r in rows:
        grouped.setdefault(r["channel_txn_no"], []).append(r)
    return grouped


def reconcile(conn, dates: list[str]) -> dict[str, int]:
    stats = {"tasks": 0, "matched": 0, "diffs": 0}

    for channel_id, channel in CHANNELS.items():
        for bill_date in dates:
            now = recon_ts(bill_date)
            ours = _our_side(conn, channel_id, bill_date)
            theirs = _channel_side(conn, channel_id, bill_date)
            if not ours and not theirs:
                continue

            task_id = f"RT{channel_id.upper()}{bill_date.replace('-', '')}"
            diffs: list[dict] = []
            matched = 0
            seq = 0

            def new_diff(**kw) -> None:
                nonlocal seq
                seq += 1
                our_signed = kw.get("our_signed", 0)
                ch_signed = kw.get("channel_signed", 0)
                diffs.append({
                    "id": f"{task_id}D{seq:05d}",
                    "recon_task_id": task_id,
                    "channel_id": channel_id,
                    "bill_date": bill_date,
                    "our_ref_type": kw.get("our_ref_type"),
                    "our_ref_id": kw.get("our_ref_id"),
                    "channel_record_id": kw.get("channel_record_id"),
                    "channel_txn_no": kw.get("channel_txn_no"),
                    "our_ref_signed": our_signed,
                    "channel_signed": ch_signed,
                    "our_gross_cents": kw.get("our_gross"),
                    "channel_gross_cents": kw.get("channel_gross"),
                    "diff_cents": our_signed - ch_signed,
                    "fee_delta_cents": kw.get("fee_delta", 0),
                    "status": "new",
                    "created_at": now,
                })

            def signed(rec_type: str, gross: int) -> int:
                return -gross if rec_type == "refund" else gross

            def ch_gross_of(rec) -> int:
                if rec["rec_type"] == "payment":
                    return to_gross(rec["amount_cents"], rec["fee_cents"], channel.bill_basis)
                return rec["amount_cents"]

            for txn, mine in ours.items():
                recs = theirs.get(txn)
                my_signed = signed(mine["rec_type"], mine["gross"])
                if not recs:
                    new_diff(our_ref_type=mine["ref_type"], our_ref_id=mine["ref_id"],
                             channel_txn_no=txn, our_gross=mine["gross"],
                             our_signed=my_signed, fee_delta=mine["fee"])
                    continue

                primary = recs[0]
                ch_gross = ch_gross_of(primary)
                ch_signed = signed(primary["rec_type"], ch_gross)

                same_currency = primary["currency"] == mine["currency"]
                fee_delta = mine["fee"] - primary["fee_cents"]
                type_ok = primary["rec_type"] == mine["rec_type"]

                if not same_currency or my_signed != ch_signed or not type_ok:
                    new_diff(our_ref_type=mine["ref_type"], our_ref_id=mine["ref_id"],
                             channel_record_id=primary["id"], channel_txn_no=txn,
                             our_gross=mine["gross"], channel_gross=ch_gross,
                             our_signed=my_signed, channel_signed=ch_signed,
                             fee_delta=fee_delta)
                elif fee_delta != 0:
                    # gross 一致、仅手续费维度有差异：diff_cents 自然为 0，看 fee_delta_cents
                    new_diff(our_ref_type=mine["ref_type"], our_ref_id=mine["ref_id"],
                             channel_record_id=primary["id"], channel_txn_no=txn,
                             our_gross=mine["gross"], channel_gross=ch_gross,
                             our_signed=my_signed, channel_signed=ch_signed,
                             fee_delta=fee_delta)
                else:
                    matched += 1

                # 同一流水号多条渠道记录 = 重复下发 / 串号，每条多余的都算一条差错
                for extra in recs[1:]:
                    eg = ch_gross_of(extra)
                    new_diff(our_ref_type=mine["ref_type"], our_ref_id=mine["ref_id"],
                             channel_record_id=extra["id"], channel_txn_no=txn,
                             our_gross=mine["gross"], channel_gross=eg,
                             channel_signed=signed(extra["rec_type"], eg))

            for txn, recs in theirs.items():
                if txn in ours:
                    continue
                for rec in recs:
                    rg = ch_gross_of(rec)
                    new_diff(channel_record_id=rec["id"], channel_txn_no=txn,
                             channel_gross=rg,
                             channel_signed=signed(rec["rec_type"], rg),
                             fee_delta=-rec["fee_cents"])

            our_total = sum(signed(m["rec_type"], m["gross"]) for m in ours.values())
            ch_total = db.scalar(conn, """
                SELECT COALESCE(SUM(CASE WHEN r.rec_type='refund' THEN -r.amount_cents
                                         ELSE r.amount_cents END), 0)
                FROM channel_bill_records r JOIN channel_bills b ON b.id = r.bill_id
                WHERE r.channel_id=? AND b.bill_date=?
            """, (channel_id, bill_date))

            db.insert(conn, "recon_tasks", {
                "id": task_id, "channel_id": channel_id, "bill_date": bill_date,
                "status": "done", "started_at": now, "finished_at": now,
                "our_total_cents": our_total, "channel_total_cents": int(ch_total),
                "matched_count": matched, "diff_count": len(diffs),
            })
            db.insert_many(conn, "recon_diffs", diffs)
            stats["tasks"] += 1
            stats["matched"] += matched
            stats["diffs"] += len(diffs)

    conn.commit()
    return stats


# --------------------------------------------------------------------------
# 把注入日志里的答案贴到对账产出的差错上
# --------------------------------------------------------------------------

_SEVERITY = {"closed": 0, "held": 1, "escalated": 2}


def attach_ground_truth(conn) -> dict[str, int]:
    groups: dict[tuple[str, str], list] = {}
    for r in db.q(conn, "SELECT * FROM injections WHERE phase='pre_match'"):
        groups.setdefault((r["channel_id"], r["match_key"]), []).append(r)

    stats = {"gt_rows": 0, "orphan_injections": 0, "diffs_without_gt": 0}
    rows: list[dict] = []

    for (channel_id, key), items in groups.items():
        diffs = db.q(conn, """
            SELECT id FROM recon_diffs WHERE channel_id=? AND channel_txn_no=?
        """, (channel_id, key))
        if not diffs:
            stats["orphan_injections"] += 1
            continue

        codes = sorted({i["code"] for i in items})
        actions: list[str] = []
        for i in items:
            if i["correct_action"] and i["correct_action"] not in actions:
                actions.append(i["correct_action"])
        statuses = [i["expected_status"] for i in items if i["expected_status"]]
        expected = max(statuses, key=lambda s: _SEVERITY.get(s, 0)) if statuses else "closed"
        substantive = [c for c in codes if c != "D19"]
        explanation = "\n".join(f"[{i['code']}] {i['explanation']}" for i in items)

        for d in diffs:
            rows.append({
                "diff_id": d["id"],
                "root_causes": db.jdump(codes),
                "correct_actions": db.jdump(actions or ["ESCALATE"]),
                "is_composite": int(len(substantive) > 1),
                "expected_status": expected,
                "explanation": explanation,
                "injected_ref": db.jdump([i["injected_ref"] for i in items]),
            })

    # 同一 diff 可能被多组注入命中（理论上不该发生），去重保底
    seen: set[str] = set()
    unique = []
    for r in rows:
        if r["diff_id"] in seen:
            continue
        seen.add(r["diff_id"])
        unique.append(r)

    db.insert_many(conn, "diff_ground_truth", unique)
    stats["gt_rows"] = len(unique)
    conn.commit()
    return stats


def diffs_without_gt(conn) -> int:
    """没有答案的差错 —— 数据质量指标，必须为 0，否则任务集有洞。"""
    return int(db.scalar(conn, """
        SELECT COUNT(*) FROM recon_diffs d
        LEFT JOIN diff_ground_truth g ON g.diff_id = d.id
        WHERE g.diff_id IS NULL
    """))


def orphan_injections(conn) -> list:
    """产不出差错的注入 —— 说明注入白做了，也要盯住。"""
    out = []
    for r in db.q(conn, "SELECT * FROM injections"):
        if r["code"] in ("D10", "D16"):        # 由扫描发现，不走流水匹配
            continue
        n = db.scalar(conn, """
            SELECT COUNT(*) FROM recon_diffs
            WHERE channel_id=? AND channel_txn_no=?
        """, (r["channel_id"], r["match_key"]))
        if not n:
            out.append(r)
    return out


# --------------------------------------------------------------------------
# post_match 注入：不来自流水匹配的检查（结算合规）
# --------------------------------------------------------------------------

def scan_business_rules(conn, dates: list[str]) -> dict[str, int]:
    """业务规则扫描 —— 两侧记录一致、流水匹配发现不了的差错。

    D10 累计退款超原额：我方和渠道都记了这笔超额退款，流水号能对上，
    只有拿订单原额去比才能发现。真实对账系统同样是匹配 + 规则扫描两条腿。
    """
    diffs: list[dict] = []
    gts: list[dict] = []

    inj_by_order = {r["match_key"]: r for r in db.q(
        conn, "SELECT * FROM injections WHERE code='D10'")}

    rows = db.q(conn, """
        SELECT o.id AS order_id, o.channel_id, o.amount_cents,
               SUM(f.amount_cents) AS refund_total,
               MAX(date(f.refunded_at)) AS last_refund_date
        FROM orders o JOIN refunds f ON f.order_id = o.id AND f.status='success'
        GROUP BY o.id HAVING refund_total > o.amount_cents
    """)
    for r in rows:
        task = db.q1(conn, "SELECT id FROM recon_tasks WHERE channel_id=? AND bill_date=?",
                     (r["channel_id"], r["last_refund_date"]))
        if task is None:
            task = db.q1(conn, "SELECT id FROM recon_tasks WHERE channel_id=? LIMIT 1",
                         (r["channel_id"],))
        if task is None:
            continue
        excess = int(r["refund_total"]) - int(r["amount_cents"])
        diff_id = f"DR{r['order_id']}"
        diffs.append({
            "id": diff_id, "recon_task_id": task["id"], "channel_id": r["channel_id"],
            "bill_date": r["last_refund_date"], "source": "rule_scan",
            "our_ref_type": "order", "our_ref_id": r["order_id"],
            "channel_record_id": None, "channel_txn_no": None,
            "our_ref_signed": 0, "channel_signed": 0,
            "our_gross_cents": r["amount_cents"], "channel_gross_cents": None,
            "diff_cents": excess, "fee_delta_cents": 0,
            "status": "new", "created_at": recon_ts(r["last_refund_date"], 8),
        })
        inj = inj_by_order.get(r["order_id"])
        explanation = inj["explanation"] if inj else (
            f"订单 {r['order_id']} 原额 {r['amount_cents']} 分，累计退款 "
            f"{r['refund_total']} 分，超额 {excess} 分。")
        gts.append({
            "diff_id": diff_id,
            "root_causes": db.jdump(["D10"]),
            "correct_actions": db.jdump(["ESCALATE"]),
            "is_composite": 0,
            "expected_status": "escalated",
            "explanation": f"[D10] {explanation}",
            "injected_ref": db.jdump([inj["injected_ref"]] if inj else []),
        })

    db.insert_many(conn, "recon_diffs", diffs)
    db.insert_many(conn, "diff_ground_truth", gts)
    conn.commit()
    return {"rule_scan_diffs": len(diffs)}


def inject_post_match(conn, dates: list[str], *, limit: int = 6) -> dict[str, int]:
    """D16：商户不允许垫资，但存在未平差错时结算单已打款。"""
    made = 0
    log: list[dict] = []
    diffs: list[dict] = []
    gts: list[dict] = []

    candidates = db.q(conn, """
        SELECT s.id AS settle_id, s.merchant_id, s.period_start, s.amount_cents,
               m.allow_advance
        FROM settlements s JOIN merchants m ON m.id = s.merchant_id
        WHERE m.allow_advance = 0 AND s.status = 'pending'
          AND s.period_start IN ({marks})
        ORDER BY s.id
    """.format(marks=",".join("?" for _ in dates)), dates)

    for row in candidates:
        if made >= limit:
            break
        open_diffs = db.q(conn, """
            SELECT d.id, d.channel_id, d.recon_task_id FROM recon_diffs d
            JOIN recon_tasks t ON t.id = d.recon_task_id
            WHERE d.bill_date = ? AND d.status = 'new'
            LIMIT 1
        """, (row["period_start"],))
        if not open_diffs:
            continue
        anchor = open_diffs[0]

        conn.execute("UPDATE settlements SET status='paid' WHERE id=?", (row["settle_id"],))
        diff_id = f"DS{row['settle_id']}"
        diffs.append({
            "id": diff_id,
            "recon_task_id": anchor["recon_task_id"],
            "channel_id": anchor["channel_id"],
            "bill_date": row["period_start"],
            "source": "settlement_scan",
            "our_ref_type": "settlement",
            "our_ref_id": row["settle_id"],
            "channel_record_id": None,
            "channel_txn_no": None,
            "our_ref_signed": row["amount_cents"],
            "channel_signed": 0,
            "our_gross_cents": row["amount_cents"],
            "channel_gross_cents": None,
            "diff_cents": row["amount_cents"],
            "fee_delta_cents": 0,
            "status": "new",
            "created_at": recon_ts(row["period_start"], 9),
        })
        explanation = (f"商户 {row['merchant_id']} 的 allow_advance=0（不允许垫资），"
                       f"但 {row['period_start']} 存在未平差错的情况下，结算单 {row['settle_id']}"
                       f"（{row['amount_cents']} 分）已置为 paid。违反结算冻结规则，必须转人工。")
        gts.append({
            "diff_id": diff_id,
            "root_causes": db.jdump(["D16"]),
            "correct_actions": db.jdump(["ESCALATE"]),
            "is_composite": 0,
            "expected_status": "escalated",
            "explanation": f"[D16] {explanation}",
            "injected_ref": db.jdump([row["settle_id"]]),
        })
        log.append({
            "id": f"INJP{made + 1:06d}", "code": "D16", "phase": "post_match",
            "channel_id": anchor["channel_id"], "bill_date": row["period_start"],
            "match_key": row["settle_id"], "group_id": f"GP{made + 1:05d}",
            "correct_action": "ESCALATE", "expected_status": "escalated",
            "explanation": explanation, "injected_ref": row["settle_id"],
        })
        made += 1

    db.insert_many(conn, "recon_diffs", diffs)
    db.insert_many(conn, "diff_ground_truth", gts)
    db.insert_many(conn, "injections", log)
    conn.commit()
    return {"post_match_diffs": made}
