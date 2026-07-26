"""受控证据访问层 —— 基线和 agent 用同一个接口取证。

三件事：
1. **只允许读 AGENT_VISIBLE_TABLES。** 每个方法显式声明它碰哪些表，运行时断言。
   答案表在这里读不到，所以「作弊」在接口层被强制拦住，不靠自觉。
   ⚠️ 措辞要准：这是**接口层强制隔离**，不是数据库级物理隔离 ——
   同一个 SQLite、同一个连接，只是求解方拿不到绕过 EvidenceView 的路径。
   要做到真物理隔离得上 set_authorizer 或给求解方独立只读库。
2. **每次访问都记轨迹。** 取证次数 / 读到多少行 / 花了多少字符，
   于是「规则基线用了几步、agent 用了几步」是可比的。
3. **返回值预算。** 一次调用最多返回多少行，超了截断并显式告知还有多少。
   阶段 2 的 context 工程要靠这个 —— 一天几千条流水不可能整个塞进 context。

阶段 2 的工具层直接包这一层，不另写一套取数逻辑。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from .. import db
from ..config import CHANNELS

POLICY_DIR = Path(__file__).resolve().parent.parent / "policies"

DEFAULT_ROW_BUDGET = 50


class EvidenceAccessError(RuntimeError):
    pass


@dataclass
class ReadRecord:
    """一次取证动作。"""
    method: str
    tables: tuple[str, ...]
    args: str
    rows: int
    truncated: bool = False
    chars: int = 0


@dataclass
class EvidenceView:
    conn: Any
    row_budget: int = DEFAULT_ROW_BUDGET
    trace: list[ReadRecord] = field(default_factory=list)

    # ------------------------------------------------------------ 内部
    def _read(self, method: str, tables: Sequence[str], sql: str,
              params: Iterable[Any] = (), *, args: Any = None,
              budget: int | None = None) -> list:
        illegal = set(tables) - db.AGENT_VISIBLE_TABLES
        if illegal:
            raise EvidenceAccessError(
                f"{method} 试图读取不可见的表 {illegal}；"
                f"答案表与注入日志对求解方永久不可见")
        rows = db.q(self.conn, sql, params)
        cap = self.row_budget if budget is None else budget
        truncated = len(rows) > cap
        out = rows[:cap]
        chars = len(json.dumps([dict(r) for r in out], ensure_ascii=False, default=str))
        self.trace.append(ReadRecord(method, tuple(tables),
                                     json.dumps(args, ensure_ascii=False, default=str),
                                     len(out), truncated, chars))
        return out

    @property
    def reads(self) -> int:
        return len(self.trace)

    @property
    def rows_read(self) -> int:
        return sum(t.rows for t in self.trace)

    @property
    def chars_read(self) -> int:
        return sum(t.chars for t in self.trace)

    def reset_trace(self) -> None:
        self.trace = []

    # ------------------------------------------------------- 差错与主数据
    def diff(self, diff_id: str):
        rows = self._read("diff", ["recon_diffs"],
                          "SELECT * FROM recon_diffs WHERE id=?", (diff_id,),
                          args=diff_id)
        return rows[0] if rows else None

    def recon_task(self, task_id: str):
        rows = self._read("recon_task", ["recon_tasks"],
                          "SELECT * FROM recon_tasks WHERE id=?", (task_id,), args=task_id)
        return rows[0] if rows else None

    def channel(self, channel_id: str):
        """渠道规则。也可以直接用 config.CHANNELS，这里走表是为了留下取证轨迹。"""
        rows = self._read("channel", ["channels"],
                          "SELECT * FROM channels WHERE id=?", (channel_id,), args=channel_id)
        return rows[0] if rows else None

    def merchant(self, merchant_id: str):
        rows = self._read("merchant", ["merchants"],
                          "SELECT * FROM merchants WHERE id=?", (merchant_id,), args=merchant_id)
        return rows[0] if rows else None

    # ----------------------------------------------------------- 渠道侧
    def channel_records_by_txn(self, txn: str) -> list:
        """按流水号取渠道明细 —— **跨所有账单日**。

        跨日期是关键：D01（记录被删）和 D09/D14（记录只是跑到了别的账单日）
        全靠这个区分。如果这里按账单日过滤，两类就永久不可分了。
        """
        if not txn:
            return []
        return self._read("channel_records_by_txn", ["channel_bill_records", "channel_bills"], """
            SELECT r.*, b.bill_date, b.file_seq, b.received_at
            FROM channel_bill_records r JOIN channel_bills b ON b.id = r.bill_id
            WHERE r.channel_txn_no = ? ORDER BY b.bill_date, r.id
        """, (txn,), args=txn)

    def channel_records_by_amount_time(self, channel_id: str, amount_cents: int,
                                       around: str, window_minutes: int = 30) -> list:
        """按金额 + 时间窗找渠道明细。重复支付（D07）没有可用的流水号线索，只能这样认。"""
        return self._read("channel_records_by_amount_time",
                          ["channel_bill_records", "channel_bills"], """
            SELECT r.*, b.bill_date FROM channel_bill_records r
            JOIN channel_bills b ON b.id = r.bill_id
            WHERE r.channel_id = ? AND r.amount_cents = ?
              AND ABS(strftime('%s', r.occurred_at) - strftime('%s', ?)) <= ?
            ORDER BY r.occurred_at
        """, (channel_id, amount_cents, around, window_minutes * 60),
            args={"channel": channel_id, "amount": amount_cents,
                  "around": around, "window_min": window_minutes})

    def channel_notices(self, channel_id: str, bill_date: str | None = None,
                        as_of: str | None = None) -> list:
        """⭐ 渠道公告 —— 自由文本证据。

        这是「规则做不到、模型能做到」的分界点。D21/D22 的结构化证据分别和
        D01/D05 完全相同，唯一的判据在公告正文里。公告表里没有任何结构化标签，
        所以关键词匹配也绕不过去 —— 必须真的读懂内容，还要能把干扰公告排除掉。
        """
        if bill_date:
            # ⚠️ as_of 过滤是必须的，不是可选的：公告在账单日次日 09:30 发布，
            #    而对账任务 07:00 就开始跑。不过滤 published_at 就等于让求解方
            #    读到了两个半小时后才存在的信息 —— 未来信息泄漏。
            sql = """
                SELECT * FROM channel_notices
                WHERE channel_id = ?
                  AND effective_from <= ? AND COALESCE(effective_to, effective_from) >= ?
            """
            params = [channel_id, bill_date, bill_date]
            if as_of:
                sql += " AND published_at <= ?"
                params.append(as_of)
            sql += " ORDER BY published_at, id"
            return self._read("channel_notices", ["channel_notices"], sql, params,
                              args=[channel_id, bill_date, as_of])
        return self._read("channel_notices", ["channel_notices"],
                          "SELECT * FROM channel_notices WHERE channel_id=? ORDER BY published_at",
                          (channel_id,), args=channel_id)

    def bill(self, channel_id: str, bill_date: str):
        rows = self._read("bill", ["channel_bills"],
                          "SELECT * FROM channel_bills WHERE channel_id=? AND bill_date=?",
                          (channel_id, bill_date), args=[channel_id, bill_date])
        return rows[0] if rows else None

    # ------------------------------------------------------------ 我方侧
    def payment_by_txn(self, txn: str):
        if not txn:
            return None
        rows = self._read("payment_by_txn", ["payments"],
                          "SELECT * FROM payments WHERE channel_txn_no=? ORDER BY id",
                          (txn,), args=txn)
        return rows[0] if rows else None

    def refund_by_txn(self, txn: str):
        if not txn:
            return None
        rows = self._read("refund_by_txn", ["refunds"],
                          "SELECT * FROM refunds WHERE channel_txn_no=? ORDER BY id",
                          (txn,), args=txn)
        return rows[0] if rows else None

    def payment(self, payment_id: str):
        rows = self._read("payment", ["payments"],
                          "SELECT * FROM payments WHERE id=?", (payment_id,), args=payment_id)
        return rows[0] if rows else None

    def refund(self, refund_id: str):
        rows = self._read("refund", ["refunds"],
                          "SELECT * FROM refunds WHERE id=?", (refund_id,), args=refund_id)
        return rows[0] if rows else None

    def order(self, order_id: str):
        rows = self._read("order", ["orders"],
                          "SELECT * FROM orders WHERE id=?", (order_id,), args=order_id)
        return rows[0] if rows else None

    def payments_by_order(self, order_id: str) -> list:
        return self._read("payments_by_order", ["payments"],
                          "SELECT * FROM payments WHERE order_id=? ORDER BY id",
                          (order_id,), args=order_id)

    def refunds_by_order(self, order_id: str) -> list:
        return self._read("refunds_by_order", ["refunds"],
                          "SELECT * FROM refunds WHERE order_id=? ORDER BY requested_at",
                          (order_id,), args=order_id)

    def payments_by_amount_time(self, channel_id: str, amount_cents: int,
                                around: str, window_minutes: int = 30) -> list:
        return self._read("payments_by_amount_time", ["payments"], """
            SELECT * FROM payments
            WHERE channel_id=? AND amount_cents=? AND paid_at IS NOT NULL
              AND ABS(strftime('%s', paid_at) - strftime('%s', ?)) <= ?
            ORDER BY paid_at
        """, (channel_id, amount_cents, around, window_minutes * 60),
            args={"channel": channel_id, "amount": amount_cents, "around": around})

    def splits(self, order_id: str) -> list:
        return self._read("splits", ["splits"],
                          "SELECT * FROM splits WHERE order_id=? ORDER BY id",
                          (order_id,), args=order_id)

    def settlement(self, settlement_id: str):
        rows = self._read("settlement", ["settlements"],
                          "SELECT * FROM settlements WHERE id=?", (settlement_id,),
                          args=settlement_id)
        return rows[0] if rows else None

    def open_diffs(self, bill_date: str, *, exclude: str | None = None,
                   until: str | None = None) -> list:
        """结算期间内的未平差错。

        ⚠️ 原实现只查 `bill_date = ?` 一天。结算单有 period_start~period_end，
           周结渠道跨 7 天 —— 只查首日的语义是错的。
        """
        end = until or bill_date
        return self._read("open_diffs", ["recon_diffs"], """
            SELECT id, channel_id, bill_date, diff_cents, status FROM recon_diffs
            WHERE bill_date BETWEEN ? AND ? AND status='new'
              AND id != COALESCE(?, '')
        """, (bill_date, end, exclude), args=[bill_date, end, exclude])

    # ---------------------------------------------------------- 政策文档
    def policy_list(self) -> list[str]:
        names = sorted(p.stem for p in POLICY_DIR.glob("*.md"))
        self.trace.append(ReadRecord("policy_list", (), "", len(names), False, 0))
        return names

    def policy(self, name: str) -> str:
        path = POLICY_DIR / f"{name}.md"
        if not path.exists():
            raise EvidenceAccessError(
                f"没有这份政策文档：{name}；可用：{sorted(p.stem for p in POLICY_DIR.glob('*.md'))}")
        text = path.read_text(encoding="utf-8")
        self.trace.append(ReadRecord("policy", (), name, 1, False, len(text)))
        return text

    # ------------------------------------------------------------- 便捷
    def channel_cfg(self, channel_id: str):
        """渠道的结构化规则（费率/舍入/日切/口径/退款方式）。"""
        return CHANNELS[channel_id]


def hours_between(a: str | None, b: str | None) -> float | None:
    if not a or not b:
        return None
    return abs((datetime.fromisoformat(a) - datetime.fromisoformat(b)).total_seconds()) / 3600.0
