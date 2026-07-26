"""高风险处置工作流：提案、审批、幂等执行与挂起恢复。

这层刻意不暴露给模型直接调用。模型或规则只能提交提案；审批必须来自人类身份，
资金动作只能由受信服务账户执行。所有状态变化都在同一个 SQLite savepoint 内完成。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterator, Sequence
from uuid import uuid4

from . import db
from .config import CHANNELS, ROLES, required_role
from .money import to_gross

VALID_ACTIONS = frozenset({
    "AUTO_WRITEOFF",
    "SUPPLEMENT",
    "REVERSAL",
    "CHANNEL_INQUIRY",
    "HOLD_NEXT_BILL",
    "ESCALATE",
    "DISCARD_DUPLICATE",
})
FUNDS_ACTIONS = frozenset({"SUPPLEMENT", "REVERSAL"})
APPROVAL_ACTIONS = frozenset({"AUTO_WRITEOFF", "SUPPLEMENT", "REVERSAL"})
ZERO_AMOUNT_ACTIONS = frozenset({
    "CHANNEL_INQUIRY", "HOLD_NEXT_BILL", "ESCALATE", "DISCARD_DUPLICATE",
})
ESCALATE_ONLY_CODES = frozenset({"D10", "D12", "D16", "D17"})
VALID_CODES = frozenset({f"D{i:02d}" for i in range(1, 23)} | {"UNKNOWN"})
ESCALATE_ONLY_CODES = ESCALATE_ONLY_CODES | {"UNKNOWN"}
ROLE_LEVEL = {role: i for i, role in enumerate(ROLES)}
WORKFLOW_NOTE_VERSION = 1


class WorkflowError(RuntimeError):
    """阶段 7 工作流的公共错误基类。"""


class WorkflowValidationError(WorkflowError):
    pass


class WorkflowNotFound(WorkflowError):
    pass


class WorkflowAuthorizationError(WorkflowError):
    pass


class WorkflowStateError(WorkflowError):
    pass


class IdempotencyConflict(WorkflowError):
    pass


@dataclass(frozen=True)
class WorkflowResult:
    adjustment_id: str
    approval_id: str | None
    adjustment_status: str
    diff_status: str
    executed_count: int
    replayed: bool = False
    outcome: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iso(value: str, field: str) -> str:
    try:
        datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise WorkflowValidationError(f"{field} 必须是 ISO8601 时刻") from exc
    return value


def _stable_id(prefix: str, *parts: object) -> str:
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"{prefix}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


@contextmanager
def _atomic(conn: sqlite3.Connection) -> Iterator[None]:
    """可嵌套原子段；不会擅自提交调用方已有的外层事务。"""
    name = f"recon_workflow_{uuid4().hex}"
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield
    except Exception:
        conn.execute(f"ROLLBACK TO {name}")
        conn.execute(f"RELEASE {name}")
        raise
    else:
        conn.execute(f"RELEASE {name}")


def _human(actor: str) -> None:
    if not actor.startswith("human:") or len(actor) <= len("human:"):
        raise WorkflowAuthorizationError("审批身份必须是 human:<name>，模型或服务账号不能审批")


def _executor(actor: str, *, funds: bool) -> None:
    if funds:
        if not actor.startswith("service:") or len(actor) <= len("service:"):
            raise WorkflowAuthorizationError("资金动作只能由 service:<name> 账户执行")
        return
    if not (actor.startswith("service:") or actor.startswith("human:")):
        raise WorkflowAuthorizationError("执行身份必须是 human:<name> 或 service:<name>")


def required_role_for_action(action: str, amount_cents: int) -> str | None:
    """返回需要的最低角色；无需审批的动作返回 None。"""
    if action not in VALID_ACTIONS:
        raise WorkflowValidationError(f"未知处置动作：{action}")
    if action not in APPROVAL_ACTIONS:
        return None
    role = required_role(amount_cents)
    if action == "REVERSAL" and ROLE_LEVEL[role] < ROLE_LEVEL["finance"]:
        return "finance"
    return role


def _normalise_codes(root_causes: Sequence[str]) -> tuple[str, ...]:
    codes = tuple(sorted(set(root_causes)))
    bad = [code for code in codes if code not in VALID_CODES]
    if bad:
        raise WorkflowValidationError(f"未知差错码：{bad}")
    if not codes:
        raise WorkflowValidationError("提案必须携带已校验的 root_causes")
    return codes


def _metadata(root_causes: Sequence[str], reason: str) -> dict[str, Any]:
    return {
        "workflow_note_version": WORKFLOW_NOTE_VERSION,
        "root_causes": list(root_causes),
        "reason": reason,
    }


def _load_metadata(note: str | None) -> dict[str, Any]:
    try:
        value = json.loads(note or "")
    except json.JSONDecodeError as exc:
        raise WorkflowStateError("调账审计元数据损坏，拒绝执行") from exc
    if (not isinstance(value, dict)
            or value.get("workflow_note_version") != WORKFLOW_NOTE_VERSION
            or not isinstance(value.get("root_causes"), list)):
        raise WorkflowStateError("调账审计元数据不完整，拒绝执行")
    return value


def _dump_metadata(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _row_result(conn: sqlite3.Connection, adjustment_id: str, *,
                replayed: bool = False, outcome: str = "") -> WorkflowResult:
    row = db.q1(conn, """
        SELECT a.*, d.status AS diff_status,
               (SELECT id FROM approvals p WHERE p.adjustment_id=a.id LIMIT 1) AS approval_id
        FROM adjustments a JOIN recon_diffs d ON d.id=a.diff_id
        WHERE a.id=?
    """, (adjustment_id,))
    if row is None:
        raise WorkflowNotFound(f"调账 {adjustment_id} 不存在")
    return WorkflowResult(
        adjustment_id=row["id"],
        approval_id=row["approval_id"],
        adjustment_status=row["status"],
        diff_status=row["diff_status"],
        executed_count=int(row["executed_count"]),
        replayed=replayed,
        outcome=outcome,
    )


def propose_adjustment(
    conn: sqlite3.Connection,
    *,
    diff_id: str,
    action: str,
    amount_cents: int,
    root_causes: Sequence[str],
    created_by: str,
    requested_at: str,
    reason: str = "",
) -> WorkflowResult:
    """创建幂等提案；同一差错的同一动作只能有一个规范提案。"""
    requested_at = _iso(requested_at, "requested_at")
    if action not in VALID_ACTIONS:
        raise WorkflowValidationError(f"未知处置动作：{action}")
    try:
        amount_cents = int(amount_cents)
    except (TypeError, ValueError) as exc:
        raise WorkflowValidationError("amount_cents 必须是整数分") from exc
    if not created_by.strip():
        raise WorkflowValidationError("created_by 不能为空")
    if action in FUNDS_ACTIONS and amount_cents == 0:
        raise WorkflowValidationError(f"{action} 是资金动作，amount_cents 不能为 0")
    if action in ZERO_AMOUNT_ACTIONS and amount_cents != 0:
        raise WorkflowValidationError(f"{action} 不动账，amount_cents 必须为 0")

    codes = _normalise_codes(root_causes)
    if set(codes) & ESCALATE_ONLY_CODES and action != "ESCALATE":
        raise WorkflowAuthorizationError(
            f"{sorted(set(codes) & ESCALATE_ONLY_CODES)} 只能 ESCALATE")

    diff = db.q1(conn, "SELECT * FROM recon_diffs WHERE id=?", (diff_id,))
    if diff is None:
        raise WorkflowNotFound(f"差错 {diff_id} 不存在")
    if diff["status"] in {"closed", "escalated"}:
        raise WorkflowStateError(f"差错 {diff_id} 已处于终态 {diff['status']}")

    role = required_role_for_action(action, amount_cents)
    approval_id = _stable_id("APR", diff_id, action) if role else None
    key_token = approval_id or "NO_APPROVAL"
    idempotency_key = f"{diff_id}:{action}:{key_token}"
    adjustment_id = _stable_id("ADJ", diff_id, action)
    meta = _metadata(codes, reason)
    status = "pending_approval" if role else "proposed"

    with _atomic(conn):
        existing = db.q1(
            conn, "SELECT * FROM adjustments WHERE idempotency_key=?",
            (idempotency_key,),
        )
        if existing is not None:
            existing_meta = _load_metadata(existing["note"])
            same = (
                existing["id"] == adjustment_id
                and existing["diff_id"] == diff_id
                and existing["action"] == action
                and int(existing["amount_cents"]) == amount_cents
                and existing["created_by"] == created_by
                and existing_meta.get("root_causes") == list(codes)
                and existing_meta.get("reason", "") == reason
            )
            if not same:
                raise IdempotencyConflict(
                    f"幂等键 {idempotency_key} 已绑定到不同提案")
            return _row_result(conn, existing["id"], replayed=True, outcome="proposal_replayed")

        db.insert(conn, "adjustments", {
            "id": adjustment_id,
            "diff_id": diff_id,
            "action": action,
            "amount_cents": amount_cents,
            "idempotency_key": idempotency_key,
            "status": status,
            "executed_count": 0,
            "created_by": created_by,
            "created_at": requested_at,
            "executed_at": None,
            "note": _dump_metadata(meta),
        })
        if role and approval_id:
            db.insert(conn, "approvals", {
                "id": approval_id,
                "adjustment_id": adjustment_id,
                "required_role": role,
                "status": "pending",
                "decided_by": None,
                "decided_at": None,
                "reason": None,
            })
            conn.execute(
                "UPDATE recon_diffs SET status='pending_approval' WHERE id=?",
                (diff_id,),
            )
    return _row_result(conn, adjustment_id, outcome="proposal_created")


def propose_solution(
    conn: sqlite3.Connection,
    *,
    diff_id: str,
    solution,
    created_by: str,
    requested_at: str,
) -> list[WorkflowResult]:
    """把已校验的规则/Agent ``Solution`` 原子转换为一个或多个提案。

    金额从差错池的结构化字段派生，调用方和模型都不能另报一个金额。UNKNOWN 或
    绝对禁止自动处置的原因会在边界上收敛成单一 ESCALATE 提案。
    """
    if solution.task_id != f"T{diff_id}":
        raise WorkflowValidationError(
            f"solution.task_id={solution.task_id!r} 与差错 {diff_id} 不匹配")
    diff = db.q1(
        conn,
        "SELECT diff_cents, fee_delta_cents FROM recon_diffs WHERE id=?",
        (diff_id,),
    )
    if diff is None:
        raise WorkflowNotFound(f"差错 {diff_id} 不存在")

    codes = tuple(solution.root_causes or ("UNKNOWN",))
    actions = list(dict.fromkeys(solution.actions or ("ESCALATE",)))
    if set(codes) & ESCALATE_ONLY_CODES:
        actions = ["ESCALATE"]
    risk_cents = int(diff["diff_cents"]) or int(diff["fee_delta_cents"])

    results: list[WorkflowResult] = []
    with _atomic(conn):
        for action in actions:
            amount = 0 if action in ZERO_AMOUNT_ACTIONS else risk_cents
            results.append(propose_adjustment(
                conn,
                diff_id=diff_id,
                action=action,
                amount_cents=amount,
                root_causes=codes,
                created_by=created_by,
                requested_at=requested_at,
                reason=solution.notes,
            ))
    return results


def decide_approval(
    conn: sqlite3.Connection,
    *,
    adjustment_id: str,
    approved: bool,
    decided_by: str,
    actor_role: str,
    decided_at: str,
    reason: str,
) -> WorkflowResult:
    """由人类审批人决定提案；角色不足、自批和重写既有决定都会被拒绝。"""
    _human(decided_by)
    decided_at = _iso(decided_at, "decided_at")
    if actor_role not in ROLE_LEVEL:
        raise WorkflowAuthorizationError(f"未知审批角色：{actor_role}")
    if not reason.strip():
        raise WorkflowValidationError("审批理由不能为空")

    with _atomic(conn):
        row = db.q1(conn, """
            SELECT a.*, p.id AS approval_id, p.required_role,
                   p.status AS approval_status, p.decided_by, p.reason AS approval_reason,
                   d.status AS diff_status
            FROM adjustments a
            JOIN approvals p ON p.adjustment_id=a.id
            JOIN recon_diffs d ON d.id=a.diff_id
            WHERE a.id=?
        """, (adjustment_id,))
        if row is None:
            raise WorkflowNotFound(f"调账 {adjustment_id} 没有待审批记录")
        if decided_by == row["created_by"]:
            raise WorkflowAuthorizationError("提案人与审批人必须分离，禁止自批")
        target = "approved" if approved else "rejected"
        if row["approval_status"] != "pending":
            same = (row["approval_status"] == target
                    and row["decided_by"] == decided_by
                    and row["approval_reason"] == reason)
            if same:
                return _row_result(
                    conn, adjustment_id, replayed=True, outcome=f"approval_{target}_replayed")
            raise WorkflowStateError(
                f"审批已有终态 {row['approval_status']}，不能改写")
        if ROLE_LEVEL[actor_role] < ROLE_LEVEL[row["required_role"]]:
            raise WorkflowAuthorizationError(
                f"需要 {row['required_role']}，{actor_role} 权限不足")

        conn.execute("""
            UPDATE approvals
            SET status=?, decided_by=?, decided_at=?, reason=?
            WHERE id=? AND status='pending'
        """, (target, decided_by, decided_at, reason, row["approval_id"]))
        conn.execute(
            "UPDATE adjustments SET status=? WHERE id=?",
            (target, adjustment_id),
        )
        if approved:
            next_status = (row["diff_status"]
                           if row["diff_status"] in {"held", "escalated"}
                           else "resolving")
            conn.execute(
                "UPDATE recon_diffs SET status=? WHERE id=?",
                (next_status, row["diff_id"]),
            )
        else:
            conn.execute(
                "UPDATE recon_diffs SET status='escalated' WHERE id=?",
                (row["diff_id"],),
            )
    return _row_result(conn, adjustment_id, outcome=f"approval_{target}")


def _write_balanced_ledger(
    conn: sqlite3.Connection,
    *,
    adjustment_id: str,
    action: str,
    amount_cents: int,
    occurred_at: str,
) -> None:
    amount = abs(amount_cents)
    debit_account, credit_account = {
        "SUPPLEMENT": ("recon_clearing", "channel_receivable"),
        "REVERSAL": ("adjustment_reversal", "recon_clearing"),
    }[action]
    rows = (
        (f"{adjustment_id}:D", debit_account, "D"),
        (f"{adjustment_id}:C", credit_account, "C"),
    )
    for entry_id, account, direction in rows:
        db.insert(conn, "ledger_entries", {
            "id": entry_id,
            "ref_type": "adjustment",
            "ref_id": adjustment_id,
            "account": account,
            "direction": direction,
            "amount_cents": amount,
            "occurred_at": occurred_at,
        })


def _merge_diff_status(current: str, requested: str) -> str:
    severity = {"closed": 0, "held": 1, "escalated": 2}
    if current not in severity:
        return requested
    return max((current, requested), key=severity.__getitem__)


def execute_adjustment(
    conn: sqlite3.Connection,
    *,
    adjustment_id: str,
    executed_by: str,
    executed_at: str,
) -> WorkflowResult:
    """原子执行提案；重放直接返回首次结果，绝不重复写分录。"""
    executed_at = _iso(executed_at, "executed_at")
    with _atomic(conn):
        row = db.q1(conn, """
            SELECT a.*, d.status AS diff_status
            FROM adjustments a JOIN recon_diffs d ON d.id=a.diff_id
            WHERE a.id=?
        """, (adjustment_id,))
        if row is None:
            raise WorkflowNotFound(f"调账 {adjustment_id} 不存在")
        funds = row["action"] in FUNDS_ACTIONS
        _executor(executed_by, funds=funds)
        if row["status"] == "executed":
            if int(row["executed_count"]) != 1:
                raise WorkflowStateError("已执行提案的 executed_count 异常")
            return _row_result(
                conn, adjustment_id, replayed=True, outcome="execution_replayed")
        if int(row["executed_count"]) != 0:
            raise WorkflowStateError("未完成提案已有执行计数，拒绝继续")

        required = required_role_for_action(row["action"], int(row["amount_cents"]))
        if required:
            approval = db.q1(
                conn,
                "SELECT * FROM approvals WHERE adjustment_id=?",
                (adjustment_id,),
            )
            if approval is None or approval["status"] != "approved":
                raise WorkflowAuthorizationError("审批尚未通过，拒绝执行")
            if row["status"] != "approved":
                raise WorkflowStateError(f"提案状态 {row['status']} 不能执行")
        elif row["status"] != "proposed":
            raise WorkflowStateError(f"提案状态 {row['status']} 不能执行")

        meta = _load_metadata(row["note"])
        codes = set(meta["root_causes"])
        if codes & ESCALATE_ONLY_CODES and row["action"] != "ESCALATE":
            raise WorkflowAuthorizationError(
                f"{sorted(codes & ESCALATE_ONLY_CODES)} 只能 ESCALATE")

        changed = conn.execute("""
            UPDATE adjustments
            SET status='executed', executed_count=1, executed_at=?
            WHERE id=? AND executed_count=0 AND status=?
        """, (executed_at, adjustment_id, row["status"]))
        if changed.rowcount != 1:
            raise WorkflowStateError("并发执行冲突，未产生任何账务变更")

        if funds:
            _write_balanced_ledger(
                conn,
                adjustment_id=adjustment_id,
                action=row["action"],
                amount_cents=int(row["amount_cents"]),
                occurred_at=executed_at,
            )

        requested_status = {
            "CHANNEL_INQUIRY": "held",
            "HOLD_NEXT_BILL": "held",
            "ESCALATE": "escalated",
        }.get(row["action"], "closed")
        target_status = _merge_diff_status(row["diff_status"], requested_status)
        conn.execute(
            "UPDATE recon_diffs SET status=? WHERE id=?",
            (target_status, row["diff_id"]),
        )
        meta["execution"] = {
            "executed_by": executed_by,
            "executed_at": executed_at,
            "target_diff_status": target_status,
        }
        conn.execute(
            "UPDATE adjustments SET note=? WHERE id=?",
            (_dump_metadata(meta), adjustment_id),
        )
    return _row_result(conn, adjustment_id, outcome="executed")


def _our_record(conn: sqlite3.Connection, diff: sqlite3.Row) -> sqlite3.Row | None:
    table = {"payment": "payments", "refund": "refunds"}.get(diff["our_ref_type"])
    if table is None or not diff["our_ref_id"]:
        return None
    return db.q1(conn, f"SELECT * FROM {table} WHERE id=?", (diff["our_ref_id"],))


def _record_matches(diff: sqlite3.Row, our: sqlite3.Row, record: sqlite3.Row) -> bool:
    if record["rec_type"] != diff["our_ref_type"]:
        return False
    if diff["our_ref_type"] == "payment":
        channel = CHANNELS[diff["channel_id"]]
        gross = to_gross(
            int(record["amount_cents"]),
            int(record["fee_cents"]),
            channel.bill_basis,
        )
        if gross != int(our["amount_cents"]):
            return False
        # 手续费维度的挂起（D22）需要看到渠道按我方正确费率更正。
        if int(diff["fee_delta_cents"]) != 0:
            return int(record["fee_cents"]) == int(our["fee_cents"])
        return True
    return int(record["amount_cents"]) == int(our["amount_cents"])


def resume_held_adjustment(
    conn: sqlite3.Connection,
    *,
    adjustment_id: str,
    as_of: str,
    checked_by: str,
) -> WorkflowResult:
    """在次日账单到达后恢复挂起；匹配则关闭，缺失或冲突则升级人工。"""
    as_of = _iso(as_of, "as_of")
    _executor(checked_by, funds=False)
    with _atomic(conn):
        row = db.q1(conn, """
            SELECT a.*, d.status AS diff_status, d.channel_id, d.bill_date,
                   d.our_ref_type, d.our_ref_id, d.channel_txn_no,
                   d.fee_delta_cents
            FROM adjustments a JOIN recon_diffs d ON d.id=a.diff_id
            WHERE a.id=?
        """, (adjustment_id,))
        if row is None:
            raise WorkflowNotFound(f"调账 {adjustment_id} 不存在")
        if row["action"] != "HOLD_NEXT_BILL":
            raise WorkflowStateError("只有 HOLD_NEXT_BILL 可以走次日恢复")
        if row["status"] != "executed" or int(row["executed_count"]) != 1:
            raise WorkflowStateError("挂起动作尚未成功执行")
        if row["diff_status"] in {"closed", "escalated"}:
            return _row_result(
                conn, adjustment_id, replayed=True, outcome="recovery_replayed")
        if row["diff_status"] != "held":
            raise WorkflowStateError(f"差错状态 {row['diff_status']} 不是 held")
        if not row["channel_txn_no"]:
            raise WorkflowStateError("挂起差错没有渠道流水号，无法自动恢复")

        next_date = (date.fromisoformat(row["bill_date"]) + timedelta(days=1)).isoformat()
        bills = db.q(conn, """
            SELECT * FROM channel_bills
            WHERE channel_id=? AND bill_date=? AND received_at<=?
            ORDER BY file_seq, id
        """, (row["channel_id"], next_date, as_of))
        if not bills:
            return _row_result(conn, adjustment_id, outcome="waiting_next_bill")

        records = db.q(conn, """
            SELECT r.*, b.bill_date, b.received_at
            FROM channel_bill_records r
            JOIN channel_bills b ON b.id=r.bill_id
            WHERE r.channel_id=? AND r.channel_txn_no=?
              AND b.bill_date=? AND b.received_at<=?
            ORDER BY b.file_seq, r.id
        """, (row["channel_id"], row["channel_txn_no"], next_date, as_of))
        our = _our_record(conn, row)
        matched = our is not None and next(
            (record for record in records if _record_matches(row, our, record)),
            None,
        )
        target = "closed" if matched is not None else "escalated"
        outcome = "recovered_from_next_bill" if matched is not None else "next_bill_conflict"
        conn.execute(
            "UPDATE recon_diffs SET status=? WHERE id=?",
            (target, row["diff_id"]),
        )
        meta = _load_metadata(row["note"])
        meta["recovery"] = {
            "checked_by": checked_by,
            "checked_at": as_of,
            "next_bill_date": next_date,
            "outcome": outcome,
            "record_id": matched["id"] if matched is not None else None,
        }
        conn.execute(
            "UPDATE adjustments SET note=? WHERE id=?",
            (_dump_metadata(meta), adjustment_id),
        )
    return _row_result(conn, adjustment_id, outcome=outcome)
