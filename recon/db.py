"""sqlite3 连接与建表。刻意不用 ORM —— 对账项目里自己写 SQL 是加分项。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "recon.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# --------------------------------------------------------------------------
# ⚠️ Agent 可见性边界
#    agent 和规则基线只允许读 AGENT_VISIBLE_TABLES 里的表。
#    ground truth 表只有判分器可读。由 tests/test_gt_isolation.py 守住。
# --------------------------------------------------------------------------
GROUND_TRUTH_TABLES = frozenset({"diff_ground_truth", "injections"})

# Agent 自己的运行轨迹。既不是业务数据也不是答案，单独归一类。
# 阶段 2 里求解方不读它；阶段 4 做 memory 时才会开放历史处置检索。
AGENT_TRACE_TABLES = frozenset({"agent_runs", "agent_steps"})

AGENT_VISIBLE_TABLES = frozenset({
    "merchants", "channels",
    "orders", "payments", "refunds", "splits", "ledger_entries", "settlements",
    "channel_bills", "channel_bill_records", "channel_notices",
    "recon_tasks", "recon_diffs",
    "adjustments", "approvals",
})


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    db_path = Path(path) if path else DEFAULT_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path: str | Path | None = None, *, reset: bool = False) -> sqlite3.Connection:
    db_path = Path(path) if path else DEFAULT_DB
    if reset and db_path.exists():
        db_path.unlink()
    conn = connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    return conn


def all_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {r["name"] for r in rows}


# --------------------------------------------------------------------------
# 极简写入/查询辅助
# --------------------------------------------------------------------------

def insert(conn: sqlite3.Connection, table: str, row: dict[str, Any]) -> None:
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))


def insert_many(conn: sqlite3.Connection, table: str, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    cols = list(rows[0])
    col_sql = ", ".join(cols)
    marks = ", ".join("?" for _ in cols)
    conn.executemany(
        f"INSERT INTO {table} ({col_sql}) VALUES ({marks})",
        [tuple(r[c] for c in cols) for r in rows],
    )


def q(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    return conn.execute(sql, tuple(params)).fetchall()


def q1(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
    return conn.execute(sql, tuple(params)).fetchone()


def scalar(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = (), default: Any = 0) -> Any:
    row = conn.execute(sql, tuple(params)).fetchone()
    if row is None or row[0] is None:
        return default
    return row[0]


def jdump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def jload(value: str | None) -> Any:
    return json.loads(value) if value else None
