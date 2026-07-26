"""轨迹归档 —— 只追加，跨世界重建存活。

## 为什么必须单独一层

原来的轨迹是这么丢的，三处叠加，一条不剩：

1. `persist_runs` 每次先 `DELETE FROM agent_runs / agent_steps` —— 只留最后一次；
2. `make build` 是 `init_db(reset=True)`，**整库重建** —— 连最后一次也没了；
3. **复核结果压根没落库** —— 那 336 条单次复核是最适合做 SFT 的数据，从来没存过。

实测到写这段时为止：`agent_runs: 0 行`。

也就是说：跑了几十轮实验、烧掉的 token 全部产生的轨迹，**一条都没留下**。
哪天要做训练，第一件事会是「把所有实验重跑一遍」。

## 这一层的四条硬规则

1. **独立数据库**（`data/archive.db`）。世界库随时可以 `--reset` 重建，归档不受影响。
2. **只追加，永不删除。** 没有 DELETE，没有 upsert。
3. **存够能重建训练样本的东西**：发给模型的完整 messages、模型原样返回的
   response、当时的答案、判分结果 —— 缺任何一样，这条轨迹就没法变成训练样本。
4. **记全溯源信息**：世界 seed、代码 git rev、模型名、配置、提示词 hash。
   没有它就分不清哪些轨迹是同一个实验条件下产生的，混在一起训只会互相污染。

## 答案怎么存

答案（gold）和输入（messages）**分两个字段存**，不拼在一起。
输入侧永远不含答案 —— 这样导出训练数据时，input 是干净的，
而 label 和判分结果就在旁边。混着存一次，整批数据就废了。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import db

ARCHIVE_PATH = db.PROJECT_ROOT / "data" / "archive.db"

SCHEMA = """
PRAGMA journal_mode = WAL;

-- 一次实验运行（一条命令 = 一条 run）
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    started_at      TEXT NOT NULL,       -- 真实墙上时钟：归档要能按时间排序
    command         TEXT NOT NULL,       -- route / eval-agent / ablate / variance …
    solver          TEXT NOT NULL,
    model           TEXT NOT NULL,
    config          TEXT NOT NULL,       -- json：AgentConfig / 闸门模式等
    world_seed      INTEGER,
    world_fingerprint TEXT,              -- 世界指纹：同指纹的轨迹才可比
    code_rev        TEXT,                -- git rev-parse HEAD（含 -dirty 标记）
    prompt_hash     TEXT,                -- system prompt 的 hash
    n_tasks         INTEGER NOT NULL DEFAULT 0,
    notes           TEXT
);

-- 一条轨迹 = 一个任务在一次 run 里的完整交互
-- ⚠️ 只追加。任何时候都不允许 DELETE / UPDATE。
CREATE TABLE IF NOT EXISTS trajectories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    task_id         TEXT NOT NULL,
    diff_id         TEXT NOT NULL,
    kind            TEXT NOT NULL,       -- review | agent_loop | rule
    step_no         INTEGER NOT NULL DEFAULT 0,   -- 多轮时的轮次；单次复核为 0

    -- ── 输入侧：干净的，绝不含答案 ──
    messages        TEXT,                -- json：发给模型的完整 messages
    tools_offered   TEXT,                -- json：本轮可用工具名

    -- ── 模型输出侧：原样存，不做任何加工 ──
    raw_response    TEXT,
    parsed_ok       INTEGER NOT NULL DEFAULT 1,
    parse_error     TEXT,

    -- ── 结论 ──
    pred_codes      TEXT,
    pred_actions    TEXT,
    pred_status     TEXT,
    confidence      REAL,

    -- ── 答案与判分：和输入分开存 ──
    gold_codes      TEXT,
    gold_actions    TEXT,
    gold_status     TEXT,
    scenario        TEXT,                -- 难点场景分档
    attr_exact      INTEGER,
    action_exact    INTEGER,
    wrong_money     INTEGER,             -- 是否错误动账

    -- ── 成本 ──
    tokens_in       INTEGER NOT NULL DEFAULT 0,
    tokens_out      INTEGER NOT NULL DEFAULT 0,
    cached_in       INTEGER NOT NULL DEFAULT 0,
    latency_ms      INTEGER NOT NULL DEFAULT 0,
    error           TEXT
);
CREATE INDEX IF NOT EXISTS idx_traj_run ON trajectories(run_id);
CREATE INDEX IF NOT EXISTS idx_traj_task ON trajectories(task_id);
CREATE INDEX IF NOT EXISTS idx_traj_scenario ON trajectories(scenario);
CREATE INDEX IF NOT EXISTS idx_traj_exact ON trajectories(attr_exact);
"""


def connect(path: str | Path | None = None):
    p = Path(path) if path else ARCHIVE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = db.connect(p)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


# --------------------------------------------------------------------------
# 溯源
# --------------------------------------------------------------------------

def code_rev() -> str:
    """git HEAD，工作区脏就加 -dirty。

    没有它就分不清「这批轨迹是哪版代码跑的」—— 而代码一改，
    提示词、工具、判分口径都可能变，混着训只会互相污染。
    """
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=db.PROJECT_ROOT, capture_output=True,
                             text=True, timeout=5)
        if rev.returncode != 0:
            return "unknown"
        h = rev.stdout.strip()
        st = subprocess.run(["git", "status", "--porcelain"], cwd=db.PROJECT_ROOT,
                            capture_output=True, text=True, timeout=5)
        return h + ("-dirty" if st.stdout.strip() else "")
    except Exception:
        return "unknown"


def world_fingerprint(conn) -> str:
    """世界指纹 —— 同指纹的轨迹才可比。

    世界一重建（换 seed、改注入器、加公告类型），任务实例就变了，
    跨指纹的准确率数字放在一起比是没有意义的。
    """
    parts = []
    for t in ("orders", "channel_bill_records", "channel_notices",
              "recon_diffs", "diff_ground_truth"):
        parts.append(f"{t}={db.scalar(conn, f'SELECT COUNT(*) FROM {t}')}")
    ids = db.q(conn, "SELECT id FROM recon_diffs ORDER BY id LIMIT 200")
    parts.append("h=" + hashlib.sha256(
        "".join(r["id"] for r in ids).encode()).hexdigest()[:12])
    return ";".join(parts)


# --------------------------------------------------------------------------

@dataclass
class Recorder:
    """一次实验的记录器。用完 close()，或者当上下文管理器用。"""
    command: str
    solver: str
    model: str
    config: dict = field(default_factory=dict)
    world_conn: Any = None
    prompt_hash: str = ""
    notes: str = ""
    archive_path: Path | None = None

    def __post_init__(self):
        from datetime import datetime
        self.conn = connect(self.archive_path)
        # run_id 用真实时间 + 随机后缀。这里**可以**用墙上时钟：
        # 归档不是任务集，不需要可复现，反而需要能按时间排序。
        self.run_id = (datetime.now().strftime("R%Y%m%d-%H%M%S-")
                       + os.urandom(3).hex())
        db.insert(self.conn, "runs", {
            "run_id": self.run_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "command": self.command, "solver": self.solver, "model": self.model,
            "config": json.dumps(self.config, ensure_ascii=False, default=str),
            "world_seed": (self.config or {}).get("seed"),
            "world_fingerprint": (world_fingerprint(self.world_conn)
                                  if self.world_conn is not None else None),
            "code_rev": code_rev(),
            "prompt_hash": self.prompt_hash,
            "n_tasks": 0, "notes": self.notes,
        })
        self.conn.commit()
        self._n = 0

    # ------------------------------------------------------------------
    def record(self, *, task_id: str, diff_id: str, kind: str, step_no: int = 0,
               messages: list[dict] | None = None,
               tools_offered: list[str] | None = None,
               raw_response: str | None = None, parsed_ok: bool = True,
               parse_error: str = "",
               pred_codes: list[str] | None = None,
               pred_actions: list[str] | None = None,
               pred_status: str | None = None, confidence: float | None = None,
               gold_codes: list[str] | None = None,
               gold_actions: list[str] | None = None,
               gold_status: str | None = None, scenario: str | None = None,
               attr_exact: bool | None = None, action_exact: bool | None = None,
               wrong_money: bool | None = None,
               tokens_in: int = 0, tokens_out: int = 0, cached_in: int = 0,
               latency_ms: int = 0, error: str = "") -> None:
        def j(v):
            return json.dumps(v, ensure_ascii=False, default=str) if v is not None else None

        db.insert(self.conn, "trajectories", {
            "run_id": self.run_id, "task_id": task_id, "diff_id": diff_id,
            "kind": kind, "step_no": step_no,
            "messages": j(messages), "tools_offered": j(tools_offered),
            "raw_response": raw_response, "parsed_ok": int(parsed_ok),
            "parse_error": parse_error,
            "pred_codes": j(pred_codes), "pred_actions": j(pred_actions),
            "pred_status": pred_status, "confidence": confidence,
            "gold_codes": j(gold_codes), "gold_actions": j(gold_actions),
            "gold_status": gold_status, "scenario": scenario,
            "attr_exact": None if attr_exact is None else int(attr_exact),
            "action_exact": None if action_exact is None else int(action_exact),
            "wrong_money": None if wrong_money is None else int(wrong_money),
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "cached_in": cached_in, "latency_ms": latency_ms, "error": error,
        })
        self._n += 1
        if self._n % 50 == 0:
            self.conn.commit()

    def close(self) -> int:
        self.conn.execute("UPDATE runs SET n_tasks=? WHERE run_id=?",
                          (self._n, self.run_id))
        self.conn.commit()
        self.conn.close()
        return self._n

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


# --------------------------------------------------------------------------
# 查询与导出
# --------------------------------------------------------------------------

def stats(path: str | Path | None = None) -> dict:
    conn = connect(path)
    out = {
        "runs": db.scalar(conn, "SELECT COUNT(*) FROM runs"),
        "trajectories": db.scalar(conn, "SELECT COUNT(*) FROM trajectories"),
        "tasks_covered": db.scalar(conn, "SELECT COUNT(DISTINCT task_id) FROM trajectories"),
        "with_gold": db.scalar(conn,
            "SELECT COUNT(*) FROM trajectories WHERE gold_codes IS NOT NULL"),
        "with_messages": db.scalar(conn,
            "SELECT COUNT(*) FROM trajectories WHERE messages IS NOT NULL"),
        "correct": db.scalar(conn, "SELECT COUNT(*) FROM trajectories WHERE attr_exact=1"),
        "wrong": db.scalar(conn, "SELECT COUNT(*) FROM trajectories WHERE attr_exact=0"),
        "wrong_money": db.scalar(conn, "SELECT COUNT(*) FROM trajectories WHERE wrong_money=1"),
        "tokens_in": db.scalar(conn, "SELECT COALESCE(SUM(tokens_in),0) FROM trajectories"),
        "tokens_out": db.scalar(conn, "SELECT COALESCE(SUM(tokens_out),0) FROM trajectories"),
    }
    out["by_kind"] = {r["kind"]: r["n"] for r in db.q(conn,
        "SELECT kind, COUNT(*) n FROM trajectories GROUP BY 1 ORDER BY n DESC")}
    out["by_scenario"] = {r["scenario"]: r["n"] for r in db.q(conn,
        "SELECT scenario, COUNT(*) n FROM trajectories "
        "WHERE scenario IS NOT NULL GROUP BY 1 ORDER BY n DESC")}
    out["by_world"] = {r["world_fingerprint"]: r["n"] for r in db.q(conn,
        "SELECT world_fingerprint, COUNT(*) n FROM runs GROUP BY 1 ORDER BY n DESC")}
    out["by_code_rev"] = {r["code_rev"]: r["n"] for r in db.q(conn,
        "SELECT code_rev, COUNT(*) n FROM runs GROUP BY 1 ORDER BY n DESC")}
    conn.close()
    return out


def export_sft(out_path: str | Path, *, path: str | Path | None = None,
               only_correct: bool | None = None,
               world_fingerprint_filter: str | None = None,
               kind: str = "review") -> int:
    """导出成 SFT 用的 JSONL。

    每行：{"messages": [...], "response": "...", "label": {...}, "meta": {...}}

    - `messages` 是当时**原样**发给模型的输入，不含答案
    - `response` 是模型原样返回的内容
    - `label` 是答案 + 判分结果，独立字段
    - `meta` 带溯源（世界指纹 / 代码版本 / 模型 / 场景）

    ⚠️ 训练时**按世界指纹和代码版本筛**，不要把不同实验条件的轨迹混在一起。
    ⚠️ 划分训练/测试要**按公告模板**，不是按任务随机 —— 随机划分会让同一条公告
       同时出现在两边，那是泄漏。见 docs/training_deferred.md。
    """
    conn = connect(path)
    sql = ("SELECT t.*, r.world_fingerprint, r.code_rev, r.model, r.config "
           "FROM trajectories t JOIN runs r ON r.run_id = t.run_id "
           "WHERE t.messages IS NOT NULL AND t.raw_response IS NOT NULL "
           "AND t.kind = ?")
    params: list = [kind]
    if only_correct is not None:
        sql += " AND t.attr_exact = ?"
        params.append(int(only_correct))
    if world_fingerprint_filter:
        sql += " AND r.world_fingerprint = ?"
        params.append(world_fingerprint_filter)
    sql += " ORDER BY t.id"

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for row in db.q(conn, sql, params):
            fh.write(json.dumps({
                "messages": db.jload(row["messages"]),
                "response": row["raw_response"],
                "label": {
                    "gold_codes": db.jload(row["gold_codes"]),
                    "gold_actions": db.jload(row["gold_actions"]),
                    "gold_status": row["gold_status"],
                    "pred_codes": db.jload(row["pred_codes"]),
                    "attr_exact": row["attr_exact"],
                    "action_exact": row["action_exact"],
                    "wrong_money": row["wrong_money"],
                },
                "meta": {
                    "task_id": row["task_id"], "diff_id": row["diff_id"],
                    "scenario": row["scenario"], "kind": row["kind"],
                    "world_fingerprint": row["world_fingerprint"],
                    "code_rev": row["code_rev"], "model": row["model"],
                    "tokens_in": row["tokens_in"], "tokens_out": row["tokens_out"],
                },
            }, ensure_ascii=False) + "\n")
            n += 1
    conn.close()
    return n
