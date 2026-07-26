"""求解器入口 —— 并发跑 agent、落轨迹、输出与基线同构的 Solution。

并发的两个坑：
1. sqlite 连接不能跨线程共享 —— 每个 worker 自己开一个只读连接。
2. 写轨迹别在 worker 里写，全部收回主线程一次性落库，避免锁争用。
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

from .. import db
from ..eval.evidence import EvidenceView
from ..eval.solution import Solution
from ..eval.tasks import Task
from .llm import DeepSeekClient, LLMClient
from .loop import AgentRunner, RunResult


class AgentSolver:
    def __init__(self, llm: LLMClient, *, max_steps: int = 14,
                 max_cost_micro_cny: int | None = None):
        self.llm = llm
        self.runner = AgentRunner(llm, max_steps=max_steps,
                                  max_cost_micro_cny=max_cost_micro_cny)
        self.name = f"agent:{getattr(llm, 'name', 'unknown')}"

    def solve(self, task: Task, ev: EvidenceView) -> Solution:
        return self.runner.run(task, ev).solution

    def run(self, task: Task, ev: EvidenceView) -> RunResult:
        return self.runner.run(task, ev)


# --------------------------------------------------------------------------

def run_agent(db_path: str | Path | None, tasks: Iterable[Task], *,
              llm: LLMClient | None = None, max_steps: int = 14,
              workers: int = 8, max_cost_micro_cny: int | None = None,
              progress: Callable[[int, int, RunResult], None] | None = None,
              ) -> tuple[dict[str, Solution], list[RunResult]]:
    tasks = list(tasks)
    client = llm or DeepSeekClient()
    solver = AgentSolver(client, max_steps=max_steps,
                         max_cost_micro_cny=max_cost_micro_cny)

    results: list[RunResult] = []

    def work(task: Task) -> RunResult:
        conn = db.connect(db_path)               # 每线程独立连接
        try:
            return solver.run(task, EvidenceView(conn))
        finally:
            conn.close()

    if workers <= 1:
        for i, t in enumerate(tasks, 1):
            r = work(t)
            results.append(r)
            if progress:
                progress(i, len(tasks), r)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(work, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                results.append(r)
                if progress:
                    progress(i, len(tasks), r)

    results.sort(key=lambda r: r.task_id)
    return {r.task_id: r.solution for r in results}, results


# --------------------------------------------------------------------------

def persist_runs(conn, results: list[RunResult], *, solver: str) -> int:
    """轨迹落库。主线程单写，避免 sqlite 锁争用。"""
    runs, steps = [], []
    for i, r in enumerate(results, 1):
        run_id = f"AR{i:06d}"
        s = r.solution
        runs.append({
            "id": run_id, "task_id": r.task_id, "diff_id": r.diff_id,
            "solver": solver, "model": r.model, "stop_reason": r.stop_reason,
            "steps": s.steps, "reads": s.reads, "chars_read": s.chars_read,
            "tokens_in": s.tokens_in, "tokens_out": s.tokens_out,
            "cost_micro_cny": s.cost_micro_cny, "latency_ms": s.latency_ms,
            "root_causes": db.jdump(s.root_causes), "actions": db.jdump(s.actions),
            "expected_status": s.expected_status, "confidence": s.confidence,
            "evidence_refs": db.jdump(s.evidence_refs), "notes": s.notes,
        })
        for st in r.steps:
            steps.append({
                "id": f"{run_id}S{len(steps):06d}", "run_id": run_id,
                "step_no": st.step_no, "thought": st.thought, "tool": st.tool,
                "arguments": json.dumps(st.arguments, ensure_ascii=False,
                                        default=str) if st.arguments else None,
                "result_digest": st.result_digest, "ok": int(st.ok),
                "tokens_in": st.tokens_in, "tokens_out": st.tokens_out,
                "latency_ms": st.latency_ms,
            })
    conn.execute("DELETE FROM agent_steps")
    conn.execute("DELETE FROM agent_runs")
    db.insert_many(conn, "agent_runs", runs)
    db.insert_many(conn, "agent_steps", steps)
    conn.commit()
    return len(runs)


def tool_usage_stats(results: list[RunResult]) -> dict[str, dict]:
    """哪些工具被用了多少次、失败多少次。用来判断工具设计是否合理。"""
    out: dict[str, dict] = {}
    for r in results:
        for st in r.steps:
            if not st.tool:
                continue
            slot = out.setdefault(st.tool, {"calls": 0, "errors": 0})
            slot["calls"] += 1
            if not st.ok:
                slot["errors"] += 1
    for slot in out.values():
        slot["error_rate"] = slot["errors"] / slot["calls"] if slot["calls"] else 0.0
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["calls"]))


def stop_reason_stats(results: list[RunResult]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results:
        out[r.stop_reason] = out.get(r.stop_reason, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
