"""规则优先路由 —— 规则先跑，只在结论可能被自由文本推翻时才付 agent 的钱。

## 为什么需要它

阶段 3 的消融表读出来的是一个负面结论：把每条差错都交给 agent，整体 exact
只有 58~60%，而纯规则基线是 88.8%。但同一张表里还有一组互补的数字：

    判据在结构化数据里（446 条）   规则 100%    agent ~50%
    判据只在自由文本里（59 条）    规则   0%    agent  85.7%

两边完全互补。那正确的架构就不是「把 agent 调得更聪明」，而是**让规则先跑，
只把规则注定读不到的那部分交给 agent**。

## 闸门依据（来自公开 SOP，不是答案）

`policies/diff_sop.md`「必须查阅渠道公告」一节写明：

    我方单边、渠道明细缺失   无覆盖性公告 -> D01    有覆盖性公告 -> D21
    手续费维度差异           无覆盖性公告 -> D05    有覆盖性公告 -> D22

也就是说**只有 D01 和 D05 两个结论的结构化证据与公告结论完全相同**。规则给出
其它结论（D09/D03/D20/…）时，公告改写不了它，那条差错没有付 agent 的必要。

## 闸门只决定「谁来答」，不决定答案

所以它可以在安全方向上不精确，而两个方向的代价是不对称的：

    漏放（该路由却放行）  -> 规则必错。D22 判成 D05 会去冲正一笔本不该动的账，
                            是有实际资金损失的错误。
    误触（不该路由却路由）-> 只是白花 token。

因此闸门按**召回优先**设计：宁可多路由，绝不漏。实测在全量 505 条上对
需读自由文本任务的召回是 59/59，路由比例 21.8%
（`python -m recon.cli route --all-tasks --dry-run` 可随时复算）。

## 一个已知的、尚未解决的损失

被路由的 110 条里有 51 条是误触 —— 规则本来全对，交给 agent 从零重做反而可能
做坏（agent 在规则可解题上只有 ~50%）。正确的下一步是让 agent 在这条路径上做
**复核**而不是**重解**：把规则的结论作为先验给它，只问「公告是否推翻它」。
那是个二分类，比完整归因容易得多，也能保住这 70 条。本模块的 inner 求解方是
可插拔的，就是为了留出这个位置。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

from . import db
from .baseline.rules import RuleBaseline
from .eval.evidence import EvidenceView
from .eval.solution import Solution
from .eval.tasks import Task

# 结构化证据与「有公告」情形完全相同的结论 —— 见 diff_sop.md §必须查阅渠道公告。
# 这是闸门的全部依据；新增这类编码时必须同步这张表，tests 会钉住这一点。
TEXT_OVERRIDABLE = {"D01": "D21", "D05": "D22"}


class Solver(Protocol):
    name: str

    def solve(self, task: Task, ev: EvidenceView) -> Solution: ...


def wants_prior(inner: object) -> bool:
    """inner 是「复核方」还是「重解方」。

    复核方（NoticeReviewer）声明 wants_prior=True，会拿到规则的结论作为先验，
    输出空间被限制成「维持」或「D01→D21 / D05→D22」；重解方（AgentSolver）
    从零开始，输出空间是全部编码。两者接口不同，这里显式分流而不是
    try/except TypeError —— 后者会把 inner 内部真正的 TypeError 吞掉。
    """
    return bool(getattr(inner, "wants_prior", False))


@dataclass(frozen=True)
class RouteDecision:
    task_id: str
    routed: bool
    reason: str
    rule_codes: tuple[str, ...]
    n_notices: int


def route_reason(sol: Solution, task: Task, ev: EvidenceView) -> tuple[bool, str, int]:
    """要不要把这条交给 agent。纯结构化判断，零 token。

    返回 (是否路由, 理由, 当日公告条数)。
    """
    # 规则自己说不知道 —— 无条件路由。这是诚实的兜底。
    # 注：当前数据集上规则从不输出 UNKNOWN（实测 0/505），所以这一支不承担结果，
    # 但去掉它就等于假设规则永远有话说，那是不成立的。
    if sol.is_unknown:
        return True, "规则未能归因", 0

    hits = sorted(set(sol.root_causes) & set(TEXT_OVERRIDABLE))
    if not hits:
        return False, "结论不含可被公告改写的编码", 0

    notices = ev.channel_notices(task.channel_id, task.bill_date)
    if not notices:
        return False, f"结论含 {'/'.join(hits)}，但该渠道该账单日无公告", 0

    alt = "/".join(TEXT_OVERRIDABLE[c] for c in hits)
    return True, f"结论含 {'/'.join(hits)}，当日 {len(notices)} 条公告可能改判为 {alt}", len(notices)


class RouterSolver:
    """规则 + 任意 inner 求解方的混合。inner 只在闸门触发时被调用。"""

    def __init__(self, inner: Solver | None = None, *,
                 rules: RuleBaseline | None = None, name: str | None = None):
        """inner 可以为 None —— 那样只有闸门可用（decide）。
        闸门是零 token 的，先单独跑它看要花多少钱是常规用法。"""
        self.rules = rules or RuleBaseline()
        self.inner = inner
        self.name = name or f"router({getattr(inner, 'name', 'gate-only')})"
        self.decisions: list[RouteDecision] = []

    def solve(self, task: Task, ev: EvidenceView) -> Solution:
        sol, decided = self.decide(task, ev)
        if not decided.routed:
            return sol
        if self.inner is None:
            raise RuntimeError(
                f"{task.task_id} 需要路由，但 RouterSolver 没有 inner 求解方。"
                "只跑闸门请用 decide()。")
        if wants_prior(self.inner):
            # 复核方自己就是在规则结论上改，成本也已由它累加，不再走 _merge
            return self.inner.solve(task, ev, prior=sol)
        return self._merge(sol, self.inner.solve(task, ev))

    # ------------------------------------------------------------------
    def decide(self, task: Task, ev: EvidenceView) -> tuple[Solution, RouteDecision]:
        """跑规则并做闸门判断，但不调用 inner。批量跑时先做这一步，
        才能只对被路由的任务开线程池。"""
        sol = self.rules.solve(task, ev)
        routed, reason, n = route_reason(sol, task, ev)
        d = RouteDecision(task.task_id, routed, reason,
                          tuple(sol.root_causes), n)
        self.decisions.append(d)
        return sol, d

    @staticmethod
    def _merge(rule_sol: Solution, agent_sol: Solution) -> Solution:
        """结论取 agent 的，成本取两者之和。

        agent 的 loop 会先 reset_trace，所以它报的取证次数不含规则那一趟。
        路由架构的成本必须把规则那一趟算进去，否则对比就是在偷成本。
        """
        agent_sol.reads += rule_sol.reads
        agent_sol.rows_read += rule_sol.rows_read
        agent_sol.chars_read += rule_sol.chars_read
        agent_sol.steps += rule_sol.steps
        seen = set(agent_sol.evidence_refs)
        agent_sol.evidence_refs += [r for r in rule_sol.evidence_refs if r not in seen]
        return agent_sol


# --------------------------------------------------------------------------

def run_router(db_path: str | Path | None, tasks: Iterable[Task], *,
               inner_run: Callable[[list[Task], dict[str, Solution]],
                                   dict[str, Solution]],
               rules: RuleBaseline | None = None,
               merge: bool = True,
               ) -> tuple[dict[str, Solution], list[RouteDecision]]:
    """批量跑。闸门是零成本的，所以先单线程全跑一遍规则，再把被路由的那批
    交给 inner_run 并发处理 —— 而不是每条任务都进线程池。

    inner_run 收 (被路由的任务, 全部规则结论)，返回 {task_id: Solution}。
    第二个参数是给复核方用的先验；重解方忽略它即可。

    merge=True 时把规则那一趟的取证成本加到 inner 的结果上（重解方走这条）；
    复核方自己就是在规则结论上改、成本已经累加过，传 merge=False。
    """
    tasks = list(tasks)
    conn = db.connect(db_path)
    try:
        ev = EvidenceView(conn)
        router = RouterSolver(rules=rules)          # 只用闸门，inner 走批量那条路
        rule_sols: dict[str, Solution] = {}
        routed: list[Task] = []
        for t in tasks:
            sol, d = router.decide(t, ev)
            rule_sols[t.task_id] = sol
            if d.routed:
                routed.append(t)
    finally:
        conn.close()

    out = dict(rule_sols)
    if routed:
        for tid, inner_sol in inner_run(routed, rule_sols).items():
            out[tid] = (RouterSolver._merge(rule_sols[tid], inner_sol)
                        if merge else inner_sol)
    return out, router.decisions


def route_summary(decisions: list[RouteDecision]) -> dict:
    routed = [d for d in decisions if d.routed]
    by_reason: dict[str, int] = {}
    for d in routed:
        key = d.reason.split("，")[0]
        by_reason[key] = by_reason.get(key, 0) + 1
    return {
        "total": len(decisions),
        "routed": len(routed),
        "routed_rate": len(routed) / len(decisions) if decisions else 0.0,
        "kept": len(decisions) - len(routed),
        "by_reason": dict(sorted(by_reason.items(), key=lambda kv: -kv[1])),
    }


__all__ = ["TEXT_OVERRIDABLE", "RouteDecision", "RouterSolver", "route_reason",
           "run_router", "route_summary"]
