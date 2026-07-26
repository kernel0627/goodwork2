"""路由闸门的测试。

这些测试的价值在于**完全不需要模型**：闸门是纯结构化判断，所以「agent 该不该
被调用」这件事可以在零成本下钉死。真正会造成资金损失的失败模式（把 D22 当成
D05 去冲正）在这里就能拦住，不必等跑完一轮 agent 才发现。
"""
from __future__ import annotations

from recon.baseline.rules import RuleBaseline
from recon.eval.evidence import EvidenceView
from recon.eval.grader import grade_one
from recon.eval.solution import UNKNOWN, Solution
from recon.eval.tasks import load_tasks
from recon.router import (TEXT_OVERRIDABLE, RouterSolver, route_reason,
                          route_summary)
from recon.world.injector import TEXT_DEPENDENT_CODES


def _rows(world):
    ev = EvidenceView(world)
    rules = RuleBaseline()
    out = []
    for t in load_tasks(world):
        sol = rules.solve(t, ev)
        routed, reason, _ = route_reason(sol, t, ev)
        out.append((t, sol, routed, reason))
    return out


# ---------------------------------------------------------------- 闸门本身

def test_gate_never_misses_a_text_dependent_task(world):
    """漏放一条需读文本的任务 = 规则必错，而 D22 那类会去动一笔不该动的账。

    这是整个路由架构的安全底线，必须是 100%，不是「足够高」。
    """
    missed = [t.task_id for t, _, routed, _ in _rows(world)
              if not routed and set(t.gold_codes) & TEXT_DEPENDENT_CODES]
    assert missed == [], f"{len(missed)} 条需读文本任务被放行给规则：{missed[:10]}"


def test_passed_through_tasks_are_all_solved_correctly(world):
    """放行部分的正确率决定了路由架构的上限。它必须是 100% ——
    低于 100% 说明闸门漏了一类规则做不到的东西，不是「可以接受的损失」。"""
    bad = [t.task_id for t, sol, routed, _ in _rows(world)
           if not routed and not grade_one(t, sol).attr_exact]
    assert bad == [], f"放行后仍答错 {len(bad)} 条：{bad[:10]}"


def test_gate_is_selective_enough_to_be_worth_it(world):
    """闸门必须真的挡下大部分任务，否则不如全交给 agent。

    这条是防退化的：如果以后有人放宽闸门（比如改成「有公告就路由」），
    路由比例会跳到 70%+，成本优势消失，这个断言会失败。
    """
    rows = _rows(world)
    rate = sum(1 for _, _, routed, _ in rows if routed) / len(rows)
    assert rate < 0.40, f"路由比例 {rate:.1%} 过高，成本优势已经没了"


def test_gate_needs_both_conditions(world):
    """闸门是「结论可被改写」AND「当日有公告」，缺一不可 —— 少一个条件
    都会让路由比例失控（实测：只看公告 71.2%，两个都要 25.2%）。"""
    ev = EvidenceView(world)
    task = load_tasks(world)[0]

    # 有公告的日子里，不可改写的结论不应被路由
    other = Solution(task_id=task.task_id, root_causes=["D09"], confidence=0.9)
    routed, reason, _ = route_reason(other, task, ev)
    assert not routed and "不含可被公告改写" in reason

    # 可改写的结论 + 无公告 也不应被路由
    hit = Solution(task_id=task.task_id, root_causes=["D01"], confidence=0.9)
    routed, reason, n = route_reason(hit, task, ev)
    if not routed:
        assert "无公告" in reason
    else:
        assert n > 0


def test_unknown_always_routes(world):
    """规则说不知道就得路由。当前数据集上这一支从不触发，但去掉它
    就等于假设规则永远有话说。"""
    ev = EvidenceView(world)
    task = load_tasks(world)[0]
    sol = Solution(task_id=task.task_id, root_causes=[UNKNOWN], confidence=0.2)
    routed, reason, _ = route_reason(sol, task, ev)
    assert routed and "未能归因" in reason


def test_overridable_table_matches_the_sop(world):
    """闸门的依据是 SOP 里那张表。SOP 改了而这里没改，闸门就会漏。"""
    sop = EvidenceView(world).policy("diff_sop")
    for structural, with_notice in TEXT_OVERRIDABLE.items():
        assert structural in sop and with_notice in sop
    # SOP 里标注为需读文本的编码，必须全部是某个结构化编码的「有公告」版本
    assert set(TEXT_OVERRIDABLE.values()) == set(TEXT_DEPENDENT_CODES)


# ------------------------------------------------------------ RouterSolver

class _StubInner:
    """记录自己被调用了几次，并返回一个可辨认的结论。"""
    name = "stub"

    def __init__(self):
        self.seen: list[str] = []

    def solve(self, task, ev):
        self.seen.append(task.task_id)
        return Solution(task_id=task.task_id, root_causes=["D21"],
                        actions=["HOLD_NEXT_BILL"], expected_status="held",
                        confidence=0.8, reads=3, rows_read=9, chars_read=1200,
                        steps=5, evidence_refs=["channel_notices:NT0001"])


def test_router_only_calls_inner_on_routed_tasks(world):
    inner = _StubInner()
    router = RouterSolver(inner)
    ev = EvidenceView(world)
    tasks = load_tasks(world)[:60]
    for t in tasks:
        router.solve(t, ev)

    routed = {d.task_id for d in router.decisions if d.routed}
    assert set(inner.seen) == routed
    assert len(routed) < len(tasks), "闸门没挡住任何东西"


def test_router_charges_the_rule_pass_too(world):
    """路由后的成本必须含规则那一趟。agent 的 loop 会 reset_trace，
    不显式加回来就是在偷成本，对比表会读出一个假的成本优势。"""
    ev = EvidenceView(world)
    rules = RuleBaseline()
    router = RouterSolver(_StubInner(), rules=rules)

    target = None
    for t in load_tasks(world):
        sol = rules.solve(t, ev)
        if route_reason(sol, t, ev)[0]:
            target, rule_reads = t, sol.reads
            break
    assert target is not None, "样本里没有被路由的任务"

    out = router.solve(target, ev)
    assert out.root_causes == ["D21"]
    assert out.reads == 3 + rule_reads
    assert out.steps == 5 + 1
    assert "channel_notices:NT0001" in out.evidence_refs


def test_route_summary_counts(world):
    inner = _StubInner()
    router = RouterSolver(inner)
    ev = EvidenceView(world)
    tasks = load_tasks(world)[:40]
    for t in tasks:
        router.solve(t, ev)
    s = route_summary(router.decisions)
    assert s["total"] == len(tasks)
    assert s["routed"] + s["kept"] == s["total"]
    assert 0.0 <= s["routed_rate"] <= 1.0


def test_gate_only_router_refuses_to_solve_a_routed_task(world):
    """没有 inner 时闸门仍可用，但真要路由的任务必须显式报错 ——
    静默退回规则结论会把「需要读公告」这件事悄悄吞掉。"""
    import pytest
    ev = EvidenceView(world)
    router = RouterSolver()          # gate-only
    rules = RuleBaseline()

    routed_task = next(t for t in load_tasks(world)
                       if route_reason(rules.solve(t, ev), t, ev)[0])
    with pytest.raises(RuntimeError, match="没有 inner"):
        router.solve(routed_task, ev)
