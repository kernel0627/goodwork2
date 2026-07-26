"""复核器测试。全部用 FakeLLM，绝不联网。

复核器的核心安全性质是**输出空间被结构性限制**：它只能维持规则结论，或者做
D01→D21 / D05→D22 这一种改判。这些测试就是钉住这个性质 —— 一旦有人把它改成
「让模型直接给 root_causes」，闸门误触的那些题就会重新暴露在乱改判的风险下。
"""
from __future__ import annotations

import pytest

from recon.agent.llm import FakeLLM, LLMError
from recon.agent.reviewer import (OVERRIDE, NoticeReviewer, _apply_override,
                                  review_stats)
from recon.baseline.rules import RuleBaseline
from recon.eval.evidence import EvidenceView
from recon.eval.solution import Solution
from recon.eval.tasks import load_tasks
from recon.router import RouterSolver, route_reason, wants_prior
from recon.world.injector import CODES


def _routed_task(world):
    """找一条真正会被闸门路由的任务，连同规则的结论。"""
    ev = EvidenceView(world)
    rules = RuleBaseline()
    for t in load_tasks(world):
        sol = rules.solve(t, ev)
        if route_reason(sol, t, ev)[0] and not sol.is_unknown:
            return t, sol
    pytest.skip("样本里没有被路由的任务")


def _covered(code="D21", nid="NT0001"):
    return {"covered": True, "notice_id": nid, "override": code,
            "reasoning": "公告正文说明当日部分明细未进入对账文件、将随次日补发"}


def _not_covered():
    return {"covered": False, "notice_id": None, "override": None,
            "reasoning": "当日只有系统维护通知，不改变处置"}


# ------------------------------------------------------------ 改判的语义

def test_override_replaces_only_the_covered_code():
    """复合差错里被公告覆盖的只有手续费那一维，其它维度必须原样保留。"""
    prior = Solution(task_id="T1", root_causes=["D05", "D11"],
                     actions=["REVERSAL", "DISCARD_DUPLICATE"],
                     expected_status="closed", confidence=0.9)
    out = _apply_override(prior, "D22")
    assert out.root_causes == ["D22", "D11"]


def test_override_recomputes_actions_so_reversal_disappears():
    """D05 的 REVERSAL 必须消失。留着就还是会去动一笔不该动的账 ——
    这是 D22 存在的全部意义。"""
    prior = Solution(task_id="T1", root_causes=["D05"], actions=["REVERSAL"],
                     expected_status="closed", confidence=0.9)
    out = _apply_override(prior, "D22")
    assert "REVERSAL" not in out.actions
    assert out.actions == [CODES["D22"].action]
    assert out.expected_status == "held"


def test_override_is_a_noop_when_the_source_code_is_absent():
    """规则给的是 D09，模型却说要改判 D21 —— 无从改判，必须原样返回。
    这是防模型乱说的结构性保护。"""
    prior = Solution(task_id="T1", root_causes=["D09"], actions=["HOLD_NEXT_BILL"])
    assert _apply_override(prior, "D21") is prior


def test_override_table_is_the_same_one_the_gate_uses():
    from recon.router import TEXT_OVERRIDABLE
    assert OVERRIDE == TEXT_OVERRIDABLE


# ------------------------------------------------------------ 复核器行为

def test_reviewer_keeps_the_rule_conclusion_when_not_covered(world):
    task, prior = _routed_task(world)
    before = list(prior.root_causes)
    llm = FakeLLM([_not_covered()])
    out = NoticeReviewer(llm).solve(task, EvidenceView(world), prior=prior)
    assert out.root_causes == before
    assert llm.calls == 1, "复核必须是单次调用，不是循环"


def test_reviewer_overrides_when_covered(world):
    task, prior = _routed_task(world)
    src = next(c for c in prior.root_causes if c in OVERRIDE)
    llm = FakeLLM([_covered(OVERRIDE[src])])
    out = NoticeReviewer(llm).solve(task, EvidenceView(world), prior=prior)
    assert OVERRIDE[src] in out.root_causes
    assert out.expected_status == "held"


def test_reviewer_cannot_invent_a_new_attribution(world):
    """模型说「其实是 D17」也没用 —— 输出空间里没有这个选项。

    这是复核相对重解的全部安全优势。去掉这个约束，闸门误触的 51 条就重新
    暴露在乱改判风险下了。
    """
    task, prior = _routed_task(world)
    before = list(prior.root_causes)
    llm = FakeLLM([{"covered": True, "notice_id": "NT0001", "override": "D17",
                    "reasoning": "我觉得是串号"}])
    out = NoticeReviewer(llm).solve(task, EvidenceView(world), prior=prior)
    assert out.root_causes == before


def test_reviewer_falls_back_to_the_rule_on_llm_failure(world):
    """复核挂了就维持规则结论。失败时改判等于凭空动账。"""
    class Boom:
        name = "boom"

        def complete_json(self, messages, **kw):
            raise LLMError("模拟调用失败")

    task, prior = _routed_task(world)
    before = list(prior.root_causes)
    rv = NoticeReviewer(Boom())
    out = rv.solve(task, EvidenceView(world), prior=prior)
    assert out.root_causes == before
    assert rv.results[-1].error and not rv.results[-1].overridden


def test_reviewer_sees_notice_bodies_but_not_the_answer(world):
    task, prior = _routed_task(world)
    llm = FakeLLM([_not_covered()])
    NoticeReviewer(llm).solve(task, EvidenceView(world), prior=prior)
    prompt = llm.seen[0][1]["content"]
    assert "公告" in prompt and str(task.diff_id) in prompt
    for leaked in task.gold_codes:
        if leaked not in prior.root_causes:
            assert leaked not in prompt, f"提示词里泄漏了答案 {leaked}"


def test_reviewer_charges_its_own_call(world):
    """复核也要花钱，不计就是偷成本。"""
    task, prior = _routed_task(world)
    t_in, steps = prior.tokens_in, prior.steps
    out = NoticeReviewer(FakeLLM([_not_covered()], tokens=400)).solve(
        task, EvidenceView(world), prior=prior)
    assert out.tokens_in == t_in + 400
    assert out.steps == steps + 1


# ------------------------------------------------------------ 与路由的接合

def test_router_passes_the_prior_to_a_reviewer(world):
    task, _ = _routed_task(world)
    rv = NoticeReviewer(FakeLLM([_not_covered()]))
    assert wants_prior(rv)
    out = RouterSolver(rv).solve(task, EvidenceView(world))
    assert out.root_causes            # 拿到了规则结论，没有崩在缺 prior 上


def test_router_does_not_call_the_reviewer_on_kept_tasks(world):
    ev = EvidenceView(world)
    rules = RuleBaseline()
    kept = next(t for t in load_tasks(world)
                if not route_reason(rules.solve(t, ev), t, ev)[0])
    llm = FakeLLM([_covered()])
    RouterSolver(NoticeReviewer(llm)).solve(kept, ev)
    assert llm.calls == 0


def test_review_stats_shape():
    from recon.agent.reviewer import ReviewResult
    rs = [ReviewResult("T1", True, "NT1", "x", tokens_in=100, cached_in=60),
          ReviewResult("T2", False, None, "y", tokens_in=100, error="boom")]
    s = review_stats(rs)
    assert s == {"reviewed": 2, "overridden": 1, "kept": 1, "errors": 1,
                 "avg_tokens_in": 100.0, "avg_tokens_out": 0.0,
                 "cached_rate": 0.3}
