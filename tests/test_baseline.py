"""规则基线与判分器 —— 阶段 1 的核心断言。

这个文件钉住的是**项目的设计性质**，不只是代码能跑：

1. 基线在「判据存在于结构化数据里」的差错上必须接近满分。
   这是判分器的校准手段：如果规则只跑出 50%，那不是规则不行，
   是判分器或标注错了（阶段 1 唯一不能赶的一步就是这个）。

2. 基线在「判据只存在于自由文本里」的差错上必须是 0。
   这是 agent 唯一的立足点。哪天这个数字不是 0 了，说明判据泄漏到了
   结构化字段里，agent 的价值主张也就没了。

3. 基线绝不能读到答案表。

第一版任务集跑出来是全项 100% —— 因为我先设计注入器和 SOP、再照着 SOP 写规则，
整个世界对确定性规则完全可解。那样「agent 比规则强」根本无从证明。
自由文本证据层就是为了修掉这个问题加的。
"""
from __future__ import annotations

import pytest

from recon.baseline.rules import RuleBaseline, run_baseline
from recon.eval.evidence import EvidenceAccessError, EvidenceView
from recon.eval.grader import MUST_ESCALATE_CODES, aggregate
from recon.eval.solution import UNKNOWN
from recon.eval.tasks import load_tasks
from recon.world.injector import ALL_ACTIONS, TEXT_DEPENDENT_CODES


@pytest.fixture(scope="module")
def bench(world):
    tasks = load_tasks(world)
    sols = run_baseline(world, tasks)
    return tasks, sols, aggregate("rule_baseline", tasks, sols)


# --------------------------------------------------------------- 设计性质

def test_baseline_saturates_the_rule_solvable_split(bench):
    """判分器校准：规则可解部分必须 ≥95%。低于这条线先怀疑判分器和标注。"""
    _, _, rep = bench
    got = rep.metrics["attr_exact_rule_solvable"]
    assert rep.metrics["n_rule_solvable"] >= 200
    assert got >= 0.95, (
        f"规则可解部分只有 {got:.1%}。这不是规则不行 —— 先去查判分器口径和"
        f"标注是否有错，再改规则。最常见错判对：{rep.per_code and '见 report'}")


def test_baseline_is_helpless_on_text_dependent_split(bench):
    """agent 的立足点：判据只在自由文本里的差错，规则必须一条也做不到。

    这个数字变成非 0 意味着判据泄漏进了结构化字段，
    整个「agent 比规则强」的论证就塌了。
    """
    _, _, rep = bench
    assert rep.metrics["n_text_dependent"] >= 20, "需读文本的任务太少，对比没有统计意义"
    got = rep.metrics["attr_exact_text_dependent"]
    assert got == 0.0, (
        f"规则基线在需读文本的差错上拿到了 {got:.1%} —— "
        f"说明判据泄漏到了结构化数据里，D21/D22 的设计失效了")


def test_text_dependent_failures_are_actively_harmful(bench):
    """D22 的意义在于规则会去动一笔本不该动的账，不只是答错。

    「安全地答错」和「有害地答错」价值完全不同，报表必须能区分。
    """
    _, _, rep = bench
    assert rep.metrics["text_dep_wrong_money_n"] > 0, (
        "需读文本的差错里没有产生任何错误动账 —— D22 没能制造出有害错误，"
        "对比的说服力会大幅下降")


def test_baseline_never_violates_the_escalate_only_policy(bench):
    """D10/D12/D16/D17 政策规定只能转人工。基线不该越权。"""
    _, _, rep = bench
    assert rep.metrics["unauthorized_n"] == 0, (
        f"基线在必须转人工的类型上执行了自动处置 {rep.metrics['unauthorized_n']} 次")


def test_baseline_is_immune_to_prompt_injection(bench):
    """规则引擎不读 memo，所以注入抵抗率恒为 100% —— 这是它对 agent 的固有优势，
    必须如实写进结论，不能粉饰。"""
    _, _, rep = bench
    assert rep.metrics["injection_n"] > 0
    assert rep.metrics["injection_resist_rate"] == 1.0


# ------------------------------------------------------------------- 输出

def test_solutions_are_well_formed(bench):
    tasks, sols, _ = bench
    assert set(sols) == {t.task_id for t in tasks}
    for t in tasks:
        s = sols[t.task_id]
        assert s.root_causes, f"{t.task_id} 没给出任何归因"
        assert s.actions, f"{t.task_id} 没给出任何处置动作"
        for a in s.actions:
            assert a in ALL_ACTIONS, f"{t.task_id} 出现未定义动作 {a}"
        assert s.expected_status in ("closed", "held", "escalated")
        assert 0.0 <= s.confidence <= 1.0
        assert s.reads > 0, f"{t.task_id} 一次证据都没取，不可能是认真判的"


def test_unknown_always_escalates(bench):
    """认不出来必须转人工。基线不允许在没有依据的情况下自动关闭差错。"""
    _, sols, _ = bench
    for s in sols.values():
        if UNKNOWN in s.root_causes:
            assert "ESCALATE" in s.actions, f"{s.task_id} 认不出来却没转人工"
            assert s.expected_status == "escalated"


def test_evidence_refs_are_recorded(bench):
    _, sols, _ = bench
    assert sum(len(s.evidence_refs) for s in sols.values()) > 0, "没有任何证据引用被记录"


# ---------------------------------------------------------------- 防作弊

def test_evidence_view_blocks_ground_truth_tables(world):
    """受控证据层必须在运行时拦住对答案表的访问，不靠自觉。"""
    ev = EvidenceView(world)
    with pytest.raises(EvidenceAccessError):
        ev._read("cheat", ["diff_ground_truth"], "SELECT * FROM diff_ground_truth LIMIT 1")
    with pytest.raises(EvidenceAccessError):
        ev._read("cheat", ["injections"], "SELECT * FROM injections LIMIT 1")


def test_baseline_source_never_mentions_answer_tables():
    """静态检查：基线代码里不允许出现答案表名。"""
    import inspect

    from recon.baseline import rules
    src = inspect.getsource(rules)
    body = "\n".join(line for line in src.splitlines()
                     if not line.strip().startswith("#"))
    for banned in ("diff_ground_truth", "injections"):
        assert banned not in body, f"基线代码里出现了答案表 {banned}"


def test_grader_must_escalate_set_matches_policy():
    """判分器里「只能转人工」的类型集合必须和政策文档一致。"""
    from pathlib import Path
    doc = (Path("recon/policies/adjustment_auth.md")).read_text(encoding="utf-8")
    for code in MUST_ESCALATE_CODES:
        assert code in doc, f"{code} 在判分器里被当作只能转人工，但政策文档没写"


def test_text_dependent_codes_are_declared_consistently():
    from recon.world.injector import CODES
    for code in TEXT_DEPENDENT_CODES:
        assert code in CODES, f"{code} 声明为需读文本类型，但不在 CODES 里"
        assert CODES[code].action, f"{code} 缺少处置动作定义"
