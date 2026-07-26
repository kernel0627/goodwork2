"""阶段 6 holdout 守卫。

这里不调用模型，也不产生正式分数。测试只确认三件事：

1. holdout 的公告措辞与开发语料隔离；
2. 每个设计场景都有足够样本，闸门仍然不漏放；
3. 数据、语料或评测器封存后发生变化会被 seal 拒绝。
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from recon import db
from recon.agent.reviewer import _task_prompt, _txn_amount, _txn_time
from recon.baseline.rules import RuleBaseline
from recon.cli import _holdout_result_payload
from recon.eval.evidence import EvidenceView
from recon.eval.grader import grade_one
from recon.eval.scenarios import ScenarioView
from recon.eval.tasks import load_tasks
from recon.holdout import (ALL_HOLDOUT_TITLES, HOLDOUT_DB, HOLDOUT_VERSION,
                           HoldoutError, build_world, corpus_fingerprint,
                           create_seal, evaluator_fingerprint,
                           set_evaluation_state, verify_seal, world_fingerprint)
from recon.router import route_reason
from recon.world.injector import TEXT_DEPENDENT_CODES
from recon.world.notices import (DELAY_TITLES, FEE_TITLES,
                                 NEAR_MISS_DELAY_TITLES,
                                 NEAR_MISS_FEE_TITLES, RETRACTION_TITLES,
                                 SCOPED_TITLES)


@pytest.fixture(scope="module")
def frozen_holdout(tmp_path_factory):
    root = tmp_path_factory.mktemp("holdout")
    db_path = root / "holdout.db"
    seal_path = root / "holdout.seal.json"
    built = build_world(db_path)
    seal = create_seal(db_path, seal_path)
    conn = db.connect(db_path)
    yield conn, db_path, seal_path, built, seal
    conn.close()


def test_holdout_has_unseen_notice_wording(frozen_holdout):
    conn, _, _, built, _ = frozen_holdout
    titles = {r["title"] for r in db.q(
        conn, "SELECT DISTINCT title FROM channel_notices")}
    dev_titles = (DELAY_TITLES | FEE_TITLES | SCOPED_TITLES
                  | NEAR_MISS_DELAY_TITLES | NEAR_MISS_FEE_TITLES
                  | RETRACTION_TITLES)

    assert titles
    assert titles <= ALL_HOLDOUT_TITLES
    assert titles.isdisjoint(dev_titles)
    assert built["titles"] == len(titles)


def test_holdout_scenario_coverage_is_adequate(frozen_holdout):
    _, _, _, built, _ = frozen_holdout
    assert built["tasks"] > 0
    assert built["text_dependent"] > 0
    assert built["notices"] > 0
    assert built["scenarios"]
    assert all(n >= 5 for n in built["scenarios"].values())
    assert set(built["rule_combinations"]) == {
        "cross_midnight", "partial_amount_retraction",
        "stacked_notice_groups", "policy_conflict",
    }
    assert built["rule_combinations"]["stacked_notice_groups"] >= 4


def test_holdout_gate_never_misses_and_pass_through_is_exact(frozen_holdout):
    conn, _, _, _, _ = frozen_holdout
    ev = EvidenceView(conn)
    rules = RuleBaseline()
    missed: list[str] = []
    bad_pass_through: list[str] = []

    for task in load_tasks(conn):
        sol = rules.solve(task, ev)
        routed, _, _ = route_reason(sol, task, ev)
        if not routed and set(task.gold_codes) & TEXT_DEPENDENT_CODES:
            missed.append(task.task_id)
        if not routed and not grade_one(task, sol).attr_exact:
            bad_pass_through.append(task.task_id)

    assert missed == []
    assert bad_pass_through == []


def test_reviewer_receives_closed_time_and_amount_comparisons(frozen_holdout):
    """规则组合虽未见过，精确比较仍交给确定性代码，模型只读语义。"""
    conn, _, _, _, _ = frozen_holdout
    ev = EvidenceView(conn)
    rules = RuleBaseline()
    scenarios = ScenarioView(conn)
    seen = {"cross": 0, "amount": 0}

    for task in load_tasks(conn):
        scenario = scenarios.classify(task)
        if not scenario or not (
                scenario.startswith("跨午夜")
                or scenario.startswith("部分撤回")):
            continue
        prior = rules.solve(task, ev)
        notices = ev.channel_notices(
            task.channel_id, task.bill_date, as_of=task.as_of)
        prompt = _task_prompt(
            task, prior, notices, _txn_time(ev, task), _txn_amount(ev, task))
        if scenario.startswith("跨午夜"):
            seen["cross"] += 1
            expected = "落在窗内" if "应D21" in scenario else "不在窗内"
            assert expected in prompt
        else:
            seen["amount"] += 1
            expected = "落在区间内" if "应D21" in scenario else "不在区间内"
            assert expected in prompt

    assert seen["cross"] > 0
    assert seen["amount"] > 0


def test_holdout_seal_contains_three_fingerprints(frozen_holdout):
    conn, db_path, seal_path, _, seal = frozen_holdout
    assert seal["version"] == HOLDOUT_VERSION
    assert seal["evaluation"]["status"] == "sealed"
    assert seal["corpus_fingerprint"] == corpus_fingerprint()
    assert seal["evaluator_fingerprint"] == evaluator_fingerprint()
    assert seal["world_fingerprint"] == world_fingerprint(conn)
    assert all(len(seal[key]) == 64 for key in (
        "corpus_fingerprint", "evaluator_fingerprint", "world_fingerprint"))
    assert verify_seal(db_path, seal_path) == seal


def test_holdout_seal_rejects_database_tampering(frozen_holdout):
    conn, db_path, seal_path, _, _ = frozen_holdout
    row = db.q1(conn, "SELECT id, body FROM channel_notices ORDER BY id LIMIT 1")
    assert row is not None
    conn.execute("UPDATE channel_notices SET body=? WHERE id=?",
                 (row["body"] + "（改动）", row["id"]))
    conn.commit()

    with pytest.raises(HoldoutError, match="数据库在封存后被修改"):
        verify_seal(db_path, seal_path)

    conn.execute("UPDATE channel_notices SET body=? WHERE id=?",
                 (row["body"], row["id"]))
    conn.commit()
    verify_seal(db_path, seal_path)


def test_evaluation_state_is_forward_only(frozen_holdout, tmp_path):
    _, _, _, _, seal = frozen_holdout
    copied = tmp_path / "state.seal.json"
    copied.write_text(json.dumps(seal, ensure_ascii=False), encoding="utf-8")

    set_evaluation_state("running", seal_path=copied)
    with pytest.raises(HoldoutError, match="非法 holdout 评测状态迁移"):
        set_evaluation_state("running", seal_path=copied)
    set_evaluation_state("failed", error="simulated", seal_path=copied)
    with pytest.raises(HoldoutError, match="非法 holdout 评测状态迁移"):
        set_evaluation_state("running", seal_path=copied)


def test_completed_seal_fingerprints_report_and_task_results(
        frozen_holdout, tmp_path):
    _, db_path, _, _, seal = frozen_holdout
    copied = tmp_path / "complete.seal.json"
    copied.write_text(json.dumps(seal, ensure_ascii=False), encoding="utf-8")
    report = tmp_path / "report.md"
    results = tmp_path / "results.json"
    report.write_text("# formal result\n", encoding="utf-8")
    results.write_text('{"rows":[]}\n', encoding="utf-8")

    set_evaluation_state("running", seal_path=copied)
    complete = set_evaluation_state(
        "complete", report=str(report), results=str(results),
        seal_path=copied)
    assert complete["evaluation"]["report_fingerprint"]
    assert complete["evaluation"]["results_fingerprint"]
    verify_seal(db_path, copied)

    results.write_text('{"rows":[{"tampered":true}]}\n', encoding="utf-8")
    with pytest.raises(HoldoutError, match="results 审计件在完成后被修改"):
        verify_seal(db_path, copied)


def test_holdout_result_payload_keeps_input_and_gold_separate():
    task = SimpleNamespace(
        task_id="T1", diff_id="D1", gold_codes=("D22",),
        gold_actions=("HOLD_NEXT_BILL",), gold_status="held")

    def grade(pred, exact):
        return SimpleNamespace(
            task_id="T1", pred_codes=(pred,), pred_actions=("HOLD_NEXT_BILL",),
            pred_status="held", attr_exact=exact, action_exact=True,
            status_ok=True, wrong_money_action=False, unauthorized=False,
            missed_escalation=False)

    review = SimpleNamespace(
        task_id="T1",
        messages=[{"role": "user", "content": "只含公告输入"}],
        raw_response='{"covered":true}', parsed_ok=True, error="")
    payload = _holdout_result_payload({
        "tasks": [task],
        "reports": [
            SimpleNamespace(solver="rule", grades=[grade("D05", False)]),
            SimpleNamespace(solver="router", grades=[grade("D22", True)]),
        ],
        "review_results": [review],
        "model": "fake", "mode": "review", "gate": "any",
    })
    row = payload["rows"][0]
    assert row["gold"]["codes"] == ["D22"]
    assert row["candidate"]["attr_exact"] is True
    assert row["model_trace"]["messages"] == review.messages
    assert "D22" not in json.dumps(
        row["model_trace"]["messages"], ensure_ascii=False)


def test_real_holdout_artifacts_are_outside_test_fixture(frozen_holdout):
    """测试永远只碰临时目录，不能意外消耗正式 holdout。"""
    _, db_path, seal_path, _, _ = frozen_holdout
    assert Path(db_path).parent == Path(seal_path).parent
    assert Path(db_path).name == "holdout.db"
    assert Path(seal_path).name == "holdout.seal.json"
    assert Path(db_path) != HOLDOUT_DB
