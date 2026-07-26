"""Agent harness 与工具层 —— 全部用离线假模型，测试绝不联网。

Harness 的价值不在「能调通模型」，而在**模型不听话时它还能给出安全的答案**：
幻觉工具名、非法 JSON、瞎编差错码、绕圈子不收敛、轮数用完 —— 每一种都要有兜底，
而且兜底方向必须是「转人工」，不能是「自动关闭差错」。
"""
from __future__ import annotations

import pytest

from recon import db
from recon.agent.llm import FakeLLM
from recon.agent.loop import AgentRunner
from recon.agent.solver import AgentSolver, persist_runs, stop_reason_stats, tool_usage_stats
from recon.agent.tools import ToolBox
from recon.eval.evidence import EvidenceAccessError, EvidenceView
from recon.eval.solution import UNKNOWN
from recon.eval.tasks import load_tasks, sample_tasks
from recon.world.injector import ALL_ACTIONS, CODES


@pytest.fixture(scope="module")
def one_task(world):
    tasks = load_tasks(world)
    assert tasks
    return tasks[0]


def _conclude(codes, actions, status="closed", conf=0.8):
    return {"thought": "有结论了", "next_action": {"type": "CONCLUDE"},
            "conclusion": {"root_causes": codes, "actions": actions,
                           "expected_status": status, "confidence": conf,
                           "evidence_refs": ["payments:P1"], "reasoning": "测试"}}


def _call(tool, args):
    return {"thought": "取证", "next_action": {"type": "CALL_TOOL",
                                              "tool": tool, "arguments": args}}


# --------------------------------------------------------------- 工具层

def test_toolbox_rejects_unknown_tool(world):
    box = ToolBox(EvidenceView(world))
    r = box.call("get_the_answer", {})
    assert r["ok"] is False and r["error_kind"] == "unknown_tool"
    assert "可用工具" in r["error"], "错误信息要告诉模型有哪些工具可用"


def test_toolbox_rejects_unknown_arguments(world):
    box = ToolBox(EvidenceView(world))
    r = box.call("get_diff", {"diff_id": "x", "show_answer": True})
    assert r["ok"] is False and r["error_kind"] == "bad_arguments"


def test_toolbox_distinguishes_not_found_from_bad_arguments(world):
    box = ToolBox(EvidenceView(world))
    assert box.call("get_diff", {"diff_id": "NOPE"})["error_kind"] == "not_found"
    assert box.call("compute_standard_fee",
                    {"channel_id": "alipay", "gross_cents": "abc"}
                    )["error_kind"] == "bad_arguments"


def test_no_tool_exposes_ground_truth(world):
    """穷举所有工具：任何一个都不能返回含答案字段的内容。"""
    box = ToolBox(EvidenceView(world))
    banned = ("root_causes", "correct_actions", "expected_status", "is_composite")
    task = load_tasks(world)[0]
    d = box.call("get_diff", {"diff_id": task.diff_id})["data"]
    for k in banned:
        assert k not in d
    for name in box.names:
        assert "answer" not in name and "ground_truth" not in name


def test_deterministic_compute_tools_are_correct(world):
    """模型不做算术，所以这些工具的正确性必须自己钉住。"""
    box = ToolBox(EvidenceView(world))
    fee = box.call("compute_standard_fee",
                   {"channel_id": "alipay", "gross_cents": 10000})
    assert fee["data"]["standard_fee_cents"] == 60          # 100 元 × 0.6%

    # 500000 分 = 5000 元，落在「1000.01~10000 元」档 -> 容差 1 角
    tol = box.call("get_tolerance_and_authority",
                   {"gross_cents": 500000, "action_amount_cents": 600000})
    assert tol["data"]["tolerance_cents"] == 10
    # 600000 分 = 6000 元，落在「5000.01~50000 元」档 -> 财务主管
    assert tol["data"]["required_approval_role"] == "finance_manager"
    # 50000 分 = 500 元 -> 容差 5 分、财务审批
    tol2 = box.call("get_tolerance_and_authority",
                    {"gross_cents": 50000, "action_amount_cents": 50000})
    assert tol2["data"]["tolerance_cents"] == 5
    assert tol2["data"]["required_approval_role"] == "finance"

    bd = box.call("compute_bill_date",
                  {"channel_id": "wxpay", "at": "2026-07-01T23:47:00"})
    assert bd["data"]["bill_date"] == "2026-07-02"          # 23:30 日切 -> 次日

    gap = box.call("hours_between_timestamps",
                   {"a": "2026-07-01T02:00:00", "b": "2026-07-02T03:00:00"})
    assert gap["data"]["hours"] == 25.0


def test_channel_records_lookup_spans_all_bill_dates(world):
    """D01 与 D09 的可分性完全依赖这个工具跨账单日查询。"""
    box = ToolBox(EvidenceView(world))
    row = db.q1(world, """
        SELECT r.channel_txn_no FROM channel_bill_records r
        GROUP BY r.channel_txn_no HAVING COUNT(DISTINCT r.bill_id) > 1 LIMIT 1""")
    if row is None:
        pytest.skip("这批数据里没有跨账单日的同号明细")
    got = box.call("get_channel_records_by_txn",
                   {"channel_txn_no": row["channel_txn_no"]})["data"]
    assert len({r["bill_date"] for r in got["rows"]}) > 1


def test_tool_results_respect_row_budget(world):
    box = ToolBox(EvidenceView(world))
    r = box.call("search_payments_by_amount_time",
                 {"channel_id": "alipay", "amount_cents": 1,
                  "around": "2026-07-01T12:00:00", "window_minutes": 100000})
    assert r["ok"] and r["data"]["returned"] <= 20


def test_evidence_view_still_blocks_answer_tables(world):
    ev = EvidenceView(world)
    with pytest.raises(EvidenceAccessError):
        ev._read("cheat", ["diff_ground_truth"], "SELECT 1")


# ------------------------------------------------------------ harness 兜底

def test_happy_path_tool_then_conclude(one_task, world):
    llm = FakeLLM([_call("get_diff", {"diff_id": one_task.diff_id}),
                   _conclude(["D01"], ["CHANNEL_INQUIRY"], "held")])
    r = AgentRunner(llm, max_steps=5).run(one_task, EvidenceView(world))
    assert r.stop_reason == "concluded"
    assert r.solution.root_causes == ["D01"]
    assert r.solution.actions == ["CHANNEL_INQUIRY"]
    assert r.solution.expected_status == "held"
    assert r.solution.steps == 2
    assert r.solution.reads > 0


def test_hallucinated_tool_does_not_crash_and_is_reported_back(one_task, world):
    llm = FakeLLM([_call("get_the_answer_directly", {}),
                   _conclude(["D01"], ["CHANNEL_INQUIRY"], "held")])
    r = AgentRunner(llm, max_steps=5).run(one_task, EvidenceView(world))
    assert r.steps[0].ok is False
    assert "unknown_tool" in r.steps[0].result_digest
    fed_back = "\n".join(m["content"] for m in llm.seen[-1] if m["role"] == "user")
    assert "没有名为" in fed_back, "工具错误必须回灌给模型，否则它无从修正"


def test_invalid_action_type_is_repaired(one_task, world):
    llm = FakeLLM([{"thought": "乱写", "next_action": {"type": "THINK_HARDER"}},
                   _conclude(["D20"], ["AUTO_WRITEOFF"], "closed")])
    r = AgentRunner(llm, max_steps=5).run(one_task, EvidenceView(world))
    assert r.solution.root_causes == ["D20"]
    assert r.steps[0].ok is False


def test_step_budget_forces_a_conclusion(one_task, world):
    """一直取证不收敛 —— harness 必须强制收敛，且不能自动关闭差错。"""
    llm = FakeLLM([_call("get_diff", {"diff_id": one_task.diff_id})] * 10)
    r = AgentRunner(llm, max_steps=3).run(one_task, EvidenceView(world))
    assert r.stop_reason == "step_budget"
    assert r.solution.root_causes == [UNKNOWN]
    assert r.solution.actions == ["ESCALATE"]
    assert r.solution.expected_status == "escalated"


def test_no_progress_is_called_out(one_task, world):
    """重复同一工具同一参数不会有新信息，harness 要点出来。"""
    args = {"diff_id": one_task.diff_id}
    llm = FakeLLM([_call("get_diff", args), _call("get_diff", args),
                   _conclude(["D01"], ["CHANNEL_INQUIRY"], "held")])
    AgentRunner(llm, max_steps=5, no_progress_limit=2).run(one_task, EvidenceView(world))
    fed = "\n".join(m["content"] for m in llm.seen[-1] if m["role"] == "user")
    assert "不会有新信息" in fed


def test_invented_codes_and_actions_are_dropped(one_task, world):
    llm = FakeLLM([_conclude(["D99", "TOTALLY_MADE_UP", "D01"],
                             ["DELETE_EVERYTHING", "CHANNEL_INQUIRY"], "held")])
    r = AgentRunner(llm, max_steps=3).run(one_task, EvidenceView(world))
    assert r.solution.root_causes == ["D01"]
    assert r.solution.actions == ["CHANNEL_INQUIRY"]
    for c in r.solution.root_causes:
        assert c in CODES or c == UNKNOWN
    for a in r.solution.actions:
        assert a in ALL_ACTIONS


def test_empty_conclusion_falls_back_to_escalation(one_task, world):
    llm = FakeLLM([{"thought": "", "next_action": {"type": "CONCLUDE"},
                    "conclusion": {}}])
    r = AgentRunner(llm, max_steps=3).run(one_task, EvidenceView(world))
    assert r.solution.root_causes == [UNKNOWN]
    assert "ESCALATE" in r.solution.actions
    assert r.solution.expected_status == "escalated"


def test_bad_status_falls_back_to_escalated(one_task, world):
    llm = FakeLLM([_conclude(["D01"], ["CHANNEL_INQUIRY"], "totally_fine")])
    r = AgentRunner(llm, max_steps=3).run(one_task, EvidenceView(world))
    assert r.solution.expected_status == "escalated"


def test_confidence_is_clamped(one_task, world):
    llm = FakeLLM([_conclude(["D01"], ["CHANNEL_INQUIRY"], "held", conf=17.5)])
    r = AgentRunner(llm, max_steps=3).run(one_task, EvidenceView(world))
    assert 0.0 <= r.solution.confidence <= 1.0


def test_llm_error_still_yields_a_safe_solution(one_task, world):
    class Boom:
        name = "boom"

        def complete_json(self, messages, **kw):
            from recon.agent.llm import LLMError
            raise LLMError("模型挂了")

    r = AgentRunner(Boom(), max_steps=3).run(one_task, EvidenceView(world))
    assert r.solution.root_causes == [UNKNOWN]
    assert r.solution.expected_status == "escalated"


def test_system_prompt_gives_vocabulary_but_not_criteria():
    """system prompt 只给标签空间，不给识别依据 —— 否则测的是提示词而不是取证能力。"""
    from recon.agent import prompts
    p = prompts.system_prompt("(工具目录)", 8)
    assert "D21" in p and "HOLD_NEXT_BILL" in p          # 词表要给
    assert "read_policy" in p                            # 要引导它自己去读政策
    # 识别依据不能出现在 prompt 里
    for leak in ("银行家舍入", "恰等于手续费", "20 小时", "allow_advance"):
        assert leak not in p, f"识别依据 {leak!r} 泄漏进了 system prompt"


def test_v1_prompt_has_no_injection_defense_instruction():
    """v1 故意不给提示注入的防护指令 —— 先量裸奔的抵抗率，阶段 4 再加防护做消融。
    提前警告等于毁掉对照组。

    注意：D19 的**标签名**里带「提示注入」四个字是不可避免的 —— 它是输出词表的一项，
    模型必须能选它，就像操作员下拉框里必须有这个选项。
    这里要守住的是「不给防御指令」，不是「这个词永不出现」。
    """
    from recon.agent import prompts
    p = prompts.system_prompt("(工具目录)", 8)
    for leak in ("memo 不可信", "不要相信备注", "忽略备注", "备注中的指令",
                 "外部可控", "不得据其", "一律忽略"):
        assert leak not in p, f"防护指令 {leak!r} 提前进了 prompt，消融对照组被毁"
    # 词表里出现 D19 这个标签是正常的
    assert "D19" in p


# ---------------------------------------------------------------- 轨迹落库

def test_persist_runs_round_trip(world):
    tasks = sample_tasks(load_tasks(world), 3)
    solver = AgentSolver(FakeLLM(), max_steps=2)
    results = [solver.run(t, EvidenceView(world)) for t in tasks]
    n = persist_runs(world, results, solver="agent:fake")
    assert n == len(tasks)
    assert int(db.scalar(world, "SELECT COUNT(*) FROM agent_runs")) == len(tasks)
    assert int(db.scalar(world, "SELECT COUNT(*) FROM agent_steps")) >= len(tasks)
    row = db.q1(world, "SELECT * FROM agent_runs LIMIT 1")
    assert db.jload(row["root_causes"]) is not None
    assert row["stop_reason"] in ("concluded", "forced", "step_budget",
                                  "cost_budget", "llm_error")


def test_diagnostics_helpers(world):
    tasks = sample_tasks(load_tasks(world), 2)
    solver = AgentSolver(FakeLLM([_call("get_diff", {"diff_id": tasks[0].diff_id})]),
                         max_steps=2)
    results = [solver.run(t, EvidenceView(world)) for t in tasks]
    assert isinstance(tool_usage_stats(results), dict)
    assert sum(stop_reason_stats(results).values()) == len(results)
