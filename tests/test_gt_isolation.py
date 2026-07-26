"""答案隔离 —— 这个测试是防作弊的物理保证。

agent 和规则基线只允许读 AGENT_VISIBLE_TABLES。答案表一旦漏进可见集合，
后面所有指标就都不可信了。这条必须由测试守住，不能靠自觉。
"""
from __future__ import annotations

from recon import db


def test_ground_truth_tables_not_visible_to_agent():
    overlap = db.GROUND_TRUTH_TABLES & db.AGENT_VISIBLE_TABLES
    assert not overlap, f"答案表泄漏进 agent 可见集合：{overlap}"


def test_every_table_is_classified(world):
    """新加表必须显式归类，不能默默游离在两个集合之外。"""
    actual = db.all_tables(world)
    classified = (db.AGENT_VISIBLE_TABLES | db.GROUND_TRUTH_TABLES
                  | db.AGENT_TRACE_TABLES)
    unclassified = actual - classified
    assert not unclassified, (
        f"这些表既不在 agent 可见集合、也不在答案集合里，必须显式归类：{unclassified}")
    stale = classified - actual
    assert not stale, f"集合里列了不存在的表：{stale}"


def test_trace_tables_are_not_agent_visible():
    """agent 的运行轨迹在阶段 2 不该被 agent 自己读到（阶段 4 做 memory 时再开）。"""
    overlap = db.AGENT_TRACE_TABLES & db.AGENT_VISIBLE_TABLES
    assert not overlap, f"轨迹表过早开放给 agent：{overlap}"


def test_answer_columns_live_only_in_ground_truth_tables(world):
    """root_causes / correct_actions / expected_status 这些字段不能出现在业务表里。"""
    answer_cols = {"root_causes", "correct_actions", "expected_status", "correct_action"}
    for table in sorted(db.AGENT_VISIBLE_TABLES):
        cols = {r["name"] for r in db.q(world, f"PRAGMA table_info({table})")}
        leaked = cols & answer_cols
        assert not leaked, f"表 {table} 里出现了答案字段：{leaked}"
