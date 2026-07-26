"""归档层的守卫。

这些测试守的是四条硬规则。它们**每一条都曾经被违反过**，代价是
「跑了几十轮实验，一条轨迹都没留下」：

1. 只追加 —— 表里不允许出现 DELETE
2. 跨世界重建存活 —— 世界库 --reset 不能动到归档
3. 溯源齐全 —— 分不清实验条件的轨迹，混着训只会互相污染
4. 输入侧不含答案 —— 一旦答案漏进 messages，整批训练数据就废了
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from recon import archive, db
from recon.eval.tasks import load_tasks


@pytest.fixture()
def arc(tmp_path):
    return tmp_path / "archive.db"


def _rec(arc, **kw):
    kw.setdefault("command", "test")
    kw.setdefault("solver", "s")
    kw.setdefault("model", "m")
    return archive.Recorder(archive_path=arc, **kw)


def test_append_only_across_recorders(arc):
    """两次运行 = 两批轨迹叠加，不是覆盖。

    原来的 `persist_runs` 就是「先 DELETE 再写」，所以永远只有最后一次。
    """
    for i in range(3):
        with _rec(arc, config={"round": i}) as r:
            r.record(task_id=f"T{i}", diff_id=f"D{i}", kind="review",
                     messages=[{"role": "user", "content": "x"}],
                     raw_response="{}", gold_codes=["D01"], attr_exact=True)
    st = archive.stats(arc)
    assert st["runs"] == 3
    assert st["trajectories"] == 3, "第二次运行覆盖了第一次 —— 归档不是只追加的"


def test_no_delete_statement_in_archive_module():
    """静态守卫：归档模块里不许出现 DELETE / DROP / UPDATE trajectories。

    靠自觉是不行的 —— 原来的 solver.py 里那两行 DELETE 也是「顺手写的」。
    """
    src = (Path(archive.__file__)).read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines()
                     if not l.strip().startswith("#"))
    lowered = code.lower()
    for bad in ("delete from trajectories", "drop table trajectories",
                "delete from runs", "update trajectories"):
        assert bad not in lowered, f"归档模块出现了破坏性语句：{bad}"


def test_survives_world_rebuild(arc, tmp_path):
    """世界库 init_db(reset=True) 不能动到归档。

    这是最初丢数据的第二个原因：`make build` 整库重建。
    """
    world = tmp_path / "world.db"
    db.init_db(world, reset=True)
    with _rec(arc, config={"seed": 42}) as r:
        r.record(task_id="T1", diff_id="D1", kind="review",
                 messages=[{"role": "user", "content": "x"}], raw_response="{}")
    assert archive.stats(arc)["trajectories"] == 1

    db.init_db(world, reset=True)            # 再来一次整库重建
    assert archive.stats(arc)["trajectories"] == 1, "归档被世界重建波及了"
    assert arc != db.DEFAULT_DB and archive.ARCHIVE_PATH != db.DEFAULT_DB, \
        "归档必须是独立的库文件，否则 --reset 一定会连它一起删"


def test_provenance_recorded(arc):
    """溯源必须齐：没有它就分不清哪些轨迹可比。"""
    conn = db.connect(None)
    try:
        with _rec(arc, config={"seed": 42, "mode": "review"},
                  world_conn=conn) as r:
            r.record(task_id="T1", diff_id="D1", kind="review")
        c2 = archive.connect(arc)
        row = db.q(c2, "SELECT * FROM runs")[0]
        assert row["code_rev"] and row["code_rev"] != ""
        assert row["world_fingerprint"], "没记世界指纹 —— 跨世界的准确率没法比"
        assert row["world_seed"] == 42
        assert json.loads(row["config"])["mode"] == "review"
        assert row["started_at"]
        c2.close()
    finally:
        conn.close()


def test_world_fingerprint_changes_with_world():
    """世界一换，指纹必须变 —— 否则筛不出可比的那批。"""
    conn = db.connect(None)
    try:
        f1 = archive.world_fingerprint(conn)
        assert f1 and "orders=" in f1
    finally:
        conn.close()


def test_gold_never_in_messages(arc):
    """⭐ 输入侧绝不含答案。

    messages 和 gold_* 是两组字段。混着存一次，整批训练数据就废了 ——
    模型会学到「答案就在输入里」。
    """
    with _rec(arc) as r:
        r.record(task_id="T1", diff_id="D1", kind="review",
                 messages=[{"role": "user", "content": "差错 D1，请判断"}],
                 raw_response='{"covered": true}',
                 gold_codes=["D22"], gold_actions=["NO_ACTION"],
                 gold_status="closed_no_action", attr_exact=True)
    conn = archive.connect(arc)
    row = db.q(conn, "SELECT messages, gold_codes FROM trajectories")[0]
    conn.close()
    assert "D22" not in row["messages"], "答案漏进了模型输入侧"
    assert "closed_no_action" not in row["messages"]
    assert "D22" in row["gold_codes"]


def test_archived_prompt_is_independent_of_gold(arc):
    """⭐ 归档下来的提示词必须**与答案无关**。

    「答案字符串没出现在 messages 里」这个检查是不够的 ——
    复核器合法地看得到规则的先验结论（`D01`/`D05` 这些码本来就在里面），
    而先验常常正好等于答案，于是那个检查会误报。

    真正的不变量是：**把 task 上的答案字段抹掉，提示词一个字节都不该变。**
    只要提示词构造碰过答案，这里就会红。
    """
    import dataclasses

    from recon.agent.reviewer import _task_prompt
    from recon.eval.solution import Solution
    from recon.eval.tasks import load_tasks

    conn = db.connect(None)
    try:
        tasks = load_tasks(conn)[:40]
        assert tasks
        for t in tasks:
            prior = Solution(t.task_id, root_causes=["D01"], actions=["REVERSAL"],
                             expected_status="held", confidence=0.8)
            scrubbed = dataclasses.replace(t, gold_codes=("ZZZ",),
                                           gold_actions=("ZZZ",),
                                           gold_status="ZZZ")
            assert _task_prompt(t, prior, [], None) == \
                   _task_prompt(scrubbed, prior, [], None), \
                f"{t.task_id} 的提示词随答案变化 —— 提示词构造读了答案字段"
    finally:
        conn.close()


def test_export_sft_shape_and_isolation(arc, tmp_path):
    """导出格式：input / response / label 三者分离。"""
    with _rec(arc) as r:
        r.record(task_id="T1", diff_id="D1", kind="review",
                 messages=[{"role": "system", "content": "你是复核员"},
                           {"role": "user", "content": "公告正文……"}],
                 raw_response='{"covered": true, "notice_id": "N1"}',
                 pred_codes=["D21"], gold_codes=["D21"], attr_exact=True,
                 scenario="all_day_cover", tokens_in=900, tokens_out=60)
        r.record(task_id="T2", diff_id="D2", kind="review",
                 messages=[{"role": "user", "content": "另一条"}],
                 raw_response='{"covered": false}',
                 pred_codes=["D01"], gold_codes=["D21"], attr_exact=False)

    out = tmp_path / "sft.jsonl"
    n = archive.export_sft(out, path=arc)
    assert n == 2, "默认应该把判错的也导出来 —— 负样本对训练同样有用"

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    r0 = rows[0]
    assert set(r0) == {"messages", "response", "label", "meta"}
    assert r0["messages"][0]["role"] == "system"
    assert r0["response"]
    assert r0["label"]["gold_codes"] == ["D21"]
    assert r0["meta"]["world_fingerprint"] is None or isinstance(
        r0["meta"]["world_fingerprint"], str)
    assert r0["meta"]["code_rev"]
    # 答案不许出现在输入侧
    dumped = json.dumps(r0["messages"], ensure_ascii=False)
    assert r0["label"]["gold_codes"][0] not in dumped

    only_ok = archive.export_sft(tmp_path / "ok.jsonl", path=arc, only_correct=True)
    assert only_ok == 1


def test_export_skips_rows_without_input_or_response(arc, tmp_path):
    """没输入或没输出的行不能进训练集 —— 它们不是样本，是残缺记录。"""
    with _rec(arc) as r:
        r.record(task_id="T1", diff_id="D1", kind="review",
                 messages=None, raw_response="{}")          # 缺输入
        r.record(task_id="T2", diff_id="D2", kind="review",
                 messages=[{"role": "user", "content": "x"}],
                 raw_response=None, parsed_ok=False)        # 缺输出
    assert archive.stats(arc)["trajectories"] == 2, "残缺记录也要存（用于排障）"
    assert archive.export_sft(tmp_path / "o.jsonl", path=arc) == 0, \
        "残缺记录不能进训练集"


def test_failure_rows_are_kept(arc):
    """失败样本必须留下。

    复核输出不合法、动账被 safe_review_failure 剥掉 —— 这些恰好是
    训练最需要的困难负样本。原来它们连内存都没出过。
    """
    with _rec(arc) as r:
        r.record(task_id="T1", diff_id="D1", kind="review",
                 messages=[{"role": "user", "content": "x"}],
                 raw_response='{"covered": "maybe"}', parsed_ok=False,
                 parse_error="covered 不是布尔", error="schema",
                 gold_codes=["D05"], attr_exact=False, wrong_money=False)
    st = archive.stats(arc)
    assert st["trajectories"] == 1
    conn = archive.connect(arc)
    row = db.q(conn, "SELECT * FROM trajectories")[0]
    conn.close()
    assert row["parsed_ok"] == 0
    assert row["parse_error"]


def test_reviewer_result_carries_trajectory():
    """ReviewResult 必须带上 messages / raw_response。

    这是最初丢数据的第三个原因：复核结果压根没落库，
    而复核器是整个项目里最适合做 SFT 的数据。
    """
    from recon.agent.reviewer import ReviewResult
    r = ReviewResult("T1", False, None, "why")
    assert hasattr(r, "messages") and hasattr(r, "raw_response")
    assert hasattr(r, "parsed_ok") and hasattr(r, "diff_id")


def test_solver_working_tables_are_documented():
    """`persist_runs` 里的 DELETE 必须有注释说明「历史在归档层」。

    保留 DELETE 是对的（那两张表是 replay 用的工作表），
    但必须写清楚，否则下一个人又会以为那就是全部历史。
    """
    src = Path("recon/agent/solver.py").read_text(encoding="utf-8")
    i = src.index("DELETE FROM agent_steps")
    assert "archive" in src[max(0, i - 500):i], \
        "persist_runs 的 DELETE 附近必须说明历史归档在哪，否则会被误当成全部历史"
