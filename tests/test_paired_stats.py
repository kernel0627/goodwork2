"""阶段 6.2 配对统计的定义与数据边界守卫。"""
from __future__ import annotations

import math

import pytest

from recon import archive
from recon.eval import paired_stats as ps


def _run(path, world, outcomes: dict[str, bool], *, solver: str) -> str:
    with archive.Recorder(
            command="paired-test", solver=solver, model="fake",
            config={"repeat": 1}, world_conn=world, archive_path=path) as rec:
        for task_id, outcome in sorted(outcomes.items()):
            rec.record(
                task_id=task_id, diff_id=f"D-{task_id}", kind="review",
                gold_codes=["D21"], pred_codes=["D21"] if outcome else ["D01"],
                attr_exact=outcome, action_exact=outcome)
    return rec.run_id


def test_exact_mcnemar_known_values():
    assert ps.exact_mcnemar_p(0, 0) == 1.0
    assert ps.exact_mcnemar_p(0, 5) == 0.0625
    assert ps.exact_mcnemar_p(5, 0) == 0.0625
    assert ps.exact_mcnemar_p(1, 9) == pytest.approx(22 / 1024)


def test_wilson_interval_is_bounded_and_contains_rate():
    lo, hi = ps.wilson_interval(7, 10)
    assert 0 <= lo < 0.7 < hi <= 1
    perfect = ps.wilson_interval(10, 10)
    assert 0 < perfect[0] < perfect[1]
    assert perfect[1] == pytest.approx(1)


def test_pair_table_and_delta_use_the_same_tasks():
    result = ps.compare_outcomes(
        "A", {"T1": True, "T2": True, "T3": False, "T4": False},
        "B", {"T1": True, "T2": False, "T3": True, "T4": False},
        reps=500, seed=7)
    assert result.n == 4
    assert (result.both_correct, result.only_a_correct,
            result.only_b_correct, result.both_wrong) == (1, 1, 1, 1)
    assert result.accuracy_a == result.accuracy_b == 0.5
    assert result.delta_b_minus_a == 0
    assert result.mcnemar_p == 1


def test_pairing_refuses_silent_intersection():
    with pytest.raises(ps.PairedStatsError, match="禁止静默取交集"):
        ps.make_paired_sample(
            "A", {"T1": True, "T2": False},
            "B", {"T1": True, "T3": False})


def test_paired_bootstrap_is_seeded_and_detects_clear_improvement():
    differences = [1] * 60 + [0] * 40
    a = ps.paired_bootstrap_ci(differences, reps=1000, seed=42)
    b = ps.paired_bootstrap_ci(differences, reps=1000, seed=42)
    assert a == b
    assert 0 < a[0] <= a[1] <= 1


def test_identical_runs_have_zero_delta_interval():
    outcomes = {f"T{i:02d}": i % 2 == 0 for i in range(20)}
    result = ps.compare_outcomes(
        "A", outcomes, "B", outcomes, reps=300, seed=1)
    assert result.delta_b_minus_a == 0
    assert result.delta_ci95 == (0.0, 0.0)
    assert result.mcnemar_p == 1.0


def test_archive_loader_requires_same_world_and_same_gold(tmp_path, world):
    path = tmp_path / "archive.db"
    a = _run(path, world, {"T1": True, "T2": False}, solver="A")
    b = _run(path, world, {"T1": True, "T2": True}, solver="B")
    run_a = ps.load_archive_run(path, a)
    run_b = ps.load_archive_run(path, b)
    sample = ps.make_archive_sample(run_a, run_b)
    assert sample.task_ids == ("T1", "T2")

    conn = archive.connect(path)
    conn.execute("UPDATE runs SET world_fingerprint='different' WHERE run_id=?", (b,))
    conn.commit()
    conn.close()
    changed = ps.load_archive_run(path, b)
    with pytest.raises(ps.PairedStatsError, match="world_fingerprint"):
        ps.make_archive_sample(run_a, changed)


def test_repeated_comparison_uses_hierarchical_bootstrap():
    tasks = {f"T{i:02d}" for i in range(30)}
    samples = []
    for repeat in range(5):
        a = {task: False for task in tasks}
        b = {task: (int(task[1:]) + repeat) % 3 != 0 for task in tasks}
        samples.append(ps.make_paired_sample(
            f"A{repeat}", a, f"B{repeat}", b))

    result = ps.compare_repeated(samples, reps=500, seed=99)
    again = ps.compare_repeated(samples, reps=500, seed=99)
    assert result == again
    assert result.n_pairs == 5 and result.n_tasks == 30
    assert result.mean_delta_b_minus_a > 0
    assert result.hierarchical_ci95[0] > 0


def test_report_and_module_entrypoint_write_reproducible_markdown(
        tmp_path, world):
    path = tmp_path / "archive.db"
    a = _run(path, world, {"T1": True, "T2": False, "T3": False}, solver="A")
    b = _run(path, world, {"T1": True, "T2": True, "T3": False}, solver="B")
    out = tmp_path / "paired.md"
    code = ps.main([
        "--archive", str(path),
        "--pair", a, b,
        "--bootstrap-reps", "300",
        "--seed", "7",
        "--out", str(out),
    ])
    text = out.read_text(encoding="utf-8")
    assert code == 0
    assert "# 配对统计报告" in text
    assert "exact McNemar" in text
    assert "paired bootstrap 95% CI" in text
    assert not math.isnan(ps.exact_mcnemar_p(1, 0))
