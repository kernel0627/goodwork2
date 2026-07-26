"""端到端：不变量、覆盖率、数据质量。"""
from __future__ import annotations

from recon import db, invariants
from recon.matching import diffs_without_gt, orphan_injections
from recon.world.injector import ALL_ACTIONS, CODES


def test_hard_invariants_hold(world):
    results = invariants.run_all(world)
    hard = invariants.hard_failures(results)
    detail = "\n".join(f"{r.code}: " + "; ".join(r.violations) for r in hard)
    assert not hard, f"必须恒成立的不变量失败了：\n{detail}"


def test_expected_invariant_violations_actually_happen(world):
    """INV1 必须被 D10 违反 —— 否则说明注入没生效，任务集是空壳。"""
    r = invariants.inv1_order_integrity(world)
    assert not r.passed, "INV1 没有被违反，说明 D10（累计退款超原额）没注入成功"


def test_no_orphan_injections(world):
    orphans = orphan_injections(world)
    detail = [f"{o['id']} {o['code']} key={o['match_key']}" for o in orphans[:10]]
    assert not orphans, f"有注入没产出任何差错（白注入）：{detail}"


def test_no_diff_without_answer(world):
    n = diffs_without_gt(world)
    assert n == 0, f"有 {n} 条差错没有答案 —— 任务集有洞，判分会不准"


def test_all_twenty_codes_covered(world):
    counts: dict[str, int] = {}
    for r in db.q(world, "SELECT root_causes FROM diff_ground_truth"):
        for code in db.jload(r["root_causes"]) or []:
            counts[code] = counts.get(code, 0) + 1
    missing = [c for c in CODES if counts.get(c, 0) == 0]
    assert not missing, f"这些差错类型一条都没生成：{missing}"
    assert len(counts) == 22, f"应覆盖 22 类，实际 {len(counts)} 类"


def test_enough_diffs_and_composites(world):
    total = int(db.scalar(world, "SELECT COUNT(*) FROM diff_ground_truth"))
    composite = int(db.scalar(
        world, "SELECT COUNT(*) FROM diff_ground_truth WHERE is_composite=1"))
    assert total >= 300, f"带答案的差错只有 {total} 条，目标 ≥300"
    assert composite >= 60, f"复合差错只有 {composite} 条，目标 ≥60"


def test_all_detection_sources_present(world):
    sources = {r["source"] for r in db.q(world, "SELECT DISTINCT source FROM recon_diffs")}
    assert sources == {"match", "rule_scan", "settlement_scan"}, (
        f"三种检测来源都要有（流水匹配/规则扫描/结算扫描），实际：{sources}")


def test_actions_are_from_the_closed_set(world):
    for r in db.q(world, "SELECT diff_id, correct_actions FROM diff_ground_truth"):
        for a in db.jload(r["correct_actions"]) or []:
            assert a in ALL_ACTIONS, f"{r['diff_id']} 出现未定义的处置动作 {a}"


def test_no_wall_clock_timestamps_leak_in(world):
    """回归测试：曾经用 datetime.now() 打对账时间戳，导致同 seed 两次构建
    只要跨了秒边界就不可复现。所有对账动作时间戳必须从账单日派生。"""
    from datetime import datetime, timedelta
    bad = []
    for r in db.q(world, "SELECT id, bill_date, created_at FROM recon_diffs"):
        want_date = (datetime.strptime(r["bill_date"], "%Y-%m-%d").date()
                     + timedelta(days=1))
        got_date = datetime.fromisoformat(r["created_at"]).date()
        if got_date != want_date:
            bad.append(f"{r['id']} bill_date={r['bill_date']} created_at={r['created_at']}")
    assert not bad, "差错时间戳不是从账单日派生的（疑似墙上时钟）：\n" + "\n".join(bad[:5])


def test_reproducible_with_same_seed(tmp_path):
    """同 seed 两次构建，差错池与答案必须逐字节一致。"""
    import hashlib
    import json

    from recon.config import GenerateConfig
    from recon.matching import (attach_ground_truth, inject_post_match, reconcile,
                                scan_business_rules)
    from recon.world.bill import build_bills, refresh_bill_totals
    from recon.world.generator import generate
    from recon.world.injector import inject_pre_match
    from recon.world.notices import build_notices

    dates = ["2026-07-01", "2026-07-02"]
    prints = []
    for i in range(2):
        cfg = GenerateConfig(seed=99, start_date="2026-07-01", days=2,
                             orders_per_day=80, inject_count_per_day=60)
        conn = db.init_db(tmp_path / f"r{i}.db", reset=True)
        generate(conn, cfg)
        build_bills(conn, "2026-07-01", 2, seed=cfg.seed)
        board = build_notices(conn, cfg.seed, dates)
        inject_pre_match(conn, cfg, dates, board)
        refresh_bill_totals(conn)
        reconcile(conn, dates)
        attach_ground_truth(conn)
        scan_business_rules(conn, dates)
        inject_post_match(conn, dates)
        payload = json.dumps([
            [dict(r) for r in db.q(conn, "SELECT * FROM recon_diffs ORDER BY id")],
            [dict(r) for r in db.q(conn, "SELECT * FROM diff_ground_truth ORDER BY diff_id")],
        ], ensure_ascii=False, sort_keys=True)
        prints.append(hashlib.sha256(payload.encode()).hexdigest())
        conn.close()
    assert prints[0] == prints[1], "同 seed 产出不一致，任务集不可复现"
