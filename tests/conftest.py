from __future__ import annotations

import pytest

from recon import db
from recon.config import GenerateConfig
from recon.matching import (attach_ground_truth, inject_post_match, reconcile,
                            scan_business_rules)
from recon.world.bill import build_bills, refresh_bill_totals
from recon.world.notices import build_notices
from recon.world.generator import generate
from recon.world.injector import inject_pre_match

START = "2026-07-01"
DAYS = 3


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    """建一次小世界，全套测试共用。固定 seed，所以结果稳定。"""
    path = tmp_path_factory.mktemp("recon") / "test.db"
    cfg = GenerateConfig(seed=7, start_date=START, days=DAYS,
                         orders_per_day=150, inject_count_per_day=100)
    dates = ["2026-07-01", "2026-07-02", "2026-07-03"]
    conn = db.init_db(path, reset=True)
    generate(conn, cfg)
    build_bills(conn, START, DAYS, seed=cfg.seed)
    board = build_notices(conn, cfg.seed, dates)
    inject_pre_match(conn, cfg, dates, board)
    refresh_bill_totals(conn)
    reconcile(conn, dates)
    attach_ground_truth(conn)
    scan_business_rules(conn, dates)
    inject_post_match(conn, dates)
    yield conn
    conn.close()
