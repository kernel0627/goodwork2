"""难点场景分档 —— 把「公告类差错」按设计意图切开。

## 为什么要单独一层

阶段 5 加了三类难点（部分时段适用 / 近似但不覆盖 / 后续收窄），但整体指标
只掉了 1 个百分点。看聚合数字会以为「难点没起作用」，实际是**两个场景根本没生成出来**：

    部分时段·窗内(应D21)    1 条   ← 时间窗取了 02:00-06:00，那时段几乎没交易
    近似延迟(应D01)         0 条   ← 那些日期被更正公告污染，全归到别的场景去了

**设计了却没生成出来，等于没做。** 而且它不会报错、不会让测试变红，
只会让报表上多出一行「100%」，让人以为难点被解决了。

所以这一层做两件事：
1. 给每条公告类差错打上「它属于哪个设计场景」的标签，报表按场景出数；
2. 被 `tests/test_ground_truth_quality.py` 用来断言**每个场景都有足够样本**，
   场景一旦静默消失就让测试红掉。
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import db
from ..world.notices import (DELAY_TITLES, FEE_TITLES, NEAR_MISS_DELAY_TITLES,
                             NEAR_MISS_FEE_TITLES, RETRACTION_TITLES,
                             SCOPED_TITLES, WINDOWS, minute_of_day)

# 场景名 -> (说明, 最少样本数)
# 最少样本数不是拍的：低于它，一条任务翻转就超过 10 个百分点，
# 这个场景的数字就没有分辨力了。
SCENARIOS: dict[str, tuple[str, int]] = {
    "整天延迟(应D21)":      ("整天覆盖的延迟公告 —— 单边应挂起等补发", 10),
    "整天费率(应D22)":      ("整天覆盖的费率误用公告 —— 手续费差异应等渠道更正", 10),
    "部分时段·窗内(应D21)": ("公告只覆盖某时间窗，本笔在窗内 —— 应挂起", 8),
    "部分时段·窗外(应D01)": ("同一天同一渠道，本笔在窗外 —— 应照常发起查询", 8),
    "更正收窄(应D01)":      ("当日有延迟公告，但更正公告把本渠道排除 —— 应查询", 5),
    "近似延迟(应D01)":      ("主题相关但正文明确不覆盖 —— 应查询", 5),
    "近似费率(应D05)":      ("主题相关但正文明确不覆盖 —— 应冲正", 8),
}

_WIN_BY_LABEL = {lab: (lo, hi) for lo, hi, lab in WINDOWS}


@dataclass
class ScenarioView:
    conn: object

    def __post_init__(self):
        self._titles: dict[tuple[str, str], set[str]] = {}
        self._txn: dict[str, str | None] = {
            r["id"]: r["channel_txn_no"]
            for r in db.q(self.conn, "SELECT id, channel_txn_no FROM recon_diffs")}

    def titles(self, channel_id: str, bill_date: str) -> set[str]:
        key = (channel_id, bill_date)
        if key not in self._titles:
            self._titles[key] = {r["title"] for r in db.q(self.conn, """
                SELECT title FROM channel_notices WHERE channel_id=?
                  AND effective_from <= ? AND COALESCE(effective_to, effective_from) >= ?
            """, (channel_id, bill_date, bill_date))}
        return self._titles[key]

    def _txn_time(self, txn: str | None) -> str | None:
        if not txn:
            return None
        r = db.q1(self.conn, "SELECT occurred_at FROM channel_bill_records "
                             "WHERE channel_txn_no=?", (txn,))
        if r and r["occurred_at"]:
            return r["occurred_at"]
        r = db.q1(self.conn, "SELECT paid_at FROM payments WHERE channel_txn_no=?", (txn,))
        return r["paid_at"] if r else None

    def in_window(self, channel_id: str, bill_date: str, txn: str | None) -> bool | None:
        body = db.q1(self.conn, """
            SELECT body FROM channel_notices WHERE channel_id=? AND effective_from=?
              AND title IN (%s)""" % ",".join("?" * len(SCOPED_TITLES)),
            [channel_id, bill_date, *sorted(SCOPED_TITLES)])
        ts = self._txn_time(txn)
        if not (body and ts):
            return None
        for lab, (lo, hi) in _WIN_BY_LABEL.items():
            if lab in body["body"]:
                return lo <= minute_of_day(ts) < hi
        return None

    def classify(self, task) -> str | None:
        """这条差错属于哪个设计场景。不属于任何一个则返回 None。"""
        t = self.titles(task.channel_id, task.bill_date)
        codes = set(task.substantive_codes)
        txn = self._txn.get(task.diff_id)

        if t & SCOPED_TITLES and codes & {"D21", "D01"}:
            w = self.in_window(task.channel_id, task.bill_date, txn)
            if w is True:
                return "部分时段·窗内(应D21)"
            if w is False:
                return "部分时段·窗外(应D01)"
            return None
        if t & RETRACTION_TITLES and "D01" in codes:
            return "更正收窄(应D01)"
        if t & NEAR_MISS_DELAY_TITLES and "D01" in codes:
            return "近似延迟(应D01)"
        if t & NEAR_MISS_FEE_TITLES and "D05" in codes:
            return "近似费率(应D05)"
        if t & DELAY_TITLES and "D21" in codes:
            return "整天延迟(应D21)"
        if t & FEE_TITLES and "D22" in codes:
            return "整天费率(应D22)"
        return None


def bucket(conn, tasks) -> dict[str, list]:
    view = ScenarioView(conn)
    out: dict[str, list] = {k: [] for k in SCENARIOS}
    for t in tasks:
        k = view.classify(t)
        if k:
            out[k].append(t)
    return out


def coverage_report(conn, tasks) -> list[tuple[str, int, int, bool]]:
    """[(场景, 实际条数, 最少要求, 是否达标)]"""
    b = bucket(conn, tasks)
    return [(k, len(b[k]), SCENARIOS[k][1], len(b[k]) >= SCENARIOS[k][1])
            for k in SCENARIOS]


def score_by_scenario(conn, tasks, *solvers: tuple[str, dict]) -> list[dict]:
    """按场景给多个求解方出分。solvers 为 [(名字, {task_id: Solution}), ...]"""
    b = bucket(conn, tasks)
    rows = []
    for name in SCENARIOS:
        ts = b[name]
        row = {"scenario": name, "n": len(ts)}
        for solver_name, sols in solvers:
            if not ts:
                row[solver_name] = None
                continue
            ok = sum(1 for t in ts
                     if {c for c in sols[t.task_id].root_causes if c != "UNKNOWN"}
                     == set(t.substantive_codes))
            row[solver_name] = ok / len(ts)
        rows.append(row)
    return rows
