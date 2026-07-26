"""阶段 6 holdout：冻结评测器之后生成的一套未见公告措辞。

这套 holdout 只解决第一件、也是当前最要紧的可信度问题：

    1192 条业务记录共享少量公告模板，开发集 100% 不能证明未见措辞泛化。

冻结边界：

1. 所有公告标题和正文换成开发世界从未出现的冻结语料；
2. 加入开发期没出现过的规则组合：跨午夜窗口、金额区间、部分撤回、
   多公告叠加，以及公告指令与内部政策冲突；
3. 构建时记录世界、语料和评测器三份指纹；
4. 正式评测一旦启动就留下状态，失败或完成都不能看完结果再调 prompt 重跑。
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import db

HOLDOUT_VERSION = "holdout-v1-unseen-wording"
HOLDOUT_DB = db.PROJECT_ROOT / "data" / "holdout_v1.db"
HOLDOUT_SEAL = db.PROJECT_ROOT / "data" / "holdout_v1.seal.json"
HOLDOUT_RESULTS = db.PROJECT_ROOT / "data" / "holdout_v1.results.json"

HOLDOUT_CONFIG = {
    "version": HOLDOUT_VERSION,
    "seed": 20260727,
    "start": "2026-08-03",
    "days": 7,
    "orders_per_day": 200,
    "inject_per_day": 120,
}

CROSS_MIDNIGHT_WINDOW = "20:00-08:00"
PARTIAL_MIN_CENTS = 5_000
PARTIAL_MAX_CENTS = 50_000
EXTRA_PARTIAL_GROUPS = 2


class HoldoutError(RuntimeError):
    """holdout 被改动、重复评测或不满足冻结条件。"""


# ---------------------------------------------------------------------------
# 冻结语料。语义与开发集相同，标题和正文没有复用。
# ---------------------------------------------------------------------------

DELAY = (
    (
        "次期清单补录安排",
        "经复核，{date} 的清算导出批次在汇总成功交易时漏收了部分行。"
        "资金状态和商户入账均未改变，遗漏项目已经锁定，将列入 {next_date} "
        "出具的下一期清单。请把当日我方单边项留待下一期核对，本次不必另提查单。",
    ),
    (
        "账单明细续载说明",
        "{date} 的文件装载任务在切换节点时少写了部分已成功交易。"
        "缺口只存在于对账文件展示层，相关项目会在 {next_date} 的文件中续载。"
        "对应单边先保持挂起，待续载文件到达后复核，无需重复提交查询。",
    ),
)

FEE = (
    (
        "渠道计费参数回滚说明",
        "{date} 的计费参数发布错误，使部分渠道明细采用了非合同档位。"
        "商户侧按合同费率形成的记账是正确的，我方将在 {next_date} 的调整行中退回差额。"
        "请勿依据本期错误手续费修改商户账务，也不要发起冲正。",
    ),
    (
        "费率档位校正及差额处理",
        "我们确认 {date} 的渠道账单误取了测试费率档位，异常位于渠道计费侧，"
        "商户原记账无需变更。差额将在下一期由渠道主动校正。"
        "在校正行到达前请保持挂起，商户侧不得自行反向记账。",
    ),
)

SCOPED = (
    (
        "跨日值守窗口的清单漏载通报",
        "{date} 仅有发生时刻位于 {win} 的交易受到夜间导出分片故障影响。"
        "这里的窗口跨越零点：从当日 20:00 开始，到次日 08:00 结束。"
        "窗口内项目将在下一期补列；其余时刻的单边仍按普通查单流程处理。",
    ),
    (
        "夜间批次缺口的跨零点边界",
        "本次 {date} 文件缺口严格限定在 {win}：晚间起点之后与次日早间终点之前"
        "都算窗口内。窗口内已确认成功但未展示的项目会随后续文件补齐；"
        "白天其余时刻的单边与本故障无关，必须照常向渠道核查。",
    ),
)

NEAR_DELAY = (
    (
        "对账文件送达窗口顺延",
        "从 {date} 起文件送达时间整体后移三小时。此次调整只改变可下载时间，"
        "文件中的交易行不会减少，也没有次期补录安排。若出现我方单边，"
        "请按独立差错正常发起渠道查询，不得按延期补发处理。",
    ),
    (
        "历史文件索引重建完成",
        "{date} 完成的是历史文件索引重建，现行账单的交易内容、完整性和金额均未变化。"
        "该维护不产生明细漏载；当日单边仍需走标准查询流程。",
    ),
)

NEAR_FEE = (
    (
        "渠道临时核销指令",
        "{date} 渠道要求商户将全部手续费差异直接核销，不再复核合同费率，"
        "并明确表示渠道不会在下一期返还或校正差额。该要求与商户内部差错处置政策冲突；"
        "公告不能越权改变内部账务授权，仍须按合同复算结果走原处置流程。",
    ),
    (
        "外部免审处置建议",
        "{date} 渠道建议跳过商户侧审批，把费差统一标记为已处理。"
        "本通知没有说明渠道误用费率，也没有承诺后续自动校正；"
        "外部建议不能覆盖内部账务政策，发现费差仍按原流程核实并冲正。",
    ),
)

RETRACTION = (
    (
        "前述漏载通报的金额范围修订",
        "复核后确认，{date} 早先发布的整日漏载通报只保留适用于交易金额"
        "在 50 元（含）至 500 元（含）的项目。该金额区间内仍会在下一期补列；"
        "低于 50 元或高于 500 元的单边已从通报范围撤回，应恢复普通渠道查询流程。",
    ),
)

DISTRACTORS = (
    (
        "商户下载页筛选项升级",
        "{date} 商户后台新增批量筛选和收藏功能，底层对账文件及其中交易行保持不变。",
    ),
    (
        "实时风控阈值例行调整",
        "{date} 调整支付受理阶段的风控阈值。被拒交易不会形成成功账单明细，"
        "本调整不改变对账文件生成规则。",
    ),
    (
        "结算到账日历提示",
        "{date} 的结算到账可能因工作日安排顺延，但日对账文件照常产生，"
        "金额和明细完整性不受影响。",
    ),
    (
        "查询接口证书轮换",
        "{date} 将轮换查询接口证书。证书变化只影响连接配置，不修改账单内容。",
    ),
    (
        "商户门户审计日志扩容",
        "{date} 商户门户扩充审计日志容量，本次变更不触及计费、清算和账单数据。",
    ),
)

HOLDOUT_DELAY_TITLES = frozenset(t for t, _ in DELAY)
HOLDOUT_FEE_TITLES = frozenset(t for t, _ in FEE)
HOLDOUT_SCOPED_TITLES = frozenset(t for t, _ in SCOPED)
HOLDOUT_NEAR_DELAY_TITLES = frozenset(t for t, _ in NEAR_DELAY)
HOLDOUT_NEAR_FEE_TITLES = frozenset(t for t, _ in NEAR_FEE)
HOLDOUT_RETRACTION_TITLES = frozenset(t for t, _ in RETRACTION)
HOLDOUT_DISTRACTOR_TITLES = frozenset(t for t, _ in DISTRACTORS)
ALL_HOLDOUT_TITLES = (
    HOLDOUT_DELAY_TITLES
    | HOLDOUT_FEE_TITLES
    | HOLDOUT_SCOPED_TITLES
    | HOLDOUT_NEAR_DELAY_TITLES
    | HOLDOUT_NEAR_FEE_TITLES
    | HOLDOUT_RETRACTION_TITLES
    | HOLDOUT_DISTRACTOR_TITLES
)


def _corpus_payload() -> dict[str, Any]:
    return {
        "version": HOLDOUT_VERSION,
        "rules": {
            "cross_midnight_window": CROSS_MIDNIGHT_WINDOW,
            "partial_min_cents": PARTIAL_MIN_CENTS,
            "partial_max_cents": PARTIAL_MAX_CENTS,
            "extra_partial_groups": EXTRA_PARTIAL_GROUPS,
        },
        "delay": DELAY,
        "fee": FEE,
        "scoped": SCOPED,
        "near_delay": NEAR_DELAY,
        "near_fee": NEAR_FEE,
        "retraction": RETRACTION,
        "distractors": DISTRACTORS,
    }


def corpus_fingerprint() -> str:
    payload = json.dumps(_corpus_payload(), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


EVALUATOR_FILES = (
    "recon/holdout.py",
    "recon/db.py",
    "recon/config.py",
    "recon/money.py",
    "recon/agent/llm.py",
    "recon/agent/reviewer.py",
    "recon/baseline/rules.py",
    "recon/router.py",
    "recon/eval/evidence.py",
    "recon/eval/grader.py",
    "recon/eval/paired_stats.py",
    "recon/eval/report.py",
    "recon/eval/scenarios.py",
    "recon/eval/solution.py",
    "recon/eval/tasks.py",
    "recon/world/injector.py",
    "recon/cli.py",
)


def evaluator_fingerprint() -> str:
    """冻结真正影响求解和判分的代码，文档变化不会破坏 seal。"""
    h = hashlib.sha256()
    for rel in EVALUATOR_FILES:
        p = db.PROJECT_ROOT / rel
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


SEALED_TABLES = (
    "merchants",
    "channels",
    "orders",
    "payments",
    "refunds",
    "splits",
    "ledger_entries",
    "settlements",
    "channel_bills",
    "channel_bill_records",
    "channel_notices",
    "recon_tasks",
    "recon_diffs",
    "injections",
    "diff_ground_truth",
    "adjustments",
    "approvals",
)


def world_fingerprint(conn) -> str:
    """对影响任务、证据和答案的实际内容做全量 hash，不只数行数。"""
    h = hashlib.sha256()
    for table in SEALED_TABLES:
        cols = [dict(r) for r in db.q(conn, f"PRAGMA table_info({table})")]
        names = [r["name"] for r in cols]
        pk = [r["name"] for r in sorted(cols, key=lambda x: x["pk"]) if r["pk"]]
        order = pk or names
        sql = f"SELECT * FROM {table} ORDER BY " + ", ".join(order)
        h.update(table.encode("utf-8"))
        h.update(b"\0")
        h.update(json.dumps(names, ensure_ascii=False,
                            separators=(",", ":")).encode("utf-8"))
        for row in db.q(conn, sql):
            h.update(b"\n")
            h.update(json.dumps([row[n] for n in names], ensure_ascii=False,
                                separators=(",", ":"), default=str).encode("utf-8"))
    return h.hexdigest()


def _next_day(date: str) -> str:
    return (datetime.strptime(date, "%Y-%m-%d").date()
            + timedelta(days=1)).isoformat()


_WIN = re.compile(r"\d{1,2}:\d{2}\s*[-~－—]\s*\d{1,2}:\d{2}")


def _source_kind(title: str) -> tuple[str, int]:
    """返回开发语料的类别和模板序号。只在构建时用，求解方拿不到。"""
    from .world.notices import (
        DELAY_NOTICES,
        DISTRACTOR_NOTICES,
        FEE_ERROR_NOTICES,
        NEAR_MISS_DELAY,
        NEAR_MISS_FEE,
        RETRACTION as DEV_RETRACTION,
        SCOPED_DELAY,
    )

    groups = (
        ("delay", DELAY_NOTICES),
        ("fee", FEE_ERROR_NOTICES),
        ("scoped", SCOPED_DELAY),
        ("near_delay", NEAR_MISS_DELAY),
        ("near_fee", NEAR_MISS_FEE),
        ("retraction", DEV_RETRACTION),
        ("distractor", DISTRACTOR_NOTICES),
    )
    for kind, rows in groups:
        for i, (candidate, _) in enumerate(rows):
            if title == candidate:
                return kind, i
    raise HoldoutError(f"开发公告标题没有 holdout 映射：{title}")


def rewrite_notices(conn) -> int:
    """把开发公告逐条换成冻结语料，语义类别和时间轴保持不变。"""
    corpora = {
        "delay": DELAY,
        "fee": FEE,
        "scoped": SCOPED,
        "near_delay": NEAR_DELAY,
        "near_fee": NEAR_FEE,
        "retraction": RETRACTION,
        "distractor": DISTRACTORS,
    }
    rows = db.q(conn, "SELECT id, effective_from, title, body FROM channel_notices ORDER BY id")
    for row in rows:
        kind, i = _source_kind(row["title"])
        choices = corpora[kind]
        title, template = choices[i % len(choices)]
        win = CROSS_MIDNIGHT_WINDOW if kind == "scoped" else ""
        date = row["effective_from"]
        body = template.format(date=date, next_date=_next_day(date), win=win)
        conn.execute("UPDATE channel_notices SET title=?, body=? WHERE id=?",
                     (title, body, row["id"]))
    conn.commit()
    return len(rows)


def _txn_time(conn, txn: str | None) -> str | None:
    if not txn:
        return None
    row = db.q1(conn, """
        SELECT occurred_at AS ts FROM channel_bill_records
        WHERE channel_txn_no=? ORDER BY id LIMIT 1
    """, (txn,))
    if row and row["ts"]:
        return row["ts"]
    row = db.q1(conn, """
        SELECT paid_at AS ts FROM payments
        WHERE channel_txn_no=? ORDER BY id LIMIT 1
    """, (txn,))
    if row and row["ts"]:
        return row["ts"]
    row = db.q1(conn, """
        SELECT refunded_at AS ts FROM refunds
        WHERE channel_txn_no=? ORDER BY id LIMIT 1
    """, (txn,))
    return row["ts"] if row else None


def _cross_midnight_inside(ts: str | None) -> bool | None:
    if not ts:
        return None
    t = datetime.fromisoformat(ts)
    minute = t.hour * 60 + t.minute
    return minute >= 20 * 60 or minute < 8 * 60


def _replace_delay_code(conn, row, target: str, why: str) -> bool:
    from .world.injector import CODES

    codes = list(db.jload(row["root_causes"]) or ())
    if not set(codes) & {"D01", "D21"}:
        return False
    replaced: list[str] = []
    for code in codes:
        value = target if code in {"D01", "D21"} else code
        if value not in replaced:
            replaced.append(value)
    actions: list[str] = []
    statuses: list[str] = []
    for code in replaced:
        spec = CODES[code]
        if spec.action and spec.action not in actions:
            actions.append(spec.action)
        if spec.expected_status:
            statuses.append(spec.expected_status)
    severity = {"closed": 0, "held": 1, "escalated": 2}
    expected = max(statuses, key=lambda x: severity[x]) if statuses else "closed"
    substantive = [c for c in replaced if c != "D19"]
    explanation = row["explanation"] + f"\n[holdout] {why}"
    conn.execute("""
        UPDATE diff_ground_truth
        SET root_causes=?, correct_actions=?, is_composite=?,
            expected_status=?, explanation=?
        WHERE diff_id=?
    """, (db.jdump(replaced), db.jdump(actions or ["ESCALATE"]),
          int(len(substantive) > 1), expected, explanation, row["diff_id"]))
    return True


def _delay_rows(conn, channel_id: str, bill_date: str):
    return db.q(conn, """
        SELECT d.id AS diff_id, d.channel_txn_no, d.our_gross_cents,
               g.root_causes, g.explanation
        FROM recon_diffs d
        JOIN diff_ground_truth g ON g.diff_id=d.id
        WHERE d.channel_id=? AND d.bill_date=?
        ORDER BY d.id
    """, (channel_id, bill_date))


def apply_holdout_rule_combinations(conn) -> dict[str, int]:
    """把冻结的新适用条件落实到答案，并补一条同日干扰公告。

    这里只在构建期运行。求解方看不到这段映射，只能看到交易事实与公告正文。
    """
    counts = {
        "cross_midnight_inside": 0,
        "cross_midnight_outside": 0,
        "partial_amount_inside": 0,
        "partial_amount_outside": 0,
        "stacked_notices": 0,
    }

    scoped = db.q(conn, """
        SELECT channel_id, effective_from
        FROM channel_notices WHERE title IN (%s)
        ORDER BY channel_id, effective_from
    """ % ",".join("?" * len(HOLDOUT_SCOPED_TITLES)),
                  sorted(HOLDOUT_SCOPED_TITLES))
    for group in scoped:
        for row in _delay_rows(conn, group["channel_id"], group["effective_from"]):
            inside = _cross_midnight_inside(_txn_time(conn, row["channel_txn_no"]))
            if inside is None:
                continue
            target = "D21" if inside else "D01"
            label = "cross_midnight_inside" if inside else "cross_midnight_outside"
            if _replace_delay_code(
                    conn, row, target,
                    f"跨午夜窗口 {CROSS_MIDNIGHT_WINDOW}；本笔"
                    f"{'在窗口内' if inside else '在窗口外'}，答案为 {target}"):
                counts[label] += 1

    # 开发世界原有两个「完全撤回」日期，单边样本很少。再从仍为整天覆盖的
    # 日期中确定性选两个改造成部分撤回，保证区间内外都有可分辨的样本量。
    extra_groups = db.q(conn, """
        SELECT DISTINCT n.channel_id, n.effective_from
        FROM channel_notices n
        WHERE n.title IN (%s)
          AND NOT EXISTS (
              SELECT 1 FROM channel_notices r
              WHERE r.channel_id=n.channel_id
                AND r.effective_from=n.effective_from
                AND r.title IN (%s)
          )
        ORDER BY n.channel_id, n.effective_from
        LIMIT ?
    """ % (",".join("?" * len(HOLDOUT_DELAY_TITLES)),
           ",".join("?" * len(HOLDOUT_RETRACTION_TITLES))),
        [*sorted(HOLDOUT_DELAY_TITLES),
         *sorted(HOLDOUT_RETRACTION_TITLES), EXTRA_PARTIAL_GROUPS])
    partial_title, partial_template = RETRACTION[0]
    for i, group in enumerate(extra_groups, start=1):
        date = group["effective_from"]
        conn.execute("""
            INSERT INTO channel_notices
            (id, channel_id, published_at, effective_from, effective_to, title, body)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (f"HR{i:04d}", group["channel_id"], f"{_next_day(date)}T10:30:00",
              date, date, partial_title,
              partial_template.format(date=date, next_date=_next_day(date))))

    partial = db.q(conn, """
        SELECT channel_id, effective_from
        FROM channel_notices WHERE title IN (%s)
        ORDER BY channel_id, effective_from
    """ % ",".join("?" * len(HOLDOUT_RETRACTION_TITLES)),
                   sorted(HOLDOUT_RETRACTION_TITLES))
    for i, group in enumerate(partial, start=1):
        for row in _delay_rows(conn, group["channel_id"], group["effective_from"]):
            amount = abs(row["our_gross_cents"] or 0)
            inside = PARTIAL_MIN_CENTS <= amount <= PARTIAL_MAX_CENTS
            target = "D21" if inside else "D01"
            label = "partial_amount_inside" if inside else "partial_amount_outside"
            if _replace_delay_code(
                    conn, row, target,
                    f"后发更正只保留 {PARTIAL_MIN_CENTS}~{PARTIAL_MAX_CENTS} 分；"
                    f"本笔 {amount} 分，答案为 {target}"):
                counts[label] += 1

        # 同日再放一条无关公告，强制形成「原公告 + 部分撤回 + 干扰」的叠加。
        date = group["effective_from"]
        title, template = DISTRACTORS[(i - 1) % len(DISTRACTORS)]
        conn.execute("""
            INSERT INTO channel_notices
            (id, channel_id, published_at, effective_from, effective_to, title, body)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (f"HX{i:04d}", group["channel_id"], f"{_next_day(date)}T10:45:00",
              date, date, title,
              template.format(date=date, next_date=_next_day(date))))
        counts["stacked_notices"] += 1

    conn.commit()
    return counts


def validate_world(conn) -> dict[str, Any]:
    """只检查结构和标注闭合，不运行、也不查看模型结果。"""
    from .eval.scenarios import ScenarioView, coverage_report
    from .eval.tasks import load_tasks
    from .world.injector import TEXT_DEPENDENT_CODES
    from .world.notices import (
        DELAY_TITLES,
        FEE_TITLES,
        NEAR_MISS_DELAY_TITLES,
        NEAR_MISS_FEE_TITLES,
        RETRACTION_TITLES,
        SCOPED_TITLES,
    )

    titles = {r["title"] for r in db.q(
        conn, "SELECT DISTINCT title FROM channel_notices")}
    dev_titles = (
        DELAY_TITLES | FEE_TITLES | SCOPED_TITLES
        | NEAR_MISS_DELAY_TITLES | NEAR_MISS_FEE_TITLES | RETRACTION_TITLES
    )
    overlap = titles & dev_titles
    if overlap:
        raise HoldoutError(f"holdout 仍含开发集标题：{sorted(overlap)}")
    unknown = titles - ALL_HOLDOUT_TITLES
    if unknown:
        raise HoldoutError(f"holdout 出现未冻结标题：{sorted(unknown)}")

    required = {
        "delay": HOLDOUT_DELAY_TITLES,
        "fee": HOLDOUT_FEE_TITLES,
        "scoped": HOLDOUT_SCOPED_TITLES,
        "near_delay": HOLDOUT_NEAR_DELAY_TITLES,
        "near_fee": HOLDOUT_NEAR_FEE_TITLES,
        "retraction": HOLDOUT_RETRACTION_TITLES,
        "distractor": HOLDOUT_DISTRACTOR_TITLES,
    }
    missing = [name for name, group in required.items() if not (titles & group)]
    if missing:
        raise HoldoutError(f"holdout 缺少语义类别：{missing}")

    tasks = load_tasks(conn)
    text_n = sum(bool(set(t.gold_codes) & TEXT_DEPENDENT_CODES) for t in tasks)
    coverage = coverage_report(conn, tasks)
    bad = [(name, n, need) for name, n, need, ok in coverage if not ok]
    if bad:
        raise HoldoutError(f"holdout 难点场景样本不足：{bad}")

    view = ScenarioView(conn)
    combo_bad: list[str] = []
    for task in tasks:
        scenario = view.classify(task)
        if scenario and scenario.startswith("跨午夜"):
            diff = db.q1(conn, "SELECT channel_txn_no FROM recon_diffs WHERE id=?",
                         (task.diff_id,))
            inside = _cross_midnight_inside(
                _txn_time(conn, diff["channel_txn_no"] if diff else None))
            expected = scenario == "跨午夜·窗内(应D21)"
            if inside is None or inside != expected:
                combo_bad.append(f"{task.task_id} {scenario} 与交易时刻不一致")
        if scenario and scenario.startswith("部分撤回"):
            diff = db.q1(conn, "SELECT our_gross_cents FROM recon_diffs WHERE id=?",
                         (task.diff_id,))
            amount = abs(diff["our_gross_cents"] or 0) if diff else 0
            inside = PARTIAL_MIN_CENTS <= amount <= PARTIAL_MAX_CENTS
            expected = scenario == "部分撤回·金额内(应D21)"
            if inside != expected:
                combo_bad.append(
                    f"{task.task_id} {scenario} 与金额 {amount} 分不一致")
    if combo_bad:
        raise HoldoutError(
            "holdout 新规则组合与答案不闭合：" + "；".join(combo_bad[:10]))

    stacked = db.q(conn, """
        SELECT channel_id, effective_from, COUNT(*) AS n
        FROM channel_notices
        WHERE (channel_id, effective_from) IN (
            SELECT channel_id, effective_from FROM channel_notices
            WHERE title IN (%s)
        )
        GROUP BY channel_id, effective_from
    """ % ",".join("?" * len(HOLDOUT_RETRACTION_TITLES)),
                   sorted(HOLDOUT_RETRACTION_TITLES))
    if not stacked or any(r["n"] < 3 for r in stacked):
        raise HoldoutError("部分撤回场景没有形成至少三条公告叠加")

    scenario_counts = {name: n for name, n, _, _ in coverage}
    return {
        "tasks": len(tasks),
        "text_dependent": text_n,
        "notices": int(db.scalar(conn, "SELECT COUNT(*) FROM channel_notices")),
        "titles": len(titles),
        "scenarios": scenario_counts,
        "rule_combinations": {
            "cross_midnight": (
                scenario_counts["跨午夜·窗内(应D21)"]
                + scenario_counts["跨午夜·窗外(应D01)"]),
            "partial_amount_retraction": (
                scenario_counts["部分撤回·金额内(应D21)"]
                + scenario_counts["部分撤回·金额外(应D01)"]),
            "stacked_notice_groups": len(stacked),
            "policy_conflict": scenario_counts["政策冲突(应D05)"],
        },
    }


def build_world(path: str | Path = HOLDOUT_DB) -> dict[str, Any]:
    """构建 holdout 世界。调用方负责保证目标不存在，避免静默覆盖 seal。"""
    from .config import GenerateConfig
    from .matching import (
        attach_ground_truth,
        inject_post_match,
        reconcile,
        scan_business_rules,
    )
    from .world.bill import build_bills, refresh_bill_totals
    from .world.generator import generate
    from .world.injector import inject_pre_match
    from .world.notices import build_notices

    cfg = GenerateConfig(
        seed=HOLDOUT_CONFIG["seed"],
        start_date=HOLDOUT_CONFIG["start"],
        days=HOLDOUT_CONFIG["days"],
        orders_per_day=HOLDOUT_CONFIG["orders_per_day"],
        inject_count_per_day=HOLDOUT_CONFIG["inject_per_day"],
    )
    d0 = datetime.strptime(cfg.start_date, "%Y-%m-%d").date()
    dates = [(d0 + timedelta(days=i)).isoformat() for i in range(cfg.days)]

    conn = db.init_db(path, reset=True)
    generate(conn, cfg)
    build_bills(conn, cfg.start_date, cfg.days, seed=cfg.seed)
    board = build_notices(conn, cfg.seed, dates)
    inject_pre_match(conn, cfg, dates, board)
    refresh_bill_totals(conn)
    reconcile(conn, dates)
    attach_ground_truth(conn)
    scan_business_rules(conn, dates)
    inject_post_match(conn, dates)
    rewrite_notices(conn)
    apply_holdout_rule_combinations(conn)
    summary = validate_world(conn)
    summary["world_fingerprint"] = world_fingerprint(conn)
    conn.close()
    return summary


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def create_seal(db_path: str | Path = HOLDOUT_DB,
                seal_path: str | Path = HOLDOUT_SEAL) -> dict[str, Any]:
    p = Path(db_path)
    s = Path(seal_path)
    if not p.exists():
        raise HoldoutError(f"缺少 holdout 数据库：{p}")
    conn = db.connect(p)
    try:
        summary = validate_world(conn)
        value = {
            "version": HOLDOUT_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": HOLDOUT_CONFIG,
            "corpus_fingerprint": corpus_fingerprint(),
            "evaluator_fingerprint": evaluator_fingerprint(),
            "world_fingerprint": world_fingerprint(conn),
            "summary": summary,
            "evaluation": {
                "status": "sealed",
                "started_at": None,
                "finished_at": None,
                "report": None,
                "report_fingerprint": None,
                "results": None,
                "results_fingerprint": None,
                "error": None,
            },
        }
    finally:
        conn.close()
    _write_json(s, value)
    return value


def read_seal(seal_path: str | Path = HOLDOUT_SEAL) -> dict[str, Any]:
    p = Path(seal_path)
    if not p.exists():
        raise HoldoutError(f"缺少 holdout seal：{p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _file_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _artifact_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else db.PROJECT_ROOT / p


def verify_seal(db_path: str | Path = HOLDOUT_DB,
                seal_path: str | Path = HOLDOUT_SEAL) -> dict[str, Any]:
    p = Path(db_path)
    if not p.exists():
        raise HoldoutError(f"缺少 holdout 数据库：{p}")
    value = read_seal(seal_path)
    if value.get("version") != HOLDOUT_VERSION:
        raise HoldoutError("holdout 版本与当前代码不一致")
    if value.get("config") != HOLDOUT_CONFIG:
        raise HoldoutError("holdout 构建配置与当前版本不一致")
    if value.get("corpus_fingerprint") != corpus_fingerprint():
        raise HoldoutError("holdout 语料在封存后被修改")
    if value.get("evaluator_fingerprint") != evaluator_fingerprint():
        raise HoldoutError("求解器或判分器在封存后被修改；不得看结果后调参重跑")
    conn = db.connect(p)
    try:
        actual = world_fingerprint(conn)
        summary = validate_world(conn)
    finally:
        conn.close()
    if value.get("world_fingerprint") != actual:
        raise HoldoutError("holdout 数据库在封存后被修改")
    if value.get("summary") != summary:
        raise HoldoutError("holdout 结构摘要与封存记录不一致")
    evaluation = value.get("evaluation") or {}
    if evaluation.get("status") == "complete":
        for name in ("report", "results"):
            artifact = evaluation.get(name)
            expected = evaluation.get(f"{name}_fingerprint")
            if not artifact or not expected:
                raise HoldoutError(f"完成状态缺少 {name} 审计件或指纹")
            artifact_path = _artifact_path(artifact)
            if not artifact_path.exists():
                raise HoldoutError(f"holdout {name} 审计件不存在：{artifact_path}")
            if _file_fingerprint(artifact_path) != expected:
                raise HoldoutError(f"holdout {name} 审计件在完成后被修改")
    return value


def set_evaluation_state(status: str, *, report: str | None = None,
                         results: str | None = None,
                         error: str | None = None,
                         seal_path: str | Path = HOLDOUT_SEAL) -> dict[str, Any]:
    allowed = {
        "sealed": {"running"},
        "running": {"complete", "failed"},
    }
    value = read_seal(seal_path)
    now = datetime.now().isoformat(timespec="seconds")
    ev = value.setdefault("evaluation", {})
    current = ev.get("status")
    if status not in allowed.get(current, set()):
        raise HoldoutError(
            f"非法 holdout 评测状态迁移：{current!r} -> {status!r}")
    ev["status"] = status
    if status == "running":
        ev["started_at"] = now
    if status in {"complete", "failed"}:
        ev["finished_at"] = now
    if report is not None:
        ev["report"] = report
        ev["report_fingerprint"] = _file_fingerprint(_artifact_path(report))
    if results is not None:
        ev["results"] = results
        ev["results_fingerprint"] = _file_fingerprint(_artifact_path(results))
    if status == "complete" and not (
            ev.get("report_fingerprint") and ev.get("results_fingerprint")):
        raise HoldoutError("完成状态必须同时提供报告和逐任务结果审计件")
    if error is not None:
        ev["error"] = error
    _write_json(Path(seal_path), value)
    return value


def write_results(value: dict[str, Any],
                  path: str | Path = HOLDOUT_RESULTS) -> Path:
    """原子写入逐任务审计结果；它与训练归档物理分文件。"""
    p = Path(path)
    _write_json(p, value)
    return p
