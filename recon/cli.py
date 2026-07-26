"""命令行入口。

一条命令跑完整条流水线：
    python -m recon.cli build --seed 42 --days 3 --orders-per-day 200
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from . import db, invariants
from .config import GenerateConfig, required_role, tolerance_for
from .matching import (attach_ground_truth, diff_shape, diffs_without_gt,
                       inject_post_match, orphan_injections, reconcile,
                       scan_business_rules)
from .money import fmt
from .world.bill import build_bills, refresh_bill_totals
from .world.notices import build_notices
from .world.generator import generate
from .world.injector import CODES, inject_pre_match

console = Console()
DOCS = db.PROJECT_ROOT / "docs"


def _dates(start: str, days: int) -> list[str]:
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    return [(d0 + timedelta(days=i)).isoformat() for i in range(days)]


def _cfg(seed: int, start: str, days: int, orders: int, inject: int) -> GenerateConfig:
    return GenerateConfig(seed=seed, start_date=start, days=days,
                          orders_per_day=orders, inject_count_per_day=inject)


@click.group()
def cli() -> None:
    """交易对账差错处置系统。阶段 0 业务世界层 + 阶段 1 评测与规则基线已完成。"""


# --------------------------------------------------------------------------

@cli.command("init-db")
@click.option("--reset", is_flag=True, help="删库重建")
@click.option("--db-path", default=None)
def init_db_cmd(reset: bool, db_path: str | None) -> None:
    conn = db.init_db(db_path, reset=reset)
    console.print(f"[green]建表完成[/] {len(db.all_tables(conn))} 张表 -> "
                  f"{db_path or db.DEFAULT_DB}")
    conn.close()


@cli.command("build")
@click.option("--seed", default=42, show_default=True)
@click.option("--start", default="2026-07-01", show_default=True)
@click.option("--days", default=3, show_default=True)
@click.option("--orders-per-day", default=200, show_default=True)
@click.option("--inject-per-day", default=120, show_default=True)
@click.option("--db-path", default=None)
@click.option("--quiet", is_flag=True)
def build_cmd(seed: int, start: str, days: int, orders_per_day: int,
              inject_per_day: int, db_path: str | None, quiet: bool) -> None:
    """一条命令：建库 -> 生成世界 -> 生成账单 -> 注入差错 -> 对账 -> 贴答案 -> 校验。"""
    cfg = _cfg(seed, start, days, orders_per_day, inject_per_day)
    dates = _dates(start, days)
    conn = db.init_db(db_path, reset=True)

    gen = generate(conn, cfg)
    bills = build_bills(conn, start, days, seed=seed)
    board = build_notices(conn, seed, dates)          # 自由文本证据层
    inj = inject_pre_match(conn, cfg, dates, board)
    refresh_bill_totals(conn)
    rec = reconcile(conn, dates)
    gt = attach_ground_truth(conn)
    scan = scan_business_rules(conn, dates)
    post = inject_post_match(conn, dates)

    total_diffs = int(db.scalar(conn, "SELECT COUNT(*) FROM recon_diffs"))
    total_gt = int(db.scalar(conn, "SELECT COUNT(*) FROM diff_ground_truth"))
    orphans = orphan_injections(conn)
    no_gt = diffs_without_gt(conn)

    if not quiet:
        t = Table(title=f"构建完成 seed={seed} {start} +{days}d", show_header=False)
        t.add_column(style="cyan")
        t.add_column(justify="right")
        for k, v in gen.as_dict().items():
            t.add_row(k, str(v))
        t.add_row("channel_bills", str(bills["bills"]))
        t.add_row("channel_bill_records", str(bills["records"]))
        t.add_row("channel_notices（自由文本）", str(len(board.rows)))
        t.add_row("injections", str(inj["injected"]))
        t.add_row("recon_tasks", str(rec["tasks"]))
        t.add_row("matched", str(rec["matched"]))
        t.add_row("diffs / 流水匹配", str(rec["diffs"]))
        t.add_row("diffs / 规则扫描", str(scan["rule_scan_diffs"]))
        t.add_row("diffs / 结算扫描", str(post["post_match_diffs"]))
        t.add_row("[bold]diffs 合计[/]", f"[bold]{total_diffs}[/]")
        t.add_row("ground_truth 行", str(total_gt))
        style = "red" if orphans else "green"
        t.add_row("孤儿注入（未产出差错）", f"[{style}]{len(orphans)}[/]")
        style = "red" if no_gt else "green"
        t.add_row("无答案差错", f"[{style}]{no_gt}[/]")
        console.print(t)
        for r in orphans[:5]:
            console.print(f"  [red]孤儿注入[/] {r['id']} {r['code']} key={r['match_key']}")

    _print_checks(conn)
    conn.close()


@cli.command("reconcile")
@click.option("--start", default="2026-07-01")
@click.option("--days", default=3)
@click.option("--db-path", default=None)
def reconcile_cmd(start: str, days: int, db_path: str | None) -> None:
    conn = db.connect(db_path)
    rec = reconcile(conn, _dates(start, days))
    gt = attach_ground_truth(conn)
    console.print(rec, gt)
    conn.close()


# --------------------------------------------------------------------------

def _print_checks(conn) -> None:
    results = invariants.run_all(conn)
    t = Table(title="不变量校验")
    t.add_column("码"); t.add_column("不变量"); t.add_column("结果")
    t.add_column("检查项数", justify="right"); t.add_column("违反", justify="right")
    t.add_column("说明", overflow="fold")
    for r in results:
        expected = r.code in invariants.EXPECTED_VIOLATIONS
        if r.passed:
            verdict = "[green]PASS[/]"
        elif expected:
            verdict = "[yellow]违反（预期）[/]"
        else:
            verdict = "[red]FAIL[/]"
        t.add_row(r.code, r.name, verdict, str(r.checked), str(r.n_violations), r.note)
    console.print(t)

    hard = invariants.hard_failures(results)
    if hard:
        console.print("[bold red]存在必须恒成立却失败的不变量：[/]")
        for r in hard:
            for v in r.violations:
                console.print(f"  [red]{r.code}[/] {v}")
    else:
        console.print("[green]恒成立类不变量全部通过[/]")

    for r in results:
        if r.code in invariants.EXPECTED_VIOLATIONS and r.violations:
            console.print(f"[yellow]{r.code} 违反样例（注入生效的证据）：[/]")
            for v in r.violations[:3]:
                console.print(f"  {v}")


@cli.command("check")
@click.option("--db-path", default=None)
def check_cmd(db_path: str | None) -> None:
    conn = db.connect(db_path)
    _print_checks(conn)
    conn.close()


# --------------------------------------------------------------------------

@cli.command("stats")
@click.option("--db-path", default=None)
@click.option("--save", is_flag=True, help="同时写 docs/stage0_report.md")
def stats_cmd(db_path: str | None, save: bool) -> None:
    conn = db.connect(db_path)
    lines: list[str] = []

    total = int(db.scalar(conn, "SELECT COUNT(*) FROM recon_diffs"))
    with_gt = int(db.scalar(conn, "SELECT COUNT(*) FROM diff_ground_truth"))
    composite = int(db.scalar(conn, "SELECT COUNT(*) FROM diff_ground_truth WHERE is_composite=1"))

    head = Table(title="差错池总览", show_header=False)
    head.add_column(style="cyan"); head.add_column(justify="right")
    head.add_row("差错总数", str(total))
    head.add_row("带答案的差错", str(with_gt))
    head.add_row("复合差错（≥2 个原因）", str(composite))
    head.add_row("无答案差错", str(total - with_gt))
    console.print(head)
    lines += ["# 阶段 0 验收报告", "",
              f"- 差错总数：**{total}**",
              f"- 带答案的差错：**{with_gt}**",
              f"- 复合差错：**{composite}**",
              f"- 无答案差错：{total - with_gt}", ""]

    # 按差错码
    counts: dict[str, int] = {}
    for r in db.q(conn, "SELECT root_causes FROM diff_ground_truth"):
        for code in db.jload(r["root_causes"]) or []:
            counts[code] = counts.get(code, 0) + 1

    t = Table(title=f"{len(CODES)} 类差错覆盖")
    t.add_column("码"); t.add_column("名称"); t.add_column("处置")
    t.add_column("命中差错数", justify="right")
    lines += [f"## {len(CODES)} 类差错覆盖", "",
              "| 码 | 名称 | 处置 | 命中差错数 |", "|---|---|---|---:|"]
    missing = []
    for code in sorted(CODES):
        c = CODES[code]
        n = counts.get(code, 0)
        if n == 0:
            missing.append(code)
        style = "" if n else "[red]"
        t.add_row(f"{style}{code}", c.name, c.action or "（同底层）", f"{style}{n}")
        lines.append(f"| {code} | {c.name} | {c.action or '（同底层）'} | {n} |")
    console.print(t)
    lines.append("")
    verdict = (f"{len(CODES)}/{len(CODES)} 全覆盖" if not missing
               else f"缺失 {len(missing)} 类：{missing}")
    console.print(f"覆盖情况：[{'green' if not missing else 'red'}]{verdict}[/]")
    lines += [f"覆盖情况：**{verdict}**", ""]

    # 形态分布
    shapes: dict[str, int] = {}
    for r in db.q(conn, "SELECT * FROM recon_diffs"):
        s = diff_shape(r)
        shapes[s] = shapes.get(s, 0) + 1
    st = Table(title="差错机械形态分布（不是归因）")
    st.add_column("形态"); st.add_column("数量", justify="right")
    lines += ["## 差错机械形态分布", "", "| 形态 | 数量 |", "|---|---:|"]
    for k, v in sorted(shapes.items(), key=lambda x: -x[1]):
        st.add_row(k, str(v))
        lines.append(f"| {k} | {v} |")
    console.print(st)

    # 处置动作分布
    acts: dict[str, int] = {}
    for r in db.q(conn, "SELECT correct_actions FROM diff_ground_truth"):
        for a in db.jload(r["correct_actions"]) or []:
            acts[a] = acts.get(a, 0) + 1
    at = Table(title="正确处置动作分布")
    at.add_column("动作"); at.add_column("数量", justify="right")
    lines += ["", "## 正确处置动作分布", "", "| 动作 | 数量 |", "|---|---:|"]
    for k, v in sorted(acts.items(), key=lambda x: -x[1]):
        at.add_row(k, str(v))
        lines.append(f"| {k} | {v} |")
    console.print(at)

    # 终态分布
    et: dict[str, int] = {}
    for r in db.q(conn, "SELECT expected_status, COUNT(*) n FROM diff_ground_truth GROUP BY 1"):
        et[r["expected_status"]] = r["n"]
    lines += ["", "## 期望终态分布", "", "| 终态 | 数量 |", "|---|---:|"]
    for k, v in sorted(et.items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v} |")
    console.print("期望终态：", et)

    if save:
        DOCS.mkdir(parents=True, exist_ok=True)
        (DOCS / "stage0_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.print(f"[green]已写入[/] {DOCS / 'stage0_report.md'}")
    conn.close()


# --------------------------------------------------------------------------

@cli.command("dump-case")
@click.argument("diff_id", required=False)
@click.option("--composite", is_flag=True, help="随机挑一条复合差错")
@click.option("--code", default=None, help="按差错码挑一条，如 D09")
@click.option("--show-answer", is_flag=True, help="显示答案（判分器视角，agent 不可见）")
@click.option("--db-path", default=None)
def dump_case_cmd(diff_id: str | None, composite: bool, code: str | None,
                  show_answer: bool, db_path: str | None) -> None:
    """打印一条差错的完整证据面 —— 这就是 agent 将要看到的输入。"""
    conn = db.connect(db_path)

    if diff_id is None:
        sql = "SELECT d.id FROM recon_diffs d JOIN diff_ground_truth g ON g.diff_id=d.id WHERE 1=1"
        params: list = []
        if composite:
            sql += " AND g.is_composite=1"
        if code:
            sql += " AND g.root_causes LIKE ?"
            params.append(f'%"{code}"%')
        sql += " ORDER BY d.id LIMIT 1"
        row = db.q1(conn, sql, params)
        if row is None:
            console.print("[red]没找到符合条件的差错[/]")
            return
        diff_id = row["id"]

    d = db.q1(conn, "SELECT * FROM recon_diffs WHERE id=?", (diff_id,))
    if d is None:
        console.print(f"[red]差错 {diff_id} 不存在[/]")
        return

    console.rule(f"[bold]差错 {diff_id}[/]  形态={diff_shape(d)}")
    info = Table(show_header=False)
    info.add_column(style="cyan"); info.add_column(overflow="fold")
    for k in ("channel_id", "bill_date", "our_ref_type", "our_ref_id",
              "channel_record_id", "channel_txn_no", "status"):
        info.add_row(k, str(d[k]))
    for k in ("our_gross_cents", "channel_gross_cents", "diff_cents", "fee_delta_cents"):
        info.add_row(k, f"{d[k]} 分" + (f"（{fmt(d[k])} 元）" if d[k] is not None else ""))
    if d["our_gross_cents"] is not None:
        info.add_row("容差", f"{fmt(tolerance_for(d['our_gross_cents']))} 元")
        info.add_row("所需审批角色", required_role(d["diff_cents"] or 0))
    console.print(info)

    ch = db.q1(conn, "SELECT * FROM channels WHERE id=?", (d["channel_id"],))
    if ch:
        console.print(f"[dim]渠道规则：{ch['name']} | {ch['fee_desc']} | "
                      f"日切 {ch['cutoff_minutes']} 分 | 口径 {ch['bill_basis']} | "
                      f"退款 {ch['refund_mode']} | 币种 {ch['currency']}[/]")

    if d["channel_txn_no"]:
        console.print("\n[bold]渠道侧明细[/]")
        for r in db.q(conn, """
            SELECT r.*, b.bill_date FROM channel_bill_records r
            JOIN channel_bills b ON b.id=r.bill_id WHERE r.channel_txn_no=?
        """, (d["channel_txn_no"],)):
            memo = r["memo"] or ""
            flag = " [red]<< memo 含可疑指令[/]" if any(
                k in memo.lower() for k in ("writeoff", "核销", "ignore", "忽略", "system", "assistant")
            ) else ""
            console.print(f"  {r['id']} bill={r['bill_date']} {r['rec_type']} "
                          f"amount={fmt(r['amount_cents'])} fee={fmt(r['fee_cents'])} "
                          f"{r['currency']} @{r['occurred_at']} memo={memo!r}{flag}")

        console.print("\n[bold]我方侧明细[/]")
        for r in db.q(conn, "SELECT * FROM payments WHERE channel_txn_no=?", (d["channel_txn_no"],)):
            console.print(f"  payment {r['id']} order={r['order_id']} "
                          f"amount={fmt(r['amount_cents'])} fee={fmt(r['fee_cents'])} "
                          f"status={r['status']} paid_at={r['paid_at']} "
                          f"callback_at={r['callback_at']}")
        for r in db.q(conn, "SELECT * FROM refunds WHERE channel_txn_no=?", (d["channel_txn_no"],)):
            console.print(f"  refund  {r['id']} order={r['order_id']} "
                          f"amount={fmt(r['amount_cents'])} kind={r['kind']} "
                          f"mode={r['mode']} refunded_at={r['refunded_at']}")

    if d["our_ref_type"] == "settlement":
        s = db.q1(conn, "SELECT * FROM settlements WHERE id=?", (d["our_ref_id"],))
        if s:
            console.print(f"\n[bold]结算单[/] {s['id']} 商户={s['merchant_id']} "
                          f"金额={fmt(s['amount_cents'])} 状态={s['status']} "
                          f"期间={s['period_start']}")

    if show_answer:
        g = db.q1(conn, "SELECT * FROM diff_ground_truth WHERE diff_id=?", (diff_id,))
        if g:
            console.rule("[bold red]答案（agent 不可见）[/]")
            console.print(f"root_causes     {db.jload(g['root_causes'])}")
            console.print(f"correct_actions {db.jload(g['correct_actions'])}")
            console.print(f"expected_status {g['expected_status']}")
            console.print(f"is_composite    {bool(g['is_composite'])}")
            console.print(f"[dim]{g['explanation']}[/]")
    conn.close()


@cli.command("tasks")
@click.option("--db-path", default=None)
def tasks_cmd(db_path: str | None) -> None:
    """任务集概览。"""
    from .eval.tasks import load_tasks, task_summary
    conn = db.connect(db_path)
    s = task_summary(load_tasks(conn))
    t = Table(title="任务集", show_header=False)
    t.add_column(style="cyan"); t.add_column(justify="right")
    t.add_row("任务数（按差错）", str(s["tasks"]))
    t.add_row("逻辑问题数（去重后）", str(s["logical_issues"]))
    t.add_row("复合差错", str(s["composite"]))
    t.add_row("含提示注入", str(s["with_injection"]))
    for k, v in s["by_source"].items():
        t.add_row(f"来源 {k}", str(v))
    console.print(t)
    conn.close()


@cli.command("eval-baseline")
@click.option("--db-path", default=None)
@click.option("--limit", default=None, type=int, help="只跑前 N 条，调试用")
@click.option("--save/--no-save", default=True, help="写 docs/stage1_baseline.md")
@click.option("--per-code/--no-per-code", default=True)
def eval_baseline_cmd(db_path: str | None, limit: int | None,
                      save: bool, per_code: bool) -> None:
    """跑纯规则基线并判分。

    ⚠️ 这是阶段 1 的核心动作，也是判分器的校准手段：
       规则在 D03/D04/D20 这类单一原因差错上应接近 90%+。
       如果规则只跑出 50%，那不是规则不行，是判分器或标注错了。
    """
    from .baseline.rules import run_baseline
    from .eval import report as rp
    from .eval.grader import aggregate
    from .eval.tasks import load_tasks

    conn = db.connect(db_path)
    tasks = load_tasks(conn, limit=limit)
    if not tasks:
        console.print("[red]任务集为空，先跑 make build[/]")
        return
    sols = run_baseline(conn, tasks)
    rep = aggregate("rule_baseline", tasks, sols)

    rp.print_report(rep)
    if per_code:
        rp.print_per_code(rep)
    rp.print_confusion(rep)

    if save:
        p = rp.save_markdown(rep, DOCS / "stage1_baseline.md", title="纯规则基线")
        console.print(f"[green]已写入[/] {p}")
    conn.close()


@cli.command("eval-agent")
@click.option("--db-path", default=None)
@click.option("--limit", default=60, show_default=True, help="抽样任务数（跑全量要花钱）")
@click.option("--workers", default=8, show_default=True, help="并发数")
@click.option("--max-steps", default=14, show_default=True, help="每条差错最多几轮决策")
@click.option("--model", default=None, help="覆盖 RECON_AGENT_MODEL")
@click.option("--all-tasks", is_flag=True, help="跑全量，不抽样")
@click.option("--save/--no-save", default=True)
def eval_agent_cmd(db_path: str | None, limit: int, workers: int, max_steps: int,
                   model: str | None, all_tasks: bool, save: bool) -> None:
    """跑 Agent 并与规则基线在**同一批任务**上对比。

    这条命令的产出就是整个项目的核心交付物：那张对比表。
    """
    from .agent.llm import DeepSeekClient, Pricing
    from .agent.solver import persist_runs, run_agent, stop_reason_stats, tool_usage_stats
    from .baseline.rules import run_baseline
    from .eval import report as rp
    from .eval.grader import aggregate
    from .eval.tasks import default_ensure, load_tasks, sample_tasks
    from .world.injector import TEXT_DEPENDENT_CODES

    conn = db.connect(db_path)
    every = load_tasks(conn)
    if not every:
        console.print("[red]任务集为空，先跑 make build[/]")
        return
    tasks = every if all_tasks else sample_tasks(every, limit, ensure=default_ensure(limit))

    n_text = sum(1 for t in tasks if set(t.gold_codes) & TEXT_DEPENDENT_CODES)
    console.print(f"任务：[bold]{len(tasks)}[/] / 全量 {len(every)}；"
                  f"其中需读自由文本 [magenta]{n_text}[/] 条")
    if not all_tasks:
        console.print("[yellow]注意[/] 抽样对「需读自由文本」两类做了定向过采样，"
                      "整体数字不代表全量分布 —— 看分组指标。")

    pricing = Pricing.from_env()
    if not pricing.configured:
        console.print("[yellow]价格未配置[/]，成本一栏按 0 显示。"
                      "配 RECON_PRICE_IN_MISS_PER_MTOK / _IN_HIT_ / _OUT_ 后才有成本数字。")

    client = DeepSeekClient(model=model)
    console.print(f"模型：[cyan]{client.name}[/]  并发 {workers}  最多 {max_steps} 轮/条")

    done = {"n": 0}

    def tick(i, total, r):
        done["n"] = i
        if i % 10 == 0 or i == total:
            console.print(f"  …{i}/{total}", end="\r")

    sols, results = run_agent(db_path, tasks, llm=client, max_steps=max_steps,
                              workers=workers, progress=tick)
    console.print(" " * 40, end="\r")

    agent_rep = aggregate(client.name, tasks, sols)
    base_rep = aggregate("rule_baseline", tasks, run_baseline(conn, tasks))

    rp.print_report(agent_rep)
    rp.print_per_code(agent_rep)
    rp.print_confusion(agent_rep)

    console.rule("[bold]核心交付物：同一批任务上的对比[/]")
    rp.print_comparison([base_rep, agent_rep])

    t = Table(title="Agent 过程诊断", show_header=True)
    t.add_column("工具"); t.add_column("调用", justify="right")
    t.add_column("失败", justify="right"); t.add_column("失败率", justify="right")
    for name, s in tool_usage_stats(results).items():
        t.add_row(name, str(s["calls"]), str(s["errors"]), f"{s['error_rate']:.1%}")
    console.print(t)
    console.print("停止原因：", stop_reason_stats(results))

    n = persist_runs(conn, results, solver=f"agent:{client.name}")
    console.print(f"[green]已落轨迹[/] {n} 次运行 -> agent_runs / agent_steps")

    if save:
        p1 = rp.save_markdown(agent_rep, DOCS / "stage2_agent.md",
                              title=f"Agent（{client.name}）")
        p2 = DOCS / "stage2_comparison.md"
        p2.write_text(rp.comparison_markdown([base_rep, agent_rep], tasks=tasks),
                      encoding="utf-8")
        console.print(f"[green]已写入[/] {p1}\n[green]已写入[/] {p2}")
    conn.close()


@cli.command("replay")
@click.argument("task_id", required=False)
@click.option("--db-path", default=None)
@click.option("--failed-only", is_flag=True, help="只挑判错的")
def replay_cmd(task_id: str | None, db_path: str | None, failed_only: bool) -> None:
    """Trace 回放 —— 逐步看 agent 那一轮到底想了什么、调了什么、拿到了什么。"""
    conn = db.connect(db_path)
    if task_id is None:
        rows = db.q(conn, """
            SELECT r.task_id, r.root_causes, g.root_causes AS gold
            FROM agent_runs r JOIN diff_ground_truth g ON g.diff_id = r.diff_id
            ORDER BY r.id""")
        if not rows:
            console.print("[red]没有轨迹，先跑 eval-agent[/]")
            return
        if failed_only:
            # ⚠️ 必须按集合比，不能比 JSON 字符串。
            #    ["D04","D14"] 和 ["D14","D04"] 是同一个答案，
            #    字符串比较会把它算成错判（判分器本身用的是集合，没这个问题）。
            rows = [r for r in rows
                    if set(db.jload(r["root_causes"]) or []) != set(db.jload(r["gold"]) or [])]
            if not rows:
                console.print("[green]没有判错的[/]")
                return
        task_id = rows[0]["task_id"]

    run = db.q1(conn, "SELECT * FROM agent_runs WHERE task_id=?", (task_id,))
    if run is None:
        console.print(f"[red]没有 {task_id} 的轨迹[/]")
        return

    console.rule(f"[bold]{task_id}[/]  model={run['model']}  停止={run['stop_reason']}")
    console.print(f"结论 root_causes={db.jload(run['root_causes'])} "
                  f"actions={db.jload(run['actions'])} "
                  f"status={run['expected_status']} conf={run['confidence']:.2f}")
    console.print(f"[dim]{run['notes'][:600]}[/]")
    console.print(f"步数 {run['steps']}  取证 {run['reads']}  "
                  f"token {run['tokens_in']}/{run['tokens_out']}  "
                  f"耗时 {run['latency_ms']}ms")

    g = db.q1(conn, "SELECT * FROM diff_ground_truth WHERE diff_id=?", (run["diff_id"],))
    if g:
        console.print(f"[bold red]答案[/] {db.jload(g['root_causes'])} "
                      f"{db.jload(g['correct_actions'])} {g['expected_status']}")

    console.rule("逐步轨迹")
    for st in db.q(conn, "SELECT * FROM agent_steps WHERE run_id=? ORDER BY step_no, id",
                   (run["id"],)):
        mark = "[green]✓[/]" if st["ok"] else "[red]✗[/]"
        console.print(f"\n[bold]step {st['step_no']}[/] {mark} "
                      f"tool=[cyan]{st['tool'] or 'CONCLUDE'}[/] "
                      f"args={st['arguments'] or ''}")
        if st["thought"]:
            console.print(f"  [dim]思考：{st['thought']}[/]")
        if st["result_digest"]:
            console.print(f"  返回：{st['result_digest'][:500]}")
    conn.close()


@cli.command("verify-repro")
@click.option("--seed", default=42)
@click.option("--days", default=2)
@click.option("--orders-per-day", default=120)
def verify_repro_cmd(seed: int, days: int, orders_per_day: int) -> None:
    """同 seed 跑两次，校验产出完全一致。"""
    import hashlib
    fingerprints = []
    for i in range(2):
        path = db.PROJECT_ROOT / "data" / f"_repro{i}.db"
        cfg = _cfg(seed, "2026-07-01", days, orders_per_day, 80)
        dates = _dates("2026-07-01", days)
        conn = db.init_db(path, reset=True)
        generate(conn, cfg)
        build_bills(conn, "2026-07-01", days, seed=seed)
        board = build_notices(conn, seed, dates)
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
        fingerprints.append(hashlib.sha256(payload.encode()).hexdigest())
        conn.close()
        path.unlink(missing_ok=True)

    if fingerprints[0] == fingerprints[1]:
        console.print(f"[green]可复现[/] sha256={fingerprints[0][:16]}…")
    else:
        console.print(f"[red]不可复现[/] {fingerprints}")
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
