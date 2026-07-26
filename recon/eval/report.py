"""出表 —— 终端看 + markdown 落盘。

报表的规矩：
1. 风险指标（误核销 / 越权 / 漏转人工）**同时给条数和金额**。金额才是真实损失度量。
2. 逐类召回必须全列。「整体 78%」会掩盖「D09 只有 30%」这种关键信息。
3. 混淆对必须列出来。它是 bad case 归因的第一入口，也是下一轮该改什么的依据。
4. 成本必须和准确率放在同一张表里。不给成本的准确率提升是没有意义的。
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from ..money import fmt
from ..world.injector import CODES
from .grader import Report, confusion

console = Console()


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _pct_or_dash(x: float, n: int) -> str:
    """分母为 0 时给「—」。

    ⚠️ 曾经把「样本里没有含注入的任务」显示成「注入抵抗率 0.0%」，
       那是在报一个不存在的失败。分母为 0 必须如实说没样本。
    """
    return _pct(x) if n else "—"


# --------------------------------------------------------------------------

def print_report(rep: Report) -> None:
    m = rep.metrics
    t = Table(title=f"[{rep.solver}]  {rep.n} 条差错 / {rep.n_logical} 个逻辑问题",
              show_header=False)
    t.add_column(style="cyan", no_wrap=True)
    t.add_column(justify="right")

    t.add_row("[bold]归因[/]", "")
    t.add_row("  完全正确 (exact)", _pct(m["attr_exact"]))
    t.add_row("  首要原因命中 (top-1)", _pct(m["attr_top1"]))
    t.add_row("  micro-F1", _pct(m["attr_f1"]))
    t.add_row("  完全正确 / 去重后", _pct(m["attr_exact_dedup"]))
    t.add_row(f"  原子差错 (n={m['n_atomic']})", _pct(m["attr_exact_atomic"]))
    t.add_row(f"  复合差错 (n={m['n_composite']})", _pct(m["attr_exact_composite"]))
    t.add_row("  认不出来 (UNKNOWN)", _pct(m["unknown_rate"]))

    t.add_row("[bold]⭐ 规则可解 vs 需读自由文本[/]", "")
    t.add_row(f"  规则可解 (n={m['n_rule_solvable']})",
              _pct_or_dash(m["attr_exact_rule_solvable"], m["n_rule_solvable"]))
    t.add_row(f"  [magenta]需读文本 (n={m['n_text_dependent']})[/]",
              f"[magenta]{_pct_or_dash(m['attr_exact_text_dependent'], m['n_text_dependent'])}[/]")
    t.add_row("  需读文本 · 动作正确",
              _pct_or_dash(m["action_exact_text_dependent"], m["n_text_dependent"]))
    t.add_row("  [red]需读文本 · 错误动账[/]",
              f"{m['text_dep_wrong_money_n']} 条 / {fmt(m['text_dep_wrong_money_amount'])} 元")

    t.add_row("[bold]处置[/]", "")
    t.add_row("  动作集合完全一致", _pct(m["action_exact"]))
    t.add_row("  动作 Jaccard", _pct(m["action_jaccard"]))
    t.add_row("  终态正确", _pct(m["status_acc"]))

    t.add_row("[bold]风险（条数 / 金额）[/]", "")
    t.add_row("  [red]误核销[/]",
              f"{m['false_writeoff_n']} 条 / {fmt(m['false_writeoff_amount'])} 元")
    t.add_row("  [red]错误动账[/]",
              f"{m['wrong_money_action_n']} 条 / {fmt(m['wrong_money_action_amount'])} 元")
    t.add_row("  [red]越权处置[/]",
              f"{m['unauthorized_n']} 条 / {fmt(m['unauthorized_amount'])} 元")
    t.add_row("  漏转人工",
              f"{m['missed_escalation_n']} 条 / {fmt(m['missed_escalation_amount'])} 元")
    t.add_row("  过度转人工", f"{m['over_escalation_n']} 条")
    t.add_row("  转人工 精确率 / 召回率",
              f"{_pct(m['escalation_precision'])} / {_pct(m['escalation_recall'])}")
    t.add_row("  任一风险发生率", _pct(m["any_risk_rate"]))
    t.add_row("  差错池总金额", f"{fmt(m['total_amount'])} 元")

    t.add_row("[bold]提示注入[/]", "")
    t.add_row(f"  含注入任务 (n={m['injection_n']}) 抵抗率",
              _pct_or_dash(m["injection_resist_rate"], m["injection_n"]))

    t.add_row("[bold]成本[/]", "")
    t.add_row("  平均取证次数", f"{m['avg_reads']:.1f}")
    t.add_row("  平均读取行数 / 字符数",
              f"{m['avg_rows_read']:.1f} / {m['avg_chars_read']:.0f}")
    t.add_row("  平均决策轮数", f"{m['avg_steps']:.2f}")
    t.add_row("  平均 token (in/out)",
              f"{m['avg_tokens_in']:.0f} / {m['avg_tokens_out']:.0f}")
    t.add_row("  其中未缓存输入 / 缓存命中率",
              f"{m['avg_uncached_in']:.0f} / {_pct(m['cache_hit_rate'])}")
    t.add_row("  总成本", f"{m['total_cost_micro_cny'] / 1_000_000:.4f} 元")
    t.add_row("  平均延迟", f"{m['avg_latency_ms']:.0f} ms")
    console.print(t)


def print_per_code(rep: Report) -> None:
    t = Table(title=f"[{rep.solver}] 逐类表现")
    t.add_column("码"); t.add_column("名称"); t.add_column("任务数", justify="right")
    t.add_column("该类召回", justify="right"); t.add_column("整条全对", justify="right")
    t.add_column("涉及金额", justify="right")
    for code, s in rep.per_code.items():
        color = "green" if s["recall"] >= 0.9 else ("yellow" if s["recall"] >= 0.6 else "red")
        t.add_row(code, CODES[code].name if code in CODES else code, str(s["n"]),
                  f"[{color}]{_pct(s['recall'])}[/]", _pct(s["exact_rate"]),
                  fmt(s["amount"]))
    console.print(t)


def print_confusion(rep: Report, top: int = 12) -> None:
    pairs = confusion(rep, top=top)
    if not pairs:
        console.print("[green]无错判[/]")
        return
    t = Table(title=f"[{rep.solver}] 最常见错判对（Bad case 归因入口）")
    t.add_column("答案"); t.add_column("判成"); t.add_column("次数", justify="right")
    for gold, pred, n in pairs:
        t.add_row(gold, pred, str(n))
    console.print(t)


# --------------------------------------------------------------------------

def to_markdown(rep: Report, *, title: str = "") -> str:
    m = rep.metrics
    L: list[str] = []
    L.append(f"# {title or rep.solver} 评测报告")
    L.append("")
    L.append(f"- 任务数：**{rep.n}** 条差错 / **{rep.n_logical}** 个逻辑问题")
    L.append("")
    L.append("## 归因")
    L.append("")
    L.append("| 指标 | 值 |")
    L.append("|---|---:|")
    for label, key in [("完全正确 exact", "attr_exact"), ("首要原因命中 top-1", "attr_top1"),
                       ("micro-F1", "attr_f1"), ("完全正确（去重后）", "attr_exact_dedup"),
                       ("原子差错 exact", "attr_exact_atomic"),
                       ("复合差错 exact", "attr_exact_composite"),
                       ("认不出来 UNKNOWN", "unknown_rate")]:
        L.append(f"| {label} | {_pct(m[key])} |")
    L.append(f"| 原子 / 复合 任务数 | {m['n_atomic']} / {m['n_composite']} |")

    L.append("")
    L.append("## 处置")
    L.append("")
    L.append("| 指标 | 值 |")
    L.append("|---|---:|")
    L.append(f"| 动作集合完全一致 | {_pct(m['action_exact'])} |")
    L.append(f"| 动作 Jaccard | {_pct(m['action_jaccard'])} |")
    L.append(f"| 终态正确 | {_pct(m['status_acc'])} |")

    L.append("")
    L.append("## 风险指标（条数 / 金额）")
    L.append("")
    L.append("> 金额才是真实损失度量，条数会骗人。")
    L.append("")
    L.append("| 风险 | 条数 | 涉及金额（元） |")
    L.append("|---|---:|---:|")
    L.append(f"| 误核销 | {m['false_writeoff_n']} | {fmt(m['false_writeoff_amount'])} |")
    L.append(f"| 错误动账 | {m['wrong_money_action_n']} | {fmt(m['wrong_money_action_amount'])} |")
    L.append(f"| 越权处置 | {m['unauthorized_n']} | {fmt(m['unauthorized_amount'])} |")
    L.append(f"| 漏转人工 | {m['missed_escalation_n']} | {fmt(m['missed_escalation_amount'])} |")
    L.append(f"| 过度转人工 | {m['over_escalation_n']} | — |")
    L.append("")
    L.append(f"- 转人工 精确率 / 召回率：{_pct(m['escalation_precision'])} / "
             f"{_pct(m['escalation_recall'])}")
    L.append(f"- 任一风险发生率：{_pct(m['any_risk_rate'])}")
    L.append(f"- 差错池总金额：{fmt(m['total_amount'])} 元")
    L.append(f"- 提示注入抵抗率（n={m['injection_n']}）：{_pct_or_dash(m['injection_resist_rate'], m['injection_n'])}")

    L.append("")
    L.append("## 成本")
    L.append("")
    L.append("| 指标 | 值 |")
    L.append("|---|---:|")
    L.append(f"| 平均取证次数 | {m['avg_reads']:.1f} |")
    L.append(f"| 平均读取行数 | {m['avg_rows_read']:.1f} |")
    L.append(f"| 平均读取字符数 | {m['avg_chars_read']:.0f} |")
    L.append(f"| 平均决策轮数 | {m['avg_steps']:.2f} |")
    L.append(f"| 平均 token in/out | {m['avg_tokens_in']:.0f} / {m['avg_tokens_out']:.0f} |")
    L.append(f"| 总成本（元） | {m['total_cost_micro_cny'] / 1_000_000:.4f} |")
    L.append(f"| 平均延迟（ms） | {m['avg_latency_ms']:.0f} |")

    L.append("")
    L.append("## 逐类表现")
    L.append("")
    L.append("| 码 | 名称 | 任务数 | 该类召回 | 整条全对 | 涉及金额（元） |")
    L.append("|---|---|---:|---:|---:|---:|")
    for code, s in rep.per_code.items():
        name = CODES[code].name if code in CODES else code
        L.append(f"| {code} | {name} | {s['n']} | {_pct(s['recall'])} | "
                 f"{_pct(s['exact_rate'])} | {fmt(s['amount'])} |")

    L.append("")
    L.append("## 最常见错判对")
    L.append("")
    L.append("| 答案 | 判成 | 次数 |")
    L.append("|---|---|---:|")
    for gold, pred, n in confusion(rep, top=15):
        L.append(f"| {gold} | {pred} | {n} |")
    L.append("")
    return "\n".join(L)


def save_markdown(rep: Report, path: str | Path, *, title: str = "") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_markdown(rep, title=title), encoding="utf-8")
    return p


# --------------------------------------------------------------------------

COMPARISON_ROWS = [
    ("归因 exact", lambda m: _pct(m["attr_exact"])),
    ("归因 top-1", lambda m: _pct(m["attr_top1"])),
    ("归因 F1", lambda m: _pct(m["attr_f1"])),
    ("原子 exact", lambda m: _pct(m["attr_exact_atomic"])),
    ("复合 exact", lambda m: _pct(m["attr_exact_composite"])),
    ("规则可解 exact", lambda m: _pct(m["attr_exact_rule_solvable"])),
    ("⭐需读文本 exact",
     lambda m: _pct_or_dash(m["attr_exact_text_dependent"], m["n_text_dependent"])),
    ("⭐需读文本 动作正确",
     lambda m: _pct_or_dash(m["action_exact_text_dependent"], m["n_text_dependent"])),
    ("动作 exact", lambda m: _pct(m["action_exact"])),
    ("终态正确", lambda m: _pct(m["status_acc"])),
    ("UNKNOWN 率", lambda m: _pct(m["unknown_rate"])),
    ("误核销 条/元", lambda m: f"{m['false_writeoff_n']}/{fmt(m['false_writeoff_amount'])}"),
    ("错误动账 条/元", lambda m: f"{m['wrong_money_action_n']}/{fmt(m['wrong_money_action_amount'])}"),
    ("越权 条/元", lambda m: f"{m['unauthorized_n']}/{fmt(m['unauthorized_amount'])}"),
    ("漏转人工 条", lambda m: str(m["missed_escalation_n"])),
    ("过度转人工 条", lambda m: str(m["over_escalation_n"])),
    ("注入抵抗率", lambda m: _pct_or_dash(m["injection_resist_rate"], m["injection_n"])),
    ("平均取证次数", lambda m: f"{m['avg_reads']:.1f}"),
    ("平均决策轮数", lambda m: f"{m['avg_steps']:.1f}"),
    ("平均 token 合计", lambda m: f"{m['avg_tokens_in'] + m['avg_tokens_out']:.0f}"),
    ("⭐未缓存输入 token", lambda m: f"{m['avg_uncached_in']:.0f}"),
    ("缓存命中率", lambda m: _pct(m["cache_hit_rate"])),
    ("总成本 元", lambda m: f"{m['total_cost_micro_cny'] / 1e6:.4f}"),
    ("平均延迟 ms", lambda m: f"{m['avg_latency_ms']:.0f}"),
]


def comparison_markdown(reps: list[Report], *, tasks=None) -> str:
    L = ["# 求解方对比", ""]
    if reps:
        L.append(f"- 任务数：**{reps[0].n}** 条差错 / {reps[0].n_logical} 个逻辑问题")
        L.append(f"- 其中判据在结构化数据里：{reps[0].metrics['n_rule_solvable']} 条；"
                 f"判据只在自由文本里：**{reps[0].metrics['n_text_dependent']}** 条")
        L.append("")
        L.append("> 抽样对「需读自由文本」两类做了定向过采样，整体数字不代表全量分布。"
                 "结论请看分组指标。")
        L.append("")
    L.append("| 指标 | " + " | ".join(r.solver for r in reps) + " |")
    L.append("|---|" + "---:|" * len(reps))
    for label, fn in COMPARISON_ROWS:
        L.append(f"| {label} | " + " | ".join(fn(r.metrics) for r in reps) + " |")
    L.append("")
    return "\n".join(L)


def print_comparison(reps: list[Report]) -> None:
    """多个求解方同表对比 —— 这张表就是这个项目最终要交付的东西。"""
    if not reps:
        return
    t = Table(title="求解方对比")
    t.add_column("指标", style="cyan", no_wrap=True)
    for r in reps:
        t.add_column(r.solver, justify="right")
    for label, fn in COMPARISON_ROWS:
        style = "magenta" if label.startswith("⭐") else None
        vals = [fn(r.metrics) for r in reps]
        if style:
            vals = [f"[{style}]{v}[/]" for v in vals]
        t.add_row(label, *vals)
    console.print(t)
