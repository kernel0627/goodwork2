"""方差与 pass^k —— 读消融表的前提条件。

为什么必须有这个：同一个配置（temperature=0）在阶段 2 跑出「需读文本 92.9%」，
阶段 3 跑出「85.7%」—— 相差 7 个百分点。而消融阶梯里各级的差异也就 5~11 个点。

**在不知道噪声下限之前，解读这种量级的差异是无效的。**
先量方差，再读消融表。顺序颠倒的话，会把噪声当成结论去追。

pass^k 才是可靠性的正确度量：同一条任务独立跑 k 次**全对**的比例。
业界（τ-bench）也是这么衡量的 —— pass^1 六成、pass^8 掉到两成是常态，
「单次成功率」会系统性高估 agent 的可用程度。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev

from .grader import Report

KEY_METRICS = (
    ("归因 exact", "attr_exact", True),
    ("需读文本 exact", "attr_exact_text_dependent", True),
    ("规则可解 exact", "attr_exact_rule_solvable", True),
    ("复合 exact", "attr_exact_composite", True),
    ("动作 exact", "action_exact", True),
    ("终态正确", "status_acc", True),
    ("UNKNOWN 率", "unknown_rate", True),
    ("过度转人工 条", "over_escalation_n", False),
    ("错误动账 条", "wrong_money_action_n", False),
    ("越权 条", "unauthorized_n", False),
    ("平均决策轮数", "avg_steps", False),
    ("未缓存输入 token", "avg_uncached_in", False),
)


@dataclass
class Spread:
    label: str
    key: str
    is_rate: bool
    values: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return mean(self.values) if self.values else 0.0

    @property
    def lo(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def hi(self) -> float:
        return max(self.values) if self.values else 0.0

    @property
    def sd(self) -> float:
        return pstdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def band(self) -> float:
        """极差 —— 噪声下限的直观度量。"""
        return self.hi - self.lo

    def fmt(self) -> tuple[str, str, str]:
        if self.is_rate:
            return (f"{self.mean * 100:.1f}%",
                    f"{self.lo * 100:.1f}% ~ {self.hi * 100:.1f}%",
                    f"±{self.band * 100 / 2:.1f}pp")
        return (f"{self.mean:.1f}", f"{self.lo:.1f} ~ {self.hi:.1f}",
                f"±{self.band / 2:.1f}")


@dataclass
class VarianceReport:
    config: str
    k: int
    n_tasks: int
    spreads: list[Spread] = field(default_factory=list)

    pass_1: float = 0.0          # 单次正确率的均值
    pass_k: float = 0.0          # k 次全对的比例  ⭐
    fail_k: float = 0.0          # k 次全错的比例
    flip_rate: float = 0.0       # 判定在 k 次之间发生过翻转的比例
    unstable_tasks: list[str] = field(default_factory=list)

    # 需读文本子集单独看一遍
    pass_1_text: float = 0.0
    pass_k_text: float = 0.0

    @property
    def noise_floor_pp(self) -> float:
        """关键准确率指标的最大极差（百分点）—— 小于这个数的差异不该当结论。"""
        rates = [s.band for s in self.spreads
                 if s.is_rate and s.key.startswith("attr_exact")]
        return max(rates, default=0.0) * 100


def analyse(config: str, reps: list[Report], *,
            text_dependent_keys: set[str] | None = None) -> VarianceReport:
    from ..world.injector import TEXT_DEPENDENT_CODES
    assert reps, "至少要一次运行"
    k = len(reps)
    vr = VarianceReport(config=config, k=k, n_tasks=reps[0].n)

    for label, key, is_rate in KEY_METRICS:
        vr.spreads.append(Spread(label, key, is_rate,
                                 [float(r.metrics.get(key, 0.0)) for r in reps]))

    # 逐任务：k 次里对了几次
    hits: dict[str, list[bool]] = {}
    is_text: dict[str, bool] = {}
    for r in reps:
        for g in r.grades:
            hits.setdefault(g.task_id, []).append(g.attr_exact)
            is_text[g.task_id] = bool(set(g.gold_codes) & TEXT_DEPENDENT_CODES)

    full = [t for t, v in hits.items() if len(v) == k]
    if full:
        vr.pass_1 = mean(sum(hits[t]) / k for t in full)
        vr.pass_k = sum(all(hits[t]) for t in full) / len(full)
        vr.fail_k = sum(not any(hits[t]) for t in full) / len(full)
        unstable = [t for t in full if any(hits[t]) and not all(hits[t])]
        vr.flip_rate = len(unstable) / len(full)
        vr.unstable_tasks = sorted(unstable)

        txt = [t for t in full if is_text[t]]
        if txt:
            vr.pass_1_text = mean(sum(hits[t]) / k for t in txt)
            vr.pass_k_text = sum(all(hits[t]) for t in txt) / len(txt)
    return vr


def print_variance(vr: VarianceReport) -> None:
    from rich.console import Console
    from rich.table import Table
    console = Console()

    t = Table(title=f"方差：{vr.config}  同配置独立跑 {vr.k} 次 / {vr.n_tasks} 条任务")
    t.add_column("指标", style="cyan"); t.add_column("均值", justify="right")
    t.add_column("区间", justify="right"); t.add_column("半极差", justify="right")
    for s in vr.spreads:
        m, band, half = s.fmt()
        t.add_row(s.label, m, band, half)
    console.print(t)

    t2 = Table(title="可靠性（pass^k）", show_header=False)
    t2.add_column(style="cyan"); t2.add_column(justify="right")
    t2.add_row("pass^1（单次正确率均值）", f"{vr.pass_1:.1%}")
    t2.add_row(f"[bold]pass^{vr.k}（{vr.k} 次全对）[/]", f"[bold]{vr.pass_k:.1%}[/]")
    t2.add_row(f"fail^{vr.k}（{vr.k} 次全错）", f"{vr.fail_k:.1%}")
    t2.add_row("[yellow]翻转率（判定不稳定）[/]", f"[yellow]{vr.flip_rate:.1%}[/]")
    t2.add_row("需读文本 pass^1", f"{vr.pass_1_text:.1%}")
    t2.add_row(f"需读文本 pass^{vr.k}", f"{vr.pass_k_text:.1%}")
    console.print(t2)

    console.print(f"\n[bold yellow]噪声下限约 ±{vr.noise_floor_pp / 2:.1f} 个百分点[/] "
                  f"—— 小于这个量级的消融差异不能当结论。")


def variance_markdown(vr: VarianceReport) -> str:
    L = [f"# 方差与 pass^k：{vr.config}", "",
         f"同配置独立跑 **{vr.k}** 次，{vr.n_tasks} 条任务，temperature=0。", "",
         "## 为什么先做这个", "",
         "同一配置在阶段 2 跑出「需读文本 92.9%」、阶段 3 跑出「85.7%」，相差 7 个百分点；",
         "而消融阶梯各级之间的差异也就 5~11 个点。**不知道噪声下限就解读这种差异是无效的。**", "",
         "## 指标波动", "",
         "| 指标 | 均值 | 区间 | 半极差 |", "|---|---:|---:|---:|"]
    for s in vr.spreads:
        m, band, half = s.fmt()
        L.append(f"| {s.label} | {m} | {band} | {half} |")
    L += ["", "## 可靠性", "", "| 指标 | 值 |", "|---|---:|",
          f"| pass^1（单次正确率均值） | {vr.pass_1:.1%} |",
          f"| **pass^{vr.k}（{vr.k} 次全对）** | **{vr.pass_k:.1%}** |",
          f"| fail^{vr.k}（{vr.k} 次全错） | {vr.fail_k:.1%} |",
          f"| 翻转率（判定不稳定） | {vr.flip_rate:.1%} |",
          f"| 需读文本 pass^1 | {vr.pass_1_text:.1%} |",
          f"| 需读文本 pass^{vr.k} | {vr.pass_k_text:.1%} |", "",
          f"> **噪声下限约 ±{vr.noise_floor_pp / 2:.1f} 个百分点。**",
          f"> 小于这个量级的消融差异不能当结论。", ""]
    if vr.unstable_tasks:
        L += ["## 判定不稳定的任务", "",
              f"{len(vr.unstable_tasks)} 条任务的判定在 {vr.k} 次运行之间发生过翻转：", "",
              "```", *vr.unstable_tasks[:40], "```", ""]
    return "\n".join(L)
