"""阶段 6.2：二元 exact 指标的严格配对统计。

聚合准确率相减会丢掉最重要的信息：两套方案是否在**同一条任务**上发生翻转。
本模块只接受同世界、同任务集合的运行，并提供：

1. 逐任务配对的四格表；
2. accuracy 的 Wilson 95% CI；
3. 候选方案相对基线的 paired bootstrap 95% CI；
4. 二元 exact 指标的 exact McNemar 检验；
5. 5~10 次同编号运行的分层 paired bootstrap（先抽运行，再在运行内抽任务）。

所有随机过程都有显式 seed。样本集合不一致时直接拒绝，不能静默取交集。
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .. import db

DEFAULT_BOOTSTRAP_REPS = 20_000
DEFAULT_SEED = 20260727
Z_95 = 1.959963984540054
SUPPORTED_METRICS = ("attr_exact", "action_exact")


class PairedStatsError(ValueError):
    """输入运行不满足严格配对条件。"""


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    solver: str
    model: str
    world_fingerprint: str
    code_rev: str
    config: dict


@dataclass(frozen=True)
class RunOutcomes:
    info: RunInfo
    outcomes: dict[str, bool]
    gold_codes: dict[str, str | None]


@dataclass(frozen=True)
class PairedSample:
    name_a: str
    name_b: str
    task_ids: tuple[str, ...]
    outcomes_a: tuple[bool, ...]
    outcomes_b: tuple[bool, ...]

    @property
    def differences(self) -> tuple[int, ...]:
        """B - A；正值表示候选 B 改善。"""
        return tuple(int(b) - int(a)
                     for a, b in zip(self.outcomes_a, self.outcomes_b))


@dataclass(frozen=True)
class PairedComparison:
    name_a: str
    name_b: str
    n: int
    both_correct: int
    only_a_correct: int
    only_b_correct: int
    both_wrong: int
    accuracy_a: float
    accuracy_b: float
    accuracy_a_ci95: tuple[float, float]
    accuracy_b_ci95: tuple[float, float]
    delta_b_minus_a: float
    delta_ci95: tuple[float, float]
    mcnemar_p: float
    bootstrap_reps: int
    bootstrap_seed: int


@dataclass(frozen=True)
class RepeatedComparison:
    n_pairs: int
    n_tasks: int
    mean_delta_b_minus_a: float
    hierarchical_ci95: tuple[float, float]
    bootstrap_reps: int
    bootstrap_seed: int


def wilson_interval(successes: int, n: int,
                    z: float = Z_95) -> tuple[float, float]:
    """二项比例的 Wilson 区间；小样本下比正态近似稳定。"""
    if n <= 0:
        raise PairedStatsError("Wilson 区间要求 n > 0")
    if not 0 <= successes <= n:
        raise PairedStatsError("successes 必须位于 [0, n]")
    p = successes / n
    z2 = z * z
    den = 1 + z2 / n
    center = (p + z2 / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / den
    return max(0.0, center - half), min(1.0, center + half)


def exact_mcnemar_p(only_a_correct: int, only_b_correct: int) -> float:
    """双侧 exact McNemar：在不一致对上做 Binomial(n, 0.5) 检验。"""
    if only_a_correct < 0 or only_b_correct < 0:
        raise PairedStatsError("McNemar 四格表计数不能为负")
    discordant = only_a_correct + only_b_correct
    if discordant == 0:
        return 1.0
    tail = min(only_a_correct, only_b_correct)
    numerator = sum(math.comb(discordant, k) for k in range(tail + 1))
    return min(1.0, 2.0 * numerator / (2 ** discordant))


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise PairedStatsError("分位数输入不能为空")
    if not 0 <= probability <= 1:
        raise PairedStatsError("probability 必须位于 [0, 1]")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * probability
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def paired_bootstrap_ci(differences: Sequence[int | float], *,
                        reps: int = DEFAULT_BOOTSTRAP_REPS,
                        seed: int = DEFAULT_SEED) -> tuple[float, float]:
    """对逐任务差值重采样，返回 percentile 95% CI。"""
    if not differences:
        raise PairedStatsError("paired bootstrap 至少需要一条任务")
    if reps < 100:
        raise PairedStatsError("bootstrap reps 至少为 100")
    values = tuple(float(x) for x in differences)
    n = len(values)
    rng = random.Random(seed)
    samples = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(reps)
    ]
    return _percentile(samples, 0.025), _percentile(samples, 0.975)


def make_paired_sample(name_a: str, outcomes_a: Mapping[str, bool],
                       name_b: str, outcomes_b: Mapping[str, bool]) -> PairedSample:
    tasks_a = set(outcomes_a)
    tasks_b = set(outcomes_b)
    if tasks_a != tasks_b:
        missing_a = sorted(tasks_b - tasks_a)
        missing_b = sorted(tasks_a - tasks_b)
        raise PairedStatsError(
            "任务集合不一致，禁止静默取交集："
            f"A 缺 {missing_a[:5]}，B 缺 {missing_b[:5]}")
    if not tasks_a:
        raise PairedStatsError("配对运行没有可判分任务")
    task_ids = tuple(sorted(tasks_a))
    return PairedSample(
        name_a=name_a,
        name_b=name_b,
        task_ids=task_ids,
        outcomes_a=tuple(bool(outcomes_a[t]) for t in task_ids),
        outcomes_b=tuple(bool(outcomes_b[t]) for t in task_ids),
    )


def compare_sample(sample: PairedSample, *,
                   reps: int = DEFAULT_BOOTSTRAP_REPS,
                   seed: int = DEFAULT_SEED) -> PairedComparison:
    pairs = tuple(zip(sample.outcomes_a, sample.outcomes_b))
    both_correct = sum(a and b for a, b in pairs)
    only_a = sum(a and not b for a, b in pairs)
    only_b = sum(not a and b for a, b in pairs)
    both_wrong = sum(not a and not b for a, b in pairs)
    n = len(pairs)
    correct_a = both_correct + only_a
    correct_b = both_correct + only_b
    return PairedComparison(
        name_a=sample.name_a,
        name_b=sample.name_b,
        n=n,
        both_correct=both_correct,
        only_a_correct=only_a,
        only_b_correct=only_b,
        both_wrong=both_wrong,
        accuracy_a=correct_a / n,
        accuracy_b=correct_b / n,
        accuracy_a_ci95=wilson_interval(correct_a, n),
        accuracy_b_ci95=wilson_interval(correct_b, n),
        delta_b_minus_a=(correct_b - correct_a) / n,
        delta_ci95=paired_bootstrap_ci(
            sample.differences, reps=reps, seed=seed),
        mcnemar_p=exact_mcnemar_p(only_a, only_b),
        bootstrap_reps=reps,
        bootstrap_seed=seed,
    )


def compare_outcomes(name_a: str, outcomes_a: Mapping[str, bool],
                     name_b: str, outcomes_b: Mapping[str, bool], *,
                     reps: int = DEFAULT_BOOTSTRAP_REPS,
                     seed: int = DEFAULT_SEED) -> PairedComparison:
    return compare_sample(
        make_paired_sample(name_a, outcomes_a, name_b, outcomes_b),
        reps=reps, seed=seed)


def hierarchical_bootstrap_ci(samples: Sequence[PairedSample], *,
                              reps: int = DEFAULT_BOOTSTRAP_REPS,
                              seed: int = DEFAULT_SEED) -> tuple[float, float]:
    """先抽运行对，再在每个运行对里抽任务，保留两层不确定性。"""
    if not samples:
        raise PairedStatsError("分层 bootstrap 至少需要一个运行对")
    if reps < 100:
        raise PairedStatsError("bootstrap reps 至少为 100")
    task_ids = samples[0].task_ids
    if any(s.task_ids != task_ids for s in samples[1:]):
        raise PairedStatsError("重复运行的任务集合必须完全一致")
    run_n = len(samples)
    task_n = len(task_ids)
    diffs = [s.differences for s in samples]
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(reps):
        selected_runs = [rng.randrange(run_n) for _ in range(run_n)]
        run_means = []
        for run_idx in selected_runs:
            d = diffs[run_idx]
            run_means.append(
                sum(d[rng.randrange(task_n)] for _ in range(task_n)) / task_n)
        draws.append(sum(run_means) / run_n)
    return _percentile(draws, 0.025), _percentile(draws, 0.975)


def compare_repeated(samples: Sequence[PairedSample], *,
                     reps: int = DEFAULT_BOOTSTRAP_REPS,
                     seed: int = DEFAULT_SEED) -> RepeatedComparison:
    if not samples:
        raise PairedStatsError("重复比较至少需要一个运行对")
    task_ids = samples[0].task_ids
    if any(s.task_ids != task_ids for s in samples[1:]):
        raise PairedStatsError("重复运行的任务集合必须完全一致")
    deltas = [sum(s.differences) / len(s.task_ids) for s in samples]
    return RepeatedComparison(
        n_pairs=len(samples),
        n_tasks=len(task_ids),
        mean_delta_b_minus_a=sum(deltas) / len(deltas),
        hierarchical_ci95=hierarchical_bootstrap_ci(
            samples, reps=reps, seed=seed),
        bootstrap_reps=reps,
        bootstrap_seed=seed,
    )


def load_archive_run(path: str | Path, run_id: str, *,
                     metric: str = "attr_exact",
                     scenario: str | None = None) -> RunOutcomes:
    """从归档加载一次运行。每任务必须恰有一个非空判分值。"""
    if metric not in SUPPORTED_METRICS:
        raise PairedStatsError(
            f"metric 只支持 {SUPPORTED_METRICS}，收到 {metric!r}")
    archive_path = Path(path)
    if not archive_path.exists():
        raise PairedStatsError(f"归档不存在：{archive_path}")
    conn = db.connect(archive_path)
    try:
        run = db.q1(conn, "SELECT * FROM runs WHERE run_id=?", (run_id,))
        if run is None:
            raise PairedStatsError(f"归档中没有 run：{run_id}")
        if not run["world_fingerprint"]:
            raise PairedStatsError(f"{run_id} 缺少 world_fingerprint")
        sql = (
            f"SELECT task_id, {metric} AS outcome, gold_codes "
            "FROM trajectories WHERE run_id=? "
            f"AND {metric} IS NOT NULL"
        )
        params: list[str] = [run_id]
        if scenario is not None:
            sql += " AND scenario=?"
            params.append(scenario)
        sql += " ORDER BY task_id, id"
        rows = db.q(conn, sql, params)
    finally:
        conn.close()

    outcomes: dict[str, bool] = {}
    gold: dict[str, str | None] = {}
    for row in rows:
        task_id = row["task_id"]
        if task_id in outcomes:
            raise PairedStatsError(
                f"{run_id} 的任务 {task_id} 有多个判分行，无法确定最终结果")
        outcomes[task_id] = bool(row["outcome"])
        gold[task_id] = row["gold_codes"]
    if not outcomes:
        suffix = f"（scenario={scenario}）" if scenario else ""
        raise PairedStatsError(f"{run_id} 没有可用的 {metric} 判分行{suffix}")
    info = RunInfo(
        run_id=run["run_id"],
        solver=run["solver"],
        model=run["model"],
        world_fingerprint=run["world_fingerprint"],
        code_rev=run["code_rev"] or "",
        config=json.loads(run["config"] or "{}"),
    )
    return RunOutcomes(info=info, outcomes=outcomes, gold_codes=gold)


def make_archive_sample(run_a: RunOutcomes,
                        run_b: RunOutcomes) -> PairedSample:
    if run_a.info.world_fingerprint != run_b.info.world_fingerprint:
        raise PairedStatsError(
            "world_fingerprint 不一致，两个运行不是同一任务世界")
    common = set(run_a.outcomes) & set(run_b.outcomes)
    gold_mismatch = sorted(
        task_id for task_id in common
        if run_a.gold_codes.get(task_id) != run_b.gold_codes.get(task_id))
    if gold_mismatch:
        raise PairedStatsError(
            f"同一任务的 gold 不一致：{gold_mismatch[:5]}")
    return make_paired_sample(
        run_a.info.run_id, run_a.outcomes,
        run_b.info.run_id, run_b.outcomes)


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _ci(interval: tuple[float, float]) -> str:
    return f"[{_pct(interval[0])}, {_pct(interval[1])}]"


def comparison_markdown(comparisons: Sequence[PairedComparison], *,
                        metric: str,
                        repeated: RepeatedComparison | None = None,
                        scenario: str | None = None,
                        heading_level: int = 1) -> str:
    if heading_level < 1:
        raise PairedStatsError("heading_level 必须 >= 1")
    h1 = "#" * heading_level
    h2 = "#" * (heading_level + 1)
    lines = [f"{h1} 配对统计报告", "",
             f"- 指标：`{metric}`",
             f"- 场景：{scenario or '全部任务'}",
             "- 口径：同世界、同任务集合；B - A 为正表示 B 改善",
             "- 区间：accuracy 用 Wilson 95% CI；差值用 paired bootstrap 95% CI",
             "- 检验：二元 exact 指标使用双侧 exact McNemar", ""]
    for i, result in enumerate(comparisons, start=1):
        lines += [
            f"{h2} 运行对 {i}",
            "",
            f"`{result.name_a}` → `{result.name_b}`，{result.n} 条配对任务。",
            "",
            "| | B 正确 | B 错误 |",
            "|---|---:|---:|",
            f"| A 正确 | {result.both_correct} | {result.only_a_correct} |",
            f"| A 错误 | {result.only_b_correct} | {result.both_wrong} |",
            "",
            "| 指标 | 值 |",
            "|---|---:|",
            f"| A accuracy（95% CI） | {_pct(result.accuracy_a)} "
            f"{_ci(result.accuracy_a_ci95)} |",
            f"| B accuracy（95% CI） | {_pct(result.accuracy_b)} "
            f"{_ci(result.accuracy_b_ci95)} |",
            f"| B - A（paired bootstrap 95% CI） | "
            f"{_pct(result.delta_b_minus_a)} {_ci(result.delta_ci95)} |",
            f"| exact McNemar p | {result.mcnemar_p:.6g} |",
            "",
        ]
    if repeated is not None:
        lines += [
            f"{h2} 重复运行汇总",
            "",
            f"- 运行对：{repeated.n_pairs}",
            f"- 每对任务：{repeated.n_tasks}",
            f"- 平均 B - A：{_pct(repeated.mean_delta_b_minus_a)}",
            f"- 分层 paired bootstrap 95% CI：{_ci(repeated.hierarchical_ci95)}",
            f"- bootstrap：{repeated.bootstrap_reps} 次，seed={repeated.bootstrap_seed}",
            "",
            "> 5 次以下只作为管线验证，不作稳定性结论；正式结论要求 5~10 次独立运行。",
            "",
        ]
    lines += [
        f"{h2} 解释边界",
        "",
        "- CI 跨过 0：当前样本不足以支持 B 相对 A 有稳定方向的变化。",
        "- McNemar 只利用两者结论不一致的任务；聚合准确率相同不代表行为相同。",
        "- p 值不代替效应量；必须同时报告差值和 95% CI。",
        "",
    ]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="严格配对的 exact 指标统计")
    parser.add_argument("--archive", type=Path,
                        default=db.PROJECT_ROOT / "data" / "archive.db")
    parser.add_argument("--pair", nargs=2, action="append", required=True,
                        metavar=("RUN_A", "RUN_B"),
                        help="可重复传入；第 k 个 A/B 必须是同编号运行")
    parser.add_argument("--metric", choices=SUPPORTED_METRICS,
                        default="attr_exact")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--bootstrap-reps", type=int,
                        default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    samples: list[PairedSample] = []
    comparisons: list[PairedComparison] = []
    for i, (run_a_id, run_b_id) in enumerate(args.pair):
        run_a = load_archive_run(
            args.archive, run_a_id, metric=args.metric, scenario=args.scenario)
        run_b = load_archive_run(
            args.archive, run_b_id, metric=args.metric, scenario=args.scenario)
        sample = make_archive_sample(run_a, run_b)
        samples.append(sample)
        comparisons.append(compare_sample(
            sample, reps=args.bootstrap_reps, seed=args.seed + i))
    repeated = (compare_repeated(
        samples, reps=args.bootstrap_reps, seed=args.seed)
        if len(samples) > 1 else None)
    text = comparison_markdown(
        comparisons, metric=args.metric,
        repeated=repeated, scenario=args.scenario)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"已写入 {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
