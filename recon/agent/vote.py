"""自一致性投票 —— 直接打翻转率。

阶段 3 量出来的主要问题不是准确率，是**判定不稳定**：
pass^1 52.2%，pass^3 只有 21.7%，58.3% 的任务判定会在多次运行之间翻转。
再去追单次准确率的几个点没有意义 —— 得先让它稳下来。

做法：同一条差错独立跑 k 次，对结论投票。代价是 k 倍 token。

两个设计取舍：

1. **只对差错码投票，处置动作由投出的码推导，不独立投票。**
   动作是码的确定性函数（见 policies/diff_sop.md），让模型对它投票等于
   给它一个自相矛盾的机会 —— 比如投出 D21 却投出 REVERSAL。
   确定性的部分交给代码，这条线和工具层里「模型不做算术」是同一条。

2. **投不出任何码就转人工。** k 次运行完全不重合说明证据本身有歧义，
   这种情况下"少数派意见"不该被采纳。
"""
from __future__ import annotations

from collections import Counter

from ..eval.solution import UNKNOWN, Solution
from ..world.injector import CODES, ESCALATE

_SEVERITY = {"closed": 0, "held": 1, "escalated": 2}


def majority_vote(sols: list[Solution], *, threshold: int | None = None) -> Solution:
    """把 k 个独立结论投成一个。"""
    assert sols, "至少要一个结论"
    k = len(sols)
    need = threshold if threshold is not None else (k // 2 + 1)

    # 逐码投票（每次运行内部去重，避免同一次里重复列同一码被多算）
    counts = Counter(c for s in sols for c in dict.fromkeys(s.root_causes))
    voted = [c for c, n in counts.most_common() if n >= need and c != UNKNOWN]

    # 一致度：投出的码在多少比例的运行里出现
    agreement = (sum(counts[c] for c in voted) / (len(voted) * k)) if voted else 0.0

    if not voted:
        # k 次没有任何一个码达到多数 —— 证据本身有歧义，转人工
        return _merge_cost(Solution(
            task_id=sols[0].task_id, root_causes=[UNKNOWN], actions=[ESCALATE],
            expected_status="escalated", confidence=0.15,
            notes=(f"{k} 次独立运行未就任何原因达成多数（"
                   + "；".join(f"{c}×{n}" for c, n in counts.most_common(6))
                   + "），证据存在歧义，转人工。")), sols)

    # 动作由投出的码推导 —— 不让模型对它投票
    actions: list[str] = []
    for c in voted:
        a = CODES[c].action
        if a and a not in actions:
            actions.append(a)
    if not actions:
        actions = [ESCALATE]
    status = max((CODES[c].expected_status for c in voted),
                 key=lambda s: _SEVERITY.get(s, 0))

    # 少数派意见如实记下来，别丢 —— bad case 分析要用
    minority = [(c, n) for c, n in counts.most_common() if n < need and c != UNKNOWN]
    note = (f"{k} 次独立运行投票：" + "；".join(f"{c}×{counts[c]}" for c in voted))
    if minority:
        note += "；未达多数：" + "、".join(f"{c}×{n}" for c, n in minority[:6])

    refs: list[str] = []
    for s in sols:
        for r in s.evidence_refs:
            if r not in refs:
                refs.append(r)

    return _merge_cost(Solution(
        task_id=sols[0].task_id, root_causes=voted, actions=actions,
        expected_status=status,
        confidence=round(min(agreement, 1.0), 3),
        notes=note, evidence_refs=refs[:40]), sols)


def _merge_cost(out: Solution, sols: list[Solution]) -> Solution:
    """过程成本是累加的 —— 投票不是免费的，代价必须如实进报表。"""
    out.reads = sum(s.reads for s in sols)
    out.rows_read = sum(s.rows_read for s in sols)
    out.chars_read = sum(s.chars_read for s in sols)
    out.steps = sum(s.steps for s in sols)
    out.tokens_in = sum(s.tokens_in for s in sols)
    out.tokens_out = sum(s.tokens_out for s in sols)
    out.cached_in = sum(s.cached_in for s in sols)
    out.cost_micro_cny = sum(s.cost_micro_cny for s in sols)
    out.latency_ms = max((s.latency_ms for s in sols), default=0)   # 可并发，取最慢
    return out


def vote_batches(runs: list[dict[str, Solution]], *,
                 group: int = 3) -> list[dict[str, Solution]]:
    """把 n 次独立运行切成 n//group 组，每组投一次票。

    这样一批运行能同时给出：
      - 单解法的 pass^1 / pass^n
      - 投票解法的 pass^(n//group)
    不用为投票解法另跑一遍，省一大半 token。
    """
    n = len(runs) // group
    out: list[dict[str, Solution]] = []
    for g in range(n):
        chunk = runs[g * group:(g + 1) * group]
        tasks = set(chunk[0])
        for c in chunk[1:]:
            tasks &= set(c)
        out.append({t: majority_vote([c[t] for c in chunk]) for t in sorted(tasks)})
    return out
