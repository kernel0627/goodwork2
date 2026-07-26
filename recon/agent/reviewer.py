"""公告复核器 —— 单次调用，只回答「当日公告是否推翻规则的结论」。

## 为什么是单次调用，不是 agent 循环

阶段 4 的闸门已经把「要不要读公告」这个判断做完了（纯结构化，零 token）。
落到模型头上的只剩一个问题：**这几条公告里，有没有一条真的覆盖当前这条差错。**

判这个问题需要的证据是**闭合的**：当日该渠道的公告全文 + SOP 里 D21/D22 的识别
依据。两样都能一次性取齐，没有「取了才知道下一步取什么」的探索需求。既然路径
是确定的，就不该让模型来决定路径 —— 那是 workflow 该干的事，不是 agent。

代价差别很实在：完整 agent 每条 6~8 轮调用，复核一条 1 轮。而且阶段 3 测出
pass^3 只有 21.7%，多轮累积正是不稳定的来源，砍掉轮数直接打在这个指标上。

## 为什么它比「让 agent 重解」安全

复核器的输出空间被**结构性地限制**成两个：维持规则结论，或把
D01→D21 / D05→D22。它**不能凭空造出一个新归因**。

这一点专治闸门的误触。阶段 4 实测被路由的 110 条里有 51 条是误触 —— 规则本来
全对，交给 agent 从零重做时它有 ~50% 的概率做坏。换成复核，这 51 条最坏也只是
被错误地改判成 D21/D22，而不是变成任意一个错误归因；而且它们本来就没有覆盖性
公告，模型读完公告正文应当维持原判。

## 它读得到的东西

只有：规则的结论、当日公告全文、SOP 的 D21/D22 一节。**拿不到答案，也拿不到
公告的类型标签**（公告表故意没有 kind 字段，干扰公告比覆盖性公告还多）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from ..eval.evidence import EvidenceView
from ..eval.solution import Solution
from ..eval.tasks import Task
from ..world.injector import CODES, HOLD_NEXT_BILL
from .llm import LLMClient, LLMError

# 复核只允许做这一种改判 —— 与 router.TEXT_OVERRIDABLE 同源，见 diff_sop.md
OVERRIDE = {"D01": "D21", "D05": "D22"}

DECISION_SCHEMA = """{
  "covered": true 或 false,
  "notice_id": "起决定作用的公告 id；covered=false 时填 null",
  "override": "D21 或 D22；covered=false 时填 null",
  "reasoning": "为什么这条公告覆盖/不覆盖当前差错，一两句话"
}"""

SYSTEM = """你是支付清结算团队的对账复核员。规则引擎已经给出了一条差错的归因，
你**只做一件事**：判断当日该渠道的公告有没有推翻这个归因。

# 判定依据（来自差错处置 SOP）

结构化证据完全相同的两组：

| 结构化证据 | 无覆盖性公告 | 有覆盖性公告 |
|---|---|---|
| 我方单边、渠道明细缺失 | D01 → CHANNEL_INQUIRY | D21 → HOLD_NEXT_BILL |
| 手续费维度差异 | D05 → REVERSAL（动账） | D22 → HOLD_NEXT_BILL（不动账） |

**覆盖性公告的判据在正文里，不在标题里。**

- 能改判成 D21 的公告，正文必须说明：当日部分明细未进入对账文件、将随次日账单补发。
- 能改判成 D22 的公告，正文必须说明：是渠道侧误用费率、商户记账正确、渠道将自行更正。

# ⚠️ 当天往往还有不改变任何处置的公告

系统维护、后台升级、风控调整、节假日结算安排、接口下线 —— 这些读起来都"相关"，
但**都不改变处置**。不能因为"当天有公告"就改判。看不出正文明确覆盖当前差错的，
一律 covered=false。

# 两个方向的代价不对称

- 该改判而没改（漏判 D22）：会去冲正一笔本不该动的账，公告已明确要求商户不要
  自行调整，这是有实际资金损失的错误。
- 不该改判却改了（误判成 D22）：把一笔本该冲正的差错挂起，次日发现渠道并未更正，
  损失是延迟一天。

所以证据确凿时要敢改判；但"确凿"指的是**正文明确说了上面那两件事之一**，
不是"感觉相关"。

# 输出

只输出一个 JSON 对象：

""" + DECISION_SCHEMA


def _fee_block(facts: dict) -> str:
    """手续费三方对照。

    刻意只摆事实、不加措辞 —— 上一版试过在提示词里加「必须核对前提条件」那类
    强约束，结果把模型整体推保守：误改判少了，但漏判了 7 条真 D22，
    错误动账从 0 涨到 7 条。见 docs/stage4_router_design.md。
    """
    f = facts.get("fee")
    if not f:
        return ""
    def mark(ok):
        return "一致" if ok else ("不一致" if ok is False else "无数据")
    ch_fee = f.get("channel_fee_cents")
    return f"""
# 手续费三方对照（规则复算，供你核对）

| | 金额 | 与合同费率 |
|---|---:|---|
| 我方记账手续费 | {f['our_fee_cents']} 分 | {mark(f.get('our_matches_standard'))} |
| 按合同费率复算 | {f['standard_fee_cents']} 分 | — |
| 渠道账单手续费 | {ch_fee if ch_fee is not None else '（无）'} 分 | {mark(f.get('channel_matches_standard'))} |
"""


def _task_prompt(task: Task, rule_sol: Solution, notices: list[dict]) -> str:
    codes = "、".join(rule_sol.root_causes) or "（无）"
    acts = "、".join(rule_sol.actions) or "（无）"
    body = "\n\n".join(
        f"【公告 {n['id']}】{n['title']}\n"
        f"渠道 {n['channel_id']}，生效 {n['effective_from']}~{n['effective_to']}，"
        f"发布 {n['published_at']}\n{n['body']}"
        for n in notices)
    return f"""# 待复核的差错

差错 {task.diff_id}，渠道 {task.channel_id}，账单日 {task.bill_date}

规则引擎的结论：**{codes}**，拟执行动作 {acts}
规则的依据：{rule_sol.notes or "（未记录）"}
{_fee_block(rule_sol.facts)}
# 该渠道该账单日的全部公告（共 {len(notices)} 条）

{body or "（无）"}

# 你的任务

判断上面这些公告里，有没有一条**明确覆盖当前这条差错**，从而应当把结论
改判为 D21 或 D22。没有就维持规则的结论。
"""


@dataclass
class ReviewResult:
    task_id: str
    overridden: bool
    notice_id: str | None
    reasoning: str
    tokens_in: int = 0
    tokens_out: int = 0
    cached_in: int = 0
    cost_micro_cny: int = 0
    latency_ms: int = 0
    error: str = ""


class NoticeReviewer:
    """单次调用的复核器。给 RouterSolver 当 inner 用。"""

    wants_prior = True          # 告诉 RouterSolver 把规则结论传进来

    def __init__(self, llm: LLMClient, *, name: str | None = None):
        self.llm = llm
        self.name = name or f"reviewer:{getattr(llm, 'name', 'unknown')}"
        self.results: list[ReviewResult] = []

    # ------------------------------------------------------------------
    def solve(self, task: Task, ev: EvidenceView, *, prior: Solution) -> Solution:
        before = (ev.reads, ev.rows_read, ev.chars_read)
        notices = ev.channel_notices(task.channel_id, task.bill_date)
        read_cost = (ev.reads - before[0], ev.rows_read - before[1],
                     ev.chars_read - before[2])
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": _task_prompt(task, prior, notices)}]

        t0 = time.time()
        try:
            data, usage = self.llm.complete_json(messages, max_tokens=1500)
        except LLMError as e:
            # 复核失败就维持规则结论。这是安全方向：规则至少是确定性的，
            # 而且失败时改判等于凭空动账。
            self.results.append(ReviewResult(
                task.task_id, False, None, f"复核调用失败：{e}",
                latency_ms=int((time.time() - t0) * 1000), error=str(e)))
            return prior

        covered = bool(data.get("covered"))
        target = data.get("override") if covered else None
        reasoning = str(data.get("reasoning") or "")[:500]

        out = prior
        applied = False
        if covered and target in OVERRIDE.values():
            out = _apply_override(prior, target)
            applied = out is not prior

        self.results.append(ReviewResult(
            task.task_id, applied, data.get("notice_id") if applied else None,
            reasoning, tokens_in=usage.tokens_in, tokens_out=usage.tokens_out,
            cached_in=usage.cached_in, cost_micro_cny=usage.cost_micro_cny,
            latency_ms=usage.latency_ms))

        # 成本记在返回的 Solution 上 —— 复核也是要花钱的，不计就是偷成本
        out.tokens_in += usage.tokens_in
        out.tokens_out += usage.tokens_out
        out.cached_in += usage.cached_in
        out.cost_micro_cny += usage.cost_micro_cny
        out.latency_ms += usage.latency_ms
        out.steps += 1
        out.reads += read_cost[0]           # 读公告那一趟，按实际计
        out.rows_read += read_cost[1]
        out.chars_read += read_cost[2]
        if applied:
            out.notes = f"{out.notes}；复核改判：{reasoning}"
        return out


def _apply_override(prior: Solution, target: str) -> Solution:
    """把 D01→D21 或 D05→D22 应用到规则结论上。

    只替换对应的那一个编码，其它编码原样保留 —— 复合差错（比如 D05,D11）
    里被公告覆盖的只是手续费那一维，重复下发那一维不受影响。
    """
    src = next((s for s, t in OVERRIDE.items() if t == target), None)
    if src is None or src not in prior.root_causes:
        return prior                        # 规则没给出这个编码，无从改判

    codes = [target if c == src else c for c in prior.root_causes]

    # 动作按编码表重算，而不是在原动作上打补丁 ——
    # D05 的 REVERSAL 必须消失，留着就还是会去动账。
    actions: list[str] = []
    for c in codes:
        a = CODES[c].action if c in CODES else None
        if a and a not in actions:
            actions.append(a)

    sev = {"closed": 0, "held": 1, "escalated": 2}
    status = max((CODES[c].expected_status for c in codes if c in CODES),
                 key=lambda s: sev.get(s, 0), default="held")

    return Solution(
        task_id=prior.task_id, root_causes=codes, actions=actions,
        expected_status=status, confidence=prior.confidence,
        notes=prior.notes, evidence_refs=list(prior.evidence_refs),
        reads=prior.reads, rows_read=prior.rows_read, chars_read=prior.chars_read,
        steps=prior.steps, tokens_in=prior.tokens_in, tokens_out=prior.tokens_out,
        cached_in=prior.cached_in, cost_micro_cny=prior.cost_micro_cny,
        latency_ms=prior.latency_ms)


def run_review(db_path, tasks: list[Task], priors: dict[str, Solution], *,
               llm: LLMClient, workers: int = 12,
               ) -> tuple[dict[str, Solution], list[ReviewResult]]:
    """并发复核。每个 worker 自己开只读连接 —— sqlite 连接不能跨线程共享。

    复核器本身是无状态的，但 self.results 会被多线程 append，所以每个 worker
    用自己的实例，最后把结果收拢。
    """
    from concurrent.futures import ThreadPoolExecutor

    from .. import db

    def work(task: Task) -> tuple[str, Solution, ReviewResult]:
        # 每条任务开关一次连接。相对一次模型调用（秒级）这点开销可以忽略，
        # 换来的是不用管线程局部状态的生命周期。
        rv = NoticeReviewer(llm)
        conn = db.connect(db_path)
        try:
            sol = rv.solve(task, EvidenceView(conn), prior=priors[task.task_id])
        finally:
            conn.close()
        return task.task_id, sol, rv.results[-1]

    sols: dict[str, Solution] = {}
    results: list[ReviewResult] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for tid, sol, res in pool.map(work, tasks):
            sols[tid] = sol
            results.append(res)
    results.sort(key=lambda r: r.task_id)
    return sols, results


def review_stats(results: list[ReviewResult]) -> dict:
    n = len(results)
    return {
        "reviewed": n,
        "overridden": sum(r.overridden for r in results),
        "kept": sum(not r.overridden for r in results),
        "errors": sum(bool(r.error) for r in results),
        "avg_tokens_in": round(sum(r.tokens_in for r in results) / n, 1) if n else 0,
        "avg_tokens_out": round(sum(r.tokens_out for r in results) / n, 1) if n else 0,
        "cached_rate": (sum(r.cached_in for r in results) /
                        max(1, sum(r.tokens_in for r in results))),
    }


__all__ = ["NoticeReviewer", "ReviewResult", "review_stats", "OVERRIDE"]
