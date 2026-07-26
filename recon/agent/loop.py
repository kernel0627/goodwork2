"""Harness —— agent 的运行循环。

这一层负责「模型不该负责」的所有事情：

  停止条件      步数上限、成本上限、无进展检测
  输出校验      非法 JSON 重灌、幻觉工具名回灌、结论不合法给一次修复机会
  兜底          轮数用完强制收敛；实在认不出来必须转人工，不许自动关闭差错
  轨迹          每一步的 thought / 工具 / 参数 / 结果摘要 / token / 耗时全量落库

「模型只生成状态转移，不生成结论文字」这一刀就划在这里。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..eval.evidence import EvidenceView
from ..eval.solution import UNKNOWN, Solution
from ..eval.tasks import Task
from ..world.injector import ALL_ACTIONS, CODES
from . import prompts
from .config import AgentConfig, V1
from .llm import LLMClient, LLMError, LLMFatalError, Usage
from .tools import ToolBox, digest

VALID_STATUS = ("closed", "held", "escalated")


@dataclass
class Step:
    step_no: int
    thought: str = ""
    tool: str | None = None
    arguments: dict | None = None
    result_digest: str = ""
    ok: bool = True
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0


@dataclass
class RunResult:
    task_id: str
    diff_id: str
    solution: Solution
    steps: list[Step] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "concluded"
    model: str = ""


class AgentRunner:
    def __init__(self, llm: LLMClient, *, max_steps: int | None = None,
                 max_cost_micro_cny: int | None = None,
                 no_progress_limit: int = 2, cfg: AgentConfig | None = None):
        self.llm = llm
        self.cfg = cfg or V1
        self.max_steps = max_steps if max_steps is not None else self.cfg.max_steps
        self.max_cost = max_cost_micro_cny
        self.no_progress_limit = no_progress_limit

    # ------------------------------------------------------------------
    def run(self, task: Task, ev: EvidenceView) -> RunResult:
        ev.reset_trace()
        box = ToolBox(ev, strip_injection_policy=self.cfg.strip_injection_policy)
        t_start = time.time()

        messages: list[dict] = [
            {"role": "system",
             "content": prompts.system_prompt(box.catalog(), self.max_steps, self.cfg)},
            {"role": "user",
             "content": prompts.task_prompt(task.diff_id, task.channel_id, task.bill_date)},
        ]

        steps: list[Step] = []
        usage = Usage()
        seen_calls: dict[tuple[str, str], int] = {}
        stop_reason = "concluded"
        conclusion: dict | None = None

        for i in range(1, self.max_steps + 1):
            try:
                decision, u = self.llm.complete_json(messages)
            except LLMFatalError:
                # 余额/密钥问题不是「这条任务失败」，是整批都跑不了。
                # 吞成 UNKNOWN 会让报表看起来像模型能力差，必须让它炸出来。
                raise
            except LLMError as e:
                stop_reason = "llm_error"
                steps.append(Step(i, thought=f"模型调用失败：{e}", ok=False))
                break
            usage.add(u)

            thought = str(decision.get("thought", ""))[:800]
            action = decision.get("next_action") or {}
            atype = str(action.get("type", "")).upper()

            if atype == "CONCLUDE":
                conclusion = decision.get("conclusion") or {}
                steps.append(Step(i, thought=thought, tool=None,
                                  tokens_in=u.tokens_in, tokens_out=u.tokens_out,
                                  latency_ms=u.latency_ms))
                break

            if atype != "CALL_TOOL":
                messages += [
                    {"role": "assistant", "content": _dump(decision)},
                    {"role": "user", "content":
                        'next_action.type 必须是 "CALL_TOOL" 或 "CONCLUDE"。请重新输出。'},
                ]
                steps.append(Step(i, thought=thought, ok=False,
                                  result_digest="非法 next_action.type",
                                  tokens_in=u.tokens_in, tokens_out=u.tokens_out,
                                  latency_ms=u.latency_ms))
                continue

            name = str(action.get("tool", ""))
            args = action.get("arguments") or {}
            key = (name, _dump(args))
            seen_calls[key] = seen_calls.get(key, 0) + 1
            repeat = seen_calls[key]

            result = box.call(name, args)
            d = digest(result)
            steps.append(Step(i, thought=thought, tool=name, arguments=args,
                              result_digest=d, ok=bool(result.get("ok")),
                              tokens_in=u.tokens_in, tokens_out=u.tokens_out,
                              latency_ms=u.latency_ms))

            note = ""
            if repeat >= self.no_progress_limit:
                note = ("\n⚠️ 这是你第 %d 次用完全相同的参数调用 %s，不会有新信息。"
                        "换个工具或换参数，或者直接给结论。" % (repeat, name))
            messages += [
                {"role": "assistant", "content": _dump(decision)},
                {"role": "user", "content": f"工具 {name} 返回：\n{_dump(result)}{note}"},
            ]

            if self.max_cost and usage.cost_micro_cny >= self.max_cost:
                stop_reason = "cost_budget"
                break
        else:
            stop_reason = "step_budget"

        if conclusion is None:
            conclusion, u2, extra = self._force_conclude(messages)
            usage.add(u2)
            if extra:
                steps.append(extra)
            if stop_reason == "concluded":
                stop_reason = "forced"

        sol = self._to_solution(task, conclusion, ev, usage, len(steps))
        sol.latency_ms = int((time.time() - t_start) * 1000)
        return RunResult(task_id=task.task_id, diff_id=task.diff_id, solution=sol,
                         steps=steps, usage=usage, stop_reason=stop_reason,
                         model=getattr(self.llm, "name", "unknown"))

    # ------------------------------------------------------------------
    def _force_conclude(self, messages: list[dict]) -> tuple[dict, Usage, Step | None]:
        msgs = messages + [{"role": "user", "content": prompts.FORCE_CONCLUDE}]
        try:
            decision, u = self.llm.complete_json(msgs)
        except LLMFatalError:
            raise
        except LLMError as e:
            return ({}, Usage(), Step(0, thought=f"强制收敛失败：{e}", ok=False))
        step = Step(0, thought=str(decision.get("thought", ""))[:400],
                    result_digest="forced conclude",
                    tokens_in=u.tokens_in, tokens_out=u.tokens_out,
                    latency_ms=u.latency_ms)
        return (decision.get("conclusion") or {}), u, step

    # ------------------------------------------------------------------
    def _to_solution(self, task: Task, c: dict, ev: EvidenceView,
                     usage: Usage, steps: int) -> Solution:
        codes = [str(x).strip().upper() for x in (c.get("root_causes") or [])
                 if str(x).strip()]
        actions = [str(x).strip().upper() for x in (c.get("actions") or [])
                   if str(x).strip()]

        # 词表外的一律丢弃，不让模型自创差错码 / 动作
        codes = [x for x in dict.fromkeys(codes) if x in CODES or x == UNKNOWN]
        actions = [x for x in dict.fromkeys(actions) if x in ALL_ACTIONS]
        status = str(c.get("expected_status", "")).strip().lower()
        if status not in VALID_STATUS:
            status = "escalated"

        # 认不出来必须转人工，不许自动关闭差错
        if not codes:
            codes = [UNKNOWN]
        if UNKNOWN in codes or not actions:
            if "ESCALATE" not in actions:
                actions.append("ESCALATE")
            status = "escalated"

        try:
            conf = float(c.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = min(max(conf, 0.0), 1.0)

        refs = [str(x) for x in (c.get("evidence_refs") or [])][:30]
        return Solution(
            task_id=task.task_id, root_causes=codes, actions=actions,
            expected_status=status, confidence=conf,
            notes=str(c.get("reasoning", ""))[:1500],
            evidence_refs=refs,
            reads=ev.reads, rows_read=ev.rows_read, chars_read=ev.chars_read,
            steps=steps, tokens_in=usage.tokens_in, tokens_out=usage.tokens_out,
            cached_in=usage.cached_in,
            cost_micro_cny=usage.cost_micro_cny)


def _dump(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, default=str)
