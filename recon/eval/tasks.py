"""任务集 —— 把差错池导出成可判分的任务。

一个任务 = 一条差错 + 一个受控证据入口。**任务里不预先塞证据**：
取哪些证据、取多少，本身就是求解方的能力，也是阶段 3 context 工程要优化的东西。

一个细节：D08/D09/D14 这类「移位」差错会同时产出两条差错记录
（原账单日的我方单边 + 新账单日的渠道单边），它们是同一个逻辑问题的两面。
两条都保留成独立任务（对账员看到的确实是两条），但记下 group_key，
报表同时给「按差错」和「按逻辑问题去重」两套数字，避免指标被重复计数放大。
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import db


@dataclass(frozen=True)
class Task:
    task_id: str
    diff_id: str
    channel_id: str
    bill_date: str
    source: str            # match | rule_scan | settlement_scan
    group_key: str         # 同一逻辑问题的多条差错共享它

    # ⭐ 决策时刻。求解方只能看到 published_at <= as_of 的公告。
    #    ⚠️ 不设它就会出现时间穿越：对账任务在账单日次日 07:00 跑，
    #       而公告 09:30 才发布，可求解方却能读到 —— 那是未来信息泄漏，
    #       测出来的指标不符合在线决策流程。
    as_of: str = ""

    # 答案：只有判分器能拿，求解方看不到
    gold_codes: tuple[str, ...] = ()
    gold_actions: tuple[str, ...] = ()
    gold_status: str = "closed"
    is_composite: bool = False
    gold_explanation: str = ""
    diff_cents: int = 0
    fee_delta_cents: int = 0

    @property
    def at_risk_cents(self) -> int:
        """一旦处置错误，会被错动的金额。

        ⚠️ 手续费维度差错的 diff_cents 恒为 0（gross 两侧相等），
           金额度量必须回退到 fee_delta_cents，否则「错误动账 30 条 / 0.00 元」
           这种自相矛盾的数字就会出现，而这几条恰恰是最该被看见的。
        """
        return abs(self.diff_cents) or abs(self.fee_delta_cents)

    @property
    def substantive_codes(self) -> tuple[str, ...]:
        """D19（提示注入）是修饰器，不是实质原因。"""
        return tuple(c for c in self.gold_codes if c != "D19")

    @property
    def has_injection(self) -> bool:
        return "D19" in self.gold_codes


def load_tasks(conn, *, limit: int | None = None,
               only_codes: set[str] | None = None,
               only_composite: bool = False,
               split: str = "all") -> list[Task]:
    """split:
         all   全部
         text  判据只在自由文本里的（D21/D22）—— agent 唯一的价值点
         rule  判据在结构化数据里的

    ⭐ 为什么要分档跑：把 text 档当混合样本的一个切片，它的样本量就被采样配额
       压到十几条，一条翻转 5~7 个百分点，什么效应都分辨不出来。
       该档必须作为**独立实验**跑全量。
    """
    from ..world.injector import TEXT_DEPENDENT_CODES
    sql = """
        SELECT d.id, d.channel_id, d.bill_date, d.source, d.channel_txn_no,
               d.our_ref_type, d.our_ref_id, d.diff_cents, d.fee_delta_cents,
               d.created_at,
               g.root_causes, g.correct_actions, g.expected_status,
               g.is_composite, g.explanation
        FROM recon_diffs d JOIN diff_ground_truth g ON g.diff_id = d.id
        ORDER BY d.id
    """
    tasks: list[Task] = []
    for r in db.q(conn, sql):
        codes = tuple(db.jload(r["root_causes"]) or ())
        if only_composite and not r["is_composite"]:
            continue
        if only_codes and not (set(codes) & only_codes):
            continue
        is_text = bool(set(codes) & TEXT_DEPENDENT_CODES)
        if split == "text" and not is_text:
            continue
        if split == "rule" and is_text:
            continue
        group = (f"{r['channel_id']}:{r['channel_txn_no']}" if r["channel_txn_no"]
                 else f"{r['our_ref_type']}:{r['our_ref_id']}")
        tasks.append(Task(
            task_id=f"T{r['id']}",
            diff_id=r["id"],
            channel_id=r["channel_id"],
            bill_date=r["bill_date"],
            source=r["source"],
            group_key=group,
            as_of=r["created_at"],          # 差错被发现的时刻 = 决策时刻
            gold_codes=codes,
            gold_actions=tuple(db.jload(r["correct_actions"]) or ()),
            gold_status=r["expected_status"],
            is_composite=bool(r["is_composite"]),
            gold_explanation=r["explanation"],
            diff_cents=r["diff_cents"] or 0,
            fee_delta_cents=r["fee_delta_cents"] or 0,
        ))
        if limit and len(tasks) >= limit:
            break
    return tasks


def task_summary(tasks: list[Task]) -> dict:
    codes: dict[str, int] = {}
    for t in tasks:
        for c in t.gold_codes:
            codes[c] = codes.get(c, 0) + 1
    return {
        "tasks": len(tasks),
        "logical_issues": len({t.group_key for t in tasks}),
        "composite": sum(1 for t in tasks if t.is_composite),
        "with_injection": sum(1 for t in tasks if t.has_injection),
        "by_source": {s: sum(1 for t in tasks if t.source == s)
                      for s in sorted({t.source for t in tasks})},
        "by_code": dict(sorted(codes.items())),
    }


def default_ensure(n: int) -> dict[str, int]:
    """默认保底：需读自由文本的两类 + 含提示注入的任务，各留够样本量。

    D19（提示注入）是修饰器、不在 substantive_codes 里，但注入抵抗率是要报的指标，
    样本里一条都没有的话那个指标就是空的 —— 所以要显式保底。
    """
    from ..world.injector import TEXT_DEPENDENT_CODES
    k = max(6, n // 10)
    out = {c: k for c in sorted(TEXT_DEPENDENT_CODES)}
    out["D19"] = k
    return out


def sample_tasks(tasks: list[Task], n: int, *,
                 ensure: dict[str, int] | None = None) -> list[Task]:
    """分层采样 —— 跑 agent 时不可能每次全量（要花钱），但抽样必须公平。

    做法：按「主要原因」分组，组内按 task_id 稳定排序，然后在组间轮转取，
    直到取满 n 个。轮转天然保证小类（D06/D12/D13 这些只有几条的）不会被
    大类淹没，同时大类的占比又不至于失真。

    确定性：只依赖 task_id 排序和轮转顺序，不用随机数，所以同样的
    (任务集, n) 一定给出同样的子集 —— agent 和基线才能在**同一批任务**上比。

    `ensure={"D21": 10, ...}` 给某些差错码设保底条数，先填满再轮转。
    需读自由文本的两类只占全量的 11%，纯轮转下 40 条样本里只剩 2 条，
    而那恰恰是整个对比的主战场。**定向过采样会改变整体配比，
    所以报表必须看分组指标（规则可解 / 需读文本），不能只看总体数字。**
    """
    ensure = dict(ensure or {})
    # 保底名额最多占一半，否则小样本下（比如 n=12）保底就把整个样本吃光了
    if ensure and sum(ensure.values()) > n // 2:
        share = max(1, (n // 2) // len(ensure))
        ensure = {c: min(k, share) for c, k in ensure.items()}

    picked: list[Task] = []
    taken: set[str] = set()

    for code, k in sorted(ensure.items()):
        pool = sorted((t for t in tasks
                       if code in t.gold_codes and t.task_id not in taken),
                      key=lambda t: t.task_id)
        for t in pool[:k]:
            picked.append(t)
            taken.add(t.task_id)

    tasks = [t for t in tasks if t.task_id not in taken]
    groups: dict[str, list[Task]] = {}
    for t in tasks:
        key = ",".join(sorted(t.substantive_codes)) or "NONE"
        groups.setdefault(key, []).append(t)
    for g in groups.values():
        g.sort(key=lambda t: t.task_id)

    order = sorted(groups)                      # 组名字典序，稳定
    idx = {k: 0 for k in order}
    while len(picked) < n:
        moved = False
        for k in order:
            if len(picked) >= n:
                break
            i = idx[k]
            if i < len(groups[k]):
                picked.append(groups[k][i])
                idx[k] = i + 1
                moved = True
        if not moved:
            break
    picked.sort(key=lambda t: t.task_id)
    return picked
