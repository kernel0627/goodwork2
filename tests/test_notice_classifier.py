"""公告分类器 + typed 闸门的测试。全部离线。

这一层的危险方向和闸门相反：闸门多路由只是白花钱，而分类器**漏标一条覆盖性
公告，那条路径上的差错就再也到不了复核器**，规则会直接去冲正不该动的账。
所以测试的重点不是准确率，是「失败时会不会静悄悄地把结论变成假的」。
"""
from __future__ import annotations

from recon import db
from recon.agent import notice_classifier as nc
from recon.agent.llm import FakeLLM, LLMError
from recon.baseline.rules import RuleBaseline
from recon.eval.evidence import EvidenceView
from recon.eval.tasks import load_tasks
from recon.router import route_reason
from recon.world.injector import TEXT_DEPENDENT_CODES
from recon.world.notices import DELAY_TITLES, FEE_TITLES


class _Boom:
    name = "boom"

    def complete_json(self, messages, **kw):
        raise LLMError("模拟欠费 402")


def _oracle(conn) -> dict[str, str]:
    """按标题给出真标签。只有测试能这么做 —— 求解方不许看标题走捷径。"""
    out = {}
    for r in db.q(conn, "SELECT id, title FROM channel_notices"):
        out[r["id"]] = (nc.DELAY if r["title"] in DELAY_TITLES
                        else nc.FEE if r["title"] in FEE_TITLES else nc.NONE)
    return out


class _Oracle:
    """按真标签作答的假模型，用来测「分类正确时」这条路径。"""
    name = "oracle"

    def __init__(self, conn):
        self.by_title = {}
        for r in db.q(conn, "SELECT title FROM channel_notices"):
            t = r["title"]
            self.by_title[t] = (nc.DELAY if t in DELAY_TITLES
                                else nc.FEE if t in FEE_TITLES else nc.NONE)
        self.calls = 0

    def complete_json(self, messages, **kw):
        self.calls += 1
        text = messages[-1]["content"]
        label = next((v for k, v in self.by_title.items() if k in text), nc.NONE)
        from recon.agent.llm import Usage
        return {"label": label, "reasoning": "oracle"}, Usage(calls=1)


# --------------------------------------------------- 失败不能伪装成结果

def test_total_failure_is_reported_not_disguised(world, tmp_path):
    """13 条全调用失败时，fail-safe 会把它们全标成 delay —— 那是一个
    看起来完全正常的分类结果。必须有一个字段把这件事暴露出来。

    这不是假想：账户欠费时真的发生过，统计出来「危险方向漏标 0 条」，
    不拿 oracle 对一遍根本看不出来。
    """
    labels = nc.classify_all(world, _Boom(), cache_path=tmp_path / "c.json")
    s = nc.label_stats(labels)
    assert s["fallbacks"] == s["notices"]
    assert s["trustworthy"] is False
    # by_label 单看是完全正常的，这正是危险所在
    assert set(s["by_label"]) == {nc.DELAY}


def test_fallback_labels_are_never_cached(world, tmp_path):
    """兜底结果入缓存就是永久污染：下次命中缓存，连『没分类成功』都看不见了。"""
    p = tmp_path / "c.json"
    nc.classify_all(world, _Boom(), cache_path=p)
    assert not p.exists() or nc._load_cache(p) == {}

    # 换成能正常作答的模型，应该全部重新分类而不是命中垃圾缓存
    labels = nc.classify_all(world, _Oracle(world), cache_path=p)
    assert nc.label_stats(labels)["fallbacks"] == 0
    assert nc._load_cache(p), "成功的分类应该入缓存"


def test_cache_makes_the_second_run_free(world, tmp_path):
    p = tmp_path / "c.json"
    llm = _Oracle(world)
    nc.classify_all(world, llm, cache_path=p)
    first = llm.calls
    assert first > 0

    labels = nc.classify_all(world, llm, cache_path=p)
    assert llm.calls == first, "第二次不该再调模型"
    assert nc.label_stats(labels)["calls"] == 0


def test_unknown_label_falls_back_to_covering(world):
    class Weird:
        name = "weird"

        def complete_json(self, messages, **kw):
            from recon.agent.llm import Usage
            return {"label": "maybe?"}, Usage(calls=1)

    row = db.q(world, "SELECT * FROM channel_notices LIMIT 1")[0]
    label, why, ok = nc.classify_one(Weird(), row)
    assert label in nc.COVERING and ok is False and "未知标签" in why


# ------------------------------------------------------------ typed 闸门

def test_typed_gate_keeps_full_recall_when_classification_is_right(world):
    """分类正确时，typed 闸门必须仍然一条需读文本任务都不漏，
    而且路由比例要明显低于 any 闸门 —— 否则这一层不值得做。"""
    labels = nc.classify_all(world, _Oracle(world), use_cache=False)
    covering = nc.covering_dates(world, labels)

    ev = EvidenceView(world)
    rules = RuleBaseline()
    n_any = n_typed = missed = 0
    tasks = load_tasks(world)
    for t in tasks:
        sol = rules.solve(t, ev)
        a = route_reason(sol, t, ev)[0]
        d = route_reason(sol, t, ev, covering)[0]
        n_any += a
        n_typed += d
        if not d and set(t.gold_codes) & TEXT_DEPENDENT_CODES:
            missed += 1

    assert missed == 0, f"typed 闸门漏放了 {missed} 条需读文本任务"
    assert n_typed < n_any, f"typed({n_typed}) 没比 any({n_any}) 少，白做"


def test_typed_gate_leaks_when_a_covering_notice_is_mislabelled(world):
    """这条测试记录的是**风险**而不是能力：把覆盖性公告误标成 none，
    typed 闸门就会漏放，规则会去冲正不该动的账。

    any 闸门没有这个风险 —— 这是 typed 用成本换来的代价，必须被看见。
    """
    labels = nc.classify_all(world, _Oracle(world), use_cache=False)
    for nid, v in labels.items():
        if v["label"] == nc.FEE:
            v["label"] = nc.NONE          # 只毒化一条
            break
    covering = nc.covering_dates(world, labels)

    ev = EvidenceView(world)
    rules = RuleBaseline()
    missed = sum(1 for t in load_tasks(world)
                 if not route_reason(rules.solve(t, ev), t, ev, covering)[0]
                 and "D22" in t.gold_codes)
    assert missed > 0, "误标一条覆盖性公告却没造成漏放，说明这条测试没测到东西"


def test_covering_dates_expands_the_effective_range(world):
    labels = nc.classify_all(world, _Oracle(world), use_cache=False)
    cov = nc.covering_dates(world, labels)
    assert set(cov) == {nc.DELAY, nc.FEE}
    for pairs in cov.values():
        for ch, day in pairs:
            assert isinstance(ch, str) and len(day) == 10


def test_oracle_labels_match_the_world(world):
    """守卫：如果以后新增了覆盖性公告类型而分类器的标签集没跟上，这里会失败。"""
    got = set(_oracle(world).values())
    assert got <= {nc.DELAY, nc.FEE, nc.NONE}
    assert {nc.DELAY, nc.FEE} <= got, "世界里应当同时存在两类覆盖性公告"
