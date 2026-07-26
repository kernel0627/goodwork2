"""公告分类器 + typed 闸门的测试。全部离线。

这一层的危险方向和闸门相反：闸门多路由只是白花钱，而分类器**漏标一条覆盖性
公告，那条路径上的差错就再也到不了复核器**，规则会直接去冲正不该动的账。
所以测试的重点不是准确率，是「失败时会不会静悄悄地把结论变成假的」。
"""
from __future__ import annotations

from recon import db
from recon.agent import notice_classifier as nc
from recon.agent.llm import LLMError
from recon.baseline.rules import RuleBaseline
from recon.eval.evidence import EvidenceView
from recon.eval.tasks import load_tasks
from recon.router import route_reason
from recon.world.injector import TEXT_DEPENDENT_CODES
from recon.world.notices import DELAY_TITLES, FEE_TITLES, SCOPED_TITLES

# 部分时段公告也属于延迟覆盖类（只是只覆盖窗内交易）。
# 假分类器必须认识它，否则 typed 闸门会把窗内的 D21 当干扰漏放掉。
DELAY_LIKE = DELAY_TITLES | SCOPED_TITLES


class _Boom:
    name = "boom"

    def complete_json(self, messages, **kw):
        raise LLMError("模拟欠费 402")


def _oracle(conn) -> dict[str, str]:
    """按标题给出真标签。只有测试能这么做 —— 求解方不许看标题走捷径。"""
    out = {}
    for r in db.q(conn, "SELECT id, title FROM channel_notices"):
        out[r["id"]] = (nc.DELAY if r["title"] in DELAY_LIKE
                        else nc.FEE if r["title"] in FEE_TITLES else nc.NONE)
    return out


class _Oracle:
    """按真标签作答的假模型，用来测「分类正确时」这条路径。"""
    name = "oracle"

    def __init__(self, conn):
        self.by_title = {}
        for r in db.q(conn, "SELECT title FROM channel_notices"):
            t = r["title"]
            self.by_title[t] = (nc.DELAY if t in DELAY_LIKE
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
    """全部调用失败时，必须从三个地方都能看出来，不能伪装成正常分类结果。

    这不是假想：账户欠费时真的发生过 —— 当时兜底标签是 `delay`，
    `by_label` 看起来完全正常（就是一堆 delay），统计出来「危险方向漏标 0 条」，
    不拿 oracle 对一遍根本看不出问题。

    现在兜底标签改成了 `unknown_covering`，它**自己就会暴露**：
      1. by_label 里出现一个明显不是正常标签的值；
      2. fallbacks 计数等于公告总数；
      3. trustworthy 为 False。
    三道都要守住 —— 只靠 by_label 的话，哪天兜底标签又被改回 delay 就失效了。
    """
    labels = nc.classify_all(world, _Boom(), cache_path=tmp_path / "c.json")
    s = nc.label_stats(labels)
    assert s["fallbacks"] == s["notices"], "兜底计数必须等于公告总数"
    assert s["trustworthy"] is False, "全失败时不能报 trustworthy"
    assert set(s["by_label"]) == {nc.UNKNOWN_COVERING}, (
        "兜底标签必须是自暴露的 unknown_covering，"
        "不能是 delay/fee 这种看起来像正常分类结果的值")


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

    # 必须毒化一条**真的承载着 D22 任务**的公告。
    # 原来是「取第一条 FEE 公告」，但有些费率公告所在的 (渠道,日期) 上并没有
    # D22 任务，毒化它自然不会造成漏放 —— 那样这条测试就形同虚设。
    d22_dates = {(t.channel_id, t.bill_date) for t in load_tasks(world)
                 if "D22" in t.gold_codes}
    target = None
    for nid, v in labels.items():
        if v["label"] != nc.FEE:
            continue
        row = db.q1(world, "SELECT channel_id, effective_from FROM channel_notices WHERE id=?",
                    (nid,))
        if row and (row["channel_id"], row["effective_from"]) in d22_dates:
            target = nid
            break
    assert target, "没有任何费率公告承载 D22 任务，世界构造有问题"
    labels[target]["label"] = nc.NONE          # 只毒化这一条
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


def test_fee_notice_classification_failure_still_routes_d22(world, tmp_path):
    """⭐ 组合级安全：一条**费率**公告分类失败，D22 仍然必须被路由。

    这是一个单模块看起来 fail-safe、组合起来却不安全的典型：
      旧实现兜底回落到 delay
        -> covering_dates 只把它放进 delay 集合
        -> typed 闸门里 D05 查的是 fee 集合，查不到
        -> 该差错不进复核器
        -> 规则直接执行 REVERSAL = 错误动账

    注释里写着「失败一律当覆盖性」，但 delay 只对 D01 是覆盖性，对 D05 不是。
    现在兜底标签同时进两个集合，宁可多路由（白花钱）也不能漏路由（动错账）。
    """
    labels = nc.classify_all(world, _Oracle(world), use_cache=False)

    # 把一条承载 D22 的费率公告改成兜底状态
    d22_dates = {(t.channel_id, t.bill_date) for t in load_tasks(world)
                 if "D22" in t.gold_codes}
    target = None
    for nid, v in labels.items():
        if v["label"] != nc.FEE:
            continue
        row = db.q1(world, "SELECT channel_id, effective_from FROM channel_notices "
                           "WHERE id=?", (nid,))
        if row and (row["channel_id"], row["effective_from"]) in d22_dates:
            target = nid
            break
    assert target, "没有承载 D22 的费率公告，世界构造有问题"
    labels[target]["label"] = nc.UNKNOWN_COVERING
    labels[target]["fallback"] = True

    covering = nc.covering_dates(world, labels)
    ev, rules = EvidenceView(world), RuleBaseline()
    missed = [t for t in load_tasks(world)
              if "D22" in t.gold_codes
              and not route_reason(rules.solve(t, ev), t, ev, covering)[0]]
    assert not missed, (
        f"费率公告分类失败后有 {len(missed)} 条 D22 被漏路由 —— "
        f"它们会被规则直接 REVERSAL，这就是错误动账")


def test_unknown_covering_lands_in_both_label_sets(world):
    """兜底标签必须同时出现在 delay 与 fee 两个集合里。"""
    labels = nc.classify_all(world, _Oracle(world), use_cache=False)
    nid = next(iter(labels))
    row = db.q1(world, "SELECT channel_id, effective_from FROM channel_notices WHERE id=?",
                (nid,))
    labels[nid]["label"] = nc.UNKNOWN_COVERING
    cov = nc.covering_dates(world, labels)
    pair = (row["channel_id"], row["effective_from"])
    assert pair in cov[nc.DELAY] and pair in cov[nc.FEE]


def test_cache_key_includes_schema_and_model_version(world, tmp_path):
    """缓存 key 必须带版本：只用 body hash 的话，改了分类标准或换了模型之后
    旧缓存照旧命中，你会拿旧标准的标签当成新标准的结果。"""
    import os
    row = db.q1(world, "SELECT * FROM channel_notices LIMIT 1")
    k1 = nc._key(row)
    old = nc.SCHEMA_VERSION
    try:
        nc.SCHEMA_VERSION = old + "-bumped"
        assert nc._key(row) != k1, "改了 SCHEMA_VERSION 后 key 必须变"
    finally:
        nc.SCHEMA_VERSION = old
    old_model = os.environ.get("RECON_LLM_MODEL")
    os.environ["RECON_LLM_MODEL"] = "some-other-model"
    try:
        assert nc._key(row) != k1, "换了模型后 key 必须变"
    finally:
        if old_model is None:
            os.environ.pop("RECON_LLM_MODEL", None)
        else:
            os.environ["RECON_LLM_MODEL"] = old_model
