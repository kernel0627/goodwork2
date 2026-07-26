"""公告分类器 —— 把「读懂自由文本」从 per-task 提到 per-notice。

## 为什么值得单独做一层

公告全库 13 条，差错 505 条。闸门 2 路由 110 条，每条都要把当日公告重读一遍，
同一份公告被读上百遍。但**一条公告是不是覆盖性的，和具体哪条差错无关** ——
分类一次，全局复用。

分类结果落盘缓存（按正文哈希），所以重复跑的边际成本是 0。

## ⚠️ 这一层的误判方向是危险的

闸门 2 的两个方向是不对称但安全的：多路由只是白花钱。加了分类之后不一样了——
**把一条覆盖性公告误标成干扰，那条路径上的差错就再也到不了复核器**，
规则会直接去冲正一笔不该动的账。这是有资金损失的。

所以分类器按 fail-safe 设计：

1. 提示词明确要求「拿不准就归为覆盖性」；
2. 解析失败、调用失败、返回了未知标签 —— 一律当作覆盖性；
3. `classify_all` 报出每一条的标签，让使用者能核对，而不是黑箱。

换句话说这一层只做**负向筛除**：只有在确信一条公告不改变任何处置时，
才允许它把差错挡在复核之外。
"""
from __future__ import annotations

import hashlib
import os
import json
from pathlib import Path

from .. import db
from .llm import LLMClient, LLMError

# 标签。COVERING_* 是会改变处置的两类；NONE 是不改变处置的干扰公告。
DELAY = "delay"          # 明细未下发、次日补发   -> 可把 D01 改判为 D21
FEE = "fee"              # 渠道误用费率、将自行更正 -> 可把 D05 改判为 D22
NONE = "none"            # 不改变任何处置

# ⭐ 兜底标签：分类失败或返回未知标签时用它。
#    ⚠️ 原实现兜底回落到 DELAY，注释写着「一律当覆盖性」—— **那是错的**：
#       delay 只对 D01 是覆盖性，对 D05 不是。而 typed 闸门按类型分集合查
#       （D01 查 delay、D05 查 fee），所以一条**费率**公告分类失败回落成 delay 后：
#           D05 去查 fee 集合 -> 查不到 -> 不进复核 -> 规则直接 REVERSAL
#       结果是「分类失败会造成错误动账」。单模块看起来 fail-safe，组合起来不安全。
#    正确做法：兜底标签同时进 delay 和 fee 两个集合，宁可多路由（只是白花钱）。
UNKNOWN_COVERING = "unknown_covering"

COVERING = (DELAY, FEE, UNKNOWN_COVERING)

# 分类标准的版本号。改了标签定义或提示词就要 +1，否则旧缓存会被误用。
SCHEMA_VERSION = "v2-unknown-covering"

CACHE_PATH = db.PROJECT_ROOT / "data" / "notice_labels.json"

SYSTEM = f"""你是支付清结算团队的对账人员。给你一条渠道公告，判断它属于哪一类。

# 三个标签

- `{DELAY}`：说明**当日部分交易明细未进入对账文件**，将随次日/后续账单补发。
  这类公告会把「我方单边、渠道账单缺失该笔」的处置从「发查询工单」改成「挂起等补发」。
- `{FEE}`：说明**渠道侧误用了费率档位**、商户侧记账正确、渠道将自行更正差额。
  这类公告会把「手续费差异」的处置从「冲正我方记账」改成「挂起等渠道更正」。
- `{NONE}`：不改变任何差错的处置。系统维护、后台功能升级、风控策略调整、
  节假日结算顺延、接口版本下线等等 —— 这些读起来"相关"，但都不改变账务处置。

# ⚠️ 拿不准就选覆盖性的那一类

误判的代价严重不对称：

- 把覆盖性公告错标成 `{NONE}`：相关差错会被直接按无公告处置，
  可能去**冲正一笔本不该动的账**，有实际资金损失。
- 把干扰公告错标成 `{DELAY}`/`{FEE}`：只是让下游多复核几条，白花一点钱。

所以只有在你**确信**这条公告不涉及「明细未下发」也不涉及「渠道费率错误」时，
才选 `{NONE}`。任何犹豫都选覆盖性的那一类。

# 输出

只输出一个 JSON 对象：

{{"label": "{DELAY} 或 {FEE} 或 {NONE}", "reasoning": "一句话理由"}}
"""


def _key(row, model_identity: str | None = None) -> str:
    # ⚠️ key 必须带版本：只用 body hash 的话，改了分类标准或换了模型之后
    #    旧缓存照旧命中，你会拿着旧标准的标签当成新标准的结果。
    payload = "|".join([
        row["body"], SCHEMA_VERSION,
        hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest()[:12],
        model_identity or "|".join([
            os.environ.get("RECON_LLM_PROVIDER", "auto"),
            os.environ.get("RECON_LLM_BASE_URL",
                           os.environ.get("OPENAI_BASE_URL",
                                          os.environ.get("DEEPSEEK_BASE_URL", ""))),
            os.environ.get("RECON_LLM_MODEL",
                           os.environ.get("OPENAI_MODEL",
                                          os.environ.get(
                                              "RECON_AGENT_MODEL",
                                              "unconfigured"))),
        ]),
    ])
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{row['id']}:{h}"


def _load_cache(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def classify_one(llm: LLMClient, row) -> tuple[str, str, bool]:
    """返回 (标签, 理由, 是否真的分类成功)。

    第三个返回值不能省。fail-safe 会把失败变成一个**看起来正常的标签**，
    如果不把它和真结果区分开，全盘失败就会伪装成一个合理的分类结果 ——
    这真的发生过：13 条公告因为账户欠费全部调用失败，全部回落成 delay，
    统计出来是「危险方向漏标 0 条」，看着完全正常。
    """
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content":
             f"渠道 {row['channel_id']}，生效 {row['effective_from']}~{row['effective_to']}\n"
             f"标题：{row['title']}\n\n正文：\n{row['body']}"}]
    try:
        data, _ = llm.complete_json(msgs, max_tokens=800)
    except LLMError as e:
        return (UNKNOWN_COVERING,
                f"分类调用失败，按 fail-safe 同时当作 delay 与 fee 覆盖：{e}", False)

    label = str(data.get("label") or "").strip().lower()
    if label not in (DELAY, FEE, NONE):
        return (UNKNOWN_COVERING,
                f"返回了未知标签 {label!r}，按 fail-safe 同时当作两类覆盖", False)
    return label, str(data.get("reasoning") or "")[:300], True


def classify_all(conn, llm: LLMClient, *, use_cache: bool = True,
                 cache_path: Path | None = None) -> dict[str, dict]:
    """给全部公告打标签。返回 {notice_id: {"label":…, "reasoning":…, "cached":bool}}。

    调用次数 = 未命中缓存的公告数，与差错条数无关 —— 这正是这一层的意义。
    """
    path = cache_path or CACHE_PATH
    cache = _load_cache(path) if use_cache else {}
    rows = db.q(conn, "SELECT * FROM channel_notices ORDER BY id")

    out: dict[str, dict] = {}
    dirty = False
    for row in rows:
        k = _key(row, getattr(llm, "identity", getattr(llm, "name", None)))
        if use_cache and k in cache:
            out[row["id"]] = {**cache[k], "cached": True, "fallback": False}
            continue
        label, why, ok = classify_one(llm, row)
        entry = {"label": label, "reasoning": why, "title": row["title"]}
        # ⚠️ 兜底结果绝不入缓存。写进去就成了永久污染：下次跑命中缓存，
        #    连「这条其实没分类成功」都看不见了。
        if ok:
            cache[k] = entry
            dirty = True
        out[row["id"]] = {**entry, "cached": False, "fallback": not ok}

    if use_cache and dirty:
        _save_cache(path, cache)
    return out


def covering_dates(conn, labels: dict[str, dict]) -> dict[str, set[tuple[str, str]]]:
    """把公告标签折成 {标签: {(渠道, 账单日), …}}，闸门直接查这个集合。

    一条公告的生效区间可能跨天，所以按天展开。
    """
    from datetime import date, timedelta

    out: dict[str, set[tuple[str, str]]] = {DELAY: set(), FEE: set()}
    for row in db.q(conn, "SELECT * FROM channel_notices ORDER BY id"):
        lab = (labels.get(row["id"]) or {}).get("label")
        if lab not in COVERING:
            continue
        # 兜底标签进两个集合 —— 不知道它是哪一类，就两类都当覆盖，
        # 代价是多路由几条（白花钱），而漏路由的代价是错误动账。
        targets = (DELAY, FEE) if lab == UNKNOWN_COVERING else (lab,)
        d0 = date.fromisoformat(row["effective_from"])
        d1 = date.fromisoformat(row["effective_to"] or row["effective_from"])
        while d0 <= d1:
            for t in targets:
                out[t].add((row["channel_id"], d0.isoformat()))
            d0 += timedelta(days=1)
    return out


def label_stats(labels: dict[str, dict]) -> dict:
    counts: dict[str, int] = {}
    for v in labels.values():
        counts[v["label"]] = counts.get(v["label"], 0) + 1
    fallbacks = sum(1 for v in labels.values() if v.get("fallback"))
    return {
        "notices": len(labels),
        "calls": sum(1 for v in labels.values() if not v.get("cached")),
        "by_label": dict(sorted(counts.items())),
        # 兜底数必须单独报出来，不能混进 by_label —— 见 classify_one 的说明
        "fallbacks": fallbacks,
        "trustworthy": fallbacks == 0,
    }


__all__ = ["DELAY", "FEE", "NONE", "COVERING", "classify_all", "classify_one",
           "covering_dates", "label_stats", "CACHE_PATH"]
