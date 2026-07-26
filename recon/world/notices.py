"""渠道公告 —— 自由文本证据层。

这一层的存在理由：**让任务集里出现规则做不到、模型能做到的东西。**

阶段 1 跑出的第一版基线是 100%，因为我先设计注入器和 SOP、再照着 SOP 写规则，
整个世界对确定性规则完全可解。于是补了公告层：同一份结构化证据，
在有没有覆盖性公告的情况下正确处置不同。

  我方单边 + 无覆盖公告        -> D01  向渠道查询（CHANNEL_INQUIRY）
  我方单边 + 有延迟下发公告    -> D21  挂起等次日（HOLD_NEXT_BILL）

  手续费差异 + 无覆盖公告      -> D05  冲正我方记账（REVERSAL，动账）
  手续费差异 + 有费率误用公告  -> D22  挂起等渠道更正（不动账）

阶段 4 的规则优先路由在这个世界上跑到了 100%（三次），但那说明的是
**天花板被世界难度封顶了**，不是模型能力的上限：当时一条公告要么整天覆盖、
要么完全不覆盖，没有歧义，「当天有覆盖性公告就改判」这种粗策略就够了。

阶段 5 把三类真实存在的难点加进来，它们都会让那种粗策略失效：

1. **部分时段适用**（`SCOPED_DELAY`）
   公告只覆盖当天某个时间窗内的交易。同一个 (渠道, 账单日) 上，
   窗内的差错是 D21、窗外的是 D01 —— **闸门在结构上分不开**，
   必须读懂窗口并比对交易时刻。

2. **近似但不覆盖**（`NEAR_MISS_*`）
   主题看着高度相关（下发时间调整、跨境附加费口径），但正文明确说明
   不影响本类差异、并要求照常走查询/冲正。读一半就会误判。

3. **后续收窄/撤回**（`RETRACTION`）
   当天先发了延迟说明，随后又发一条把范围收窄到别的渠道。
   必须读到第二条并以它为准。

表里**故意没有 kind/type 之类的结构化标签**，语义全在中文正文里。
一旦给了标签，规则关键词一匹配就绕过去了，整个设计就白做。
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

from .. import db
from ..config import CHANNELS

# --------------------------------------------------------------------------
# 一、整天覆盖（阶段 1.5 就有的）
# --------------------------------------------------------------------------

DELAY_NOTICES = [
    ("对账文件下发延迟说明",
     "受我方结算集群扩容影响，{date} 的对账文件在生成过程中出现部分明细缺失，"
     "缺失明细已定位，将随 {next_date} 的对账文件一并补发。"
     "请商户对该日出现的单边情况先行挂账，无需重复发起查询工单，"
     "我方不再对该日缺失明细单独受理申诉。"),
    ("部分交易明细补发通知",
     "因 {date} 凌晨的数据库主从切换，当日部分交易明细未能进入对账文件。"
     "受影响明细我方已完成核对，确认交易本身成功、资金已入账，"
     "仅为账单下发环节遗漏，将在下一期对账文件中补齐。"
     "建议商户按待下发处理，暂不做冲正或补记。"),
]

FEE_ERROR_NOTICES = [
    ("手续费计费异常公告",
     "经自查发现，{date} 我方计费服务在灰度发布期间对部分交易错误套用了"
     "非签约费率档位，导致对账文件中的手续费字段与合同约定不一致。"
     "该问题已修复，差额部分我方将在 {next_date} 的对账文件中以调整明细形式返还。"
     "请商户不要按账单手续费金额调整自身账务，以免二次差错。"),
    ("计费档位误用及更正安排",
     "{date} 部分交易的手续费按错误费率档位计收。经核实，商户侧记账金额是正确的，"
     "问题出在我方账单生成环节。我方将于 {next_date} 主动更正并补齐差额，"
     "无需商户发起冲正流程。"),
]

# --------------------------------------------------------------------------
# 二、⭐ 部分时段适用 —— 同一 (渠道,账单日) 内窗内 D21 / 窗外 D01
# --------------------------------------------------------------------------

SCOPED_DELAY = [
    ("部分时段交易明细缺失说明",
     "{date} {win} 之间发生的交易，因我方账务网关分片异常，"
     "部分明细未能进入当日对账文件，受影响明细将随 {next_date} 的账单补发，"
     "该时段内的单边请先行挂账、无需发起查询工单。"
     "**该时间窗以外的交易不受本次影响**，如出现单边请按正常流程发起渠道查询。"),
    ("账务网关分片异常影响范围",
     "经定位，{date} 的明细缺失集中在 {win} 这一时间窗内，成因为分片路由配置回滚不完整。"
     "窗内受影响明细我方已确认交易成功、将于次日补发，商户按待下发处理即可。"
     "窗外交易的对账文件完整，若有差异属独立问题，仍需照常发起查询。"),
]

# --------------------------------------------------------------------------
# 三、⭐ 近似但不覆盖 —— 主题看着相关，正文明确要求照常走原流程
# --------------------------------------------------------------------------

NEAR_MISS_DELAY = [
    ("对账文件下发时间调整通知",
     "自 {date} 起，本渠道对账文件的下发时间由每日 06:00 调整为 09:00，"
     "以配合上游清算窗口变更。**本次调整仅涉及下发时间，"
     "明细完整性、金额口径与手续费字段均不受影响。**"
     "若当日仍出现我方有、渠道无的单边情况，属独立问题，"
     "请照常发起渠道查询工单，不适用挂账待补处理。"),
    ("历史对账文件重新归档说明",
     "{date} 我方完成了历史对账文件的重新归档，商户后台可下载的文件版本号有所变化。"
     "**归档不改变任何明细内容，也不存在明细缺失或延迟下发。**"
     "当日如有单边差错，与本次归档无关，请按标准流程发起查询。"),
]

NEAR_MISS_FEE = [
    ("跨境交易附加费计费口径说明",
     "自 {date} 起，跨境交易的货币转换附加费由按比例计收改为按笔计收。"
     "**本说明仅涉及跨境附加费，不涉及境内交易的标准费率档位。**"
     "境内交易的手续费如与合同约定不符，属我方或商户侧记账问题，"
     "请照常发起冲正流程处理，不在本次调整范围内。"),
    ("手续费发票开具规则变更",
     "{date} 起手续费发票按月合并开具。**本变更仅涉及发票开具方式，"
     "不改变任何交易的手续费金额与计费口径。**"
     "对账中若发现手续费金额与合同费率复算不一致，仍属记账差错，请照常冲正。"),
]

# --------------------------------------------------------------------------
# 四、⭐ 后续收窄 —— 当天先发延迟说明，再发一条把范围收窄到别的渠道
# --------------------------------------------------------------------------

RETRACTION = [
    ("关于前述明细延迟说明的更正",
     "经进一步复核，{date} 的明细缺失**仅影响 {other} 渠道**，"
     "本渠道当日对账文件明细完整、无遗漏。"
     "此前发布的延迟补发说明**对本渠道不适用**，"
     "本渠道当日如出现单边差错，请按正常流程发起渠道查询工单。"),
]

# --------------------------------------------------------------------------
# 五、纯干扰 —— 读起来相关，但不改变任何处置
# --------------------------------------------------------------------------

DISTRACTOR_NOTICES = [
    ("系统维护通知",
     "我方将于 {date} 02:00-04:00 进行例行系统维护，维护期间支付接口可用性不受影响，"
     "对账文件生成时间可能顺延最多 30 分钟。本次维护不影响明细完整性与金额准确性。"),
    ("商户后台功能升级",
     "{date} 起商户后台对账下载页面改版，新增按商户号批量导出功能。"
     "对账文件的字段结构、金额口径与下发时间均保持不变。"),
    ("风控策略调整说明",
     "自 {date} 起我方对高风险交易的实时拦截阈值有所调整，可能导致部分交易在支付阶段被拒。"
     "被拦截交易不会进入对账文件，也不产生手续费，不属于对账差异。"),
    ("节假日结算安排",
     "{date} 为法定节假日，当日结算款项将顺延至下一工作日到账。"
     "对账文件仍按日正常下发，明细与金额不受影响。"),
    ("接口版本下线提醒",
     "旧版查询接口 v1 将于 {date} 起停止维护，请尽快迁移至 v2。"
     "此变更仅涉及查询接口，不影响对账文件内容。"),
]


# 供答案质量守卫使用：靠标题识别各类公告。
# 求解方不允许用标题走捷径 —— 判据在正文里，而且近似公告的标题同样"看起来相关"。
DELAY_TITLES = frozenset(t for t, _ in DELAY_NOTICES)
FEE_TITLES = frozenset(t for t, _ in FEE_ERROR_NOTICES)
SCOPED_TITLES = frozenset(t for t, _ in SCOPED_DELAY)
NEAR_MISS_DELAY_TITLES = frozenset(t for t, _ in NEAR_MISS_DELAY)
NEAR_MISS_FEE_TITLES = frozenset(t for t, _ in NEAR_MISS_FEE)
RETRACTION_TITLES = frozenset(t for t, _ in RETRACTION)

COVERING_TITLES = DELAY_TITLES | FEE_TITLES | SCOPED_TITLES
NON_COVERING_TITLES = (NEAR_MISS_DELAY_TITLES | NEAR_MISS_FEE_TITLES
                       | RETRACTION_TITLES | frozenset(t for t, _ in DISTRACTOR_NOTICES))

# 部分时段公告用的时间窗（当日分钟数区间）。
# ⚠️ 必须落在**有交易**的时段。第一版取了 02:00-06:00，而生成器把 70% 的订单
#    放在 06:00-22:00，结果「窗内应判 D21」这个最难的场景只生成出 1 条任务 ——
#    设计了却没生成出来，等于没做。
WINDOWS = ((9 * 60, 12 * 60, "09:00-12:00"),
           (14 * 60, 18 * 60, "14:00-18:00"),
           (19 * 60, 22 * 60, "19:00-22:00"))


def minute_of_day(ts: str) -> int:
    d = datetime.fromisoformat(ts)
    return d.hour * 60 + d.minute


class NoticeBoard:
    """生成公告，并记住每个 (渠道, 日期) 上的覆盖情况。

    四个集合的语义必须分清，注入守卫和答案质量守卫都依赖它们：

      delay_cover   整天覆盖，任何单边都是 D21
      fee_cover     整天覆盖，任何手续费差异都是 D22
      scoped_delay  只覆盖某个时间窗 —— 窗内 D21，窗外 D01
      near_miss     有"看着相关但明确不覆盖"的公告 —— 一律仍是 D01/D05
    """

    def __init__(self, conn, seed: int, dates: list[str], cover_k: int = 5,
                 scoped_k: int = 4, near_miss_k: int = 2):
        self.conn = conn
        self.rng = random.Random(seed + 4242)
        self.dates = dates
        self.cover_k = cover_k
        self.scoped_k = scoped_k
        self.near_miss_k = near_miss_k

        self.delay_cover: set[tuple[str, str]] = set()
        self.fee_cover: set[tuple[str, str]] = set()
        # (channel, date) -> (start_min, end_min)
        self.scoped_delay: dict[tuple[str, str], tuple[int, int]] = {}
        self.near_miss_delay: set[tuple[str, str]] = set()
        self.near_miss_fee: set[tuple[str, str]] = set()
        # 有整天延迟公告、但随后被更正公告把本渠道排除掉 —— 语义等同「不覆盖」
        self.retracted: set[tuple[str, str]] = set()
        self.rows: list[dict] = []

    # ------------------------------------------------------------- helpers
    def _next_day(self, d: str) -> str:
        return (datetime.strptime(d, "%Y-%m-%d").date() + timedelta(days=1)).isoformat()

    def _add(self, channel_id: str, date: str, title: str, body: str, **fmt) -> None:
        self.rows.append({
            "id": f"NT{len(self.rows) + 1:04d}",
            "channel_id": channel_id,
            "published_at": f"{self._next_day(date)}T09:30:00",
            "effective_from": date,
            "effective_to": date,
            "title": title,
            "body": body.format(date=date, next_date=self._next_day(date), **fmt),
        })

    def _free_combos(self, eligible: list[str]) -> list[tuple[str, str]]:
        """还没被任何覆盖/近似公告占用的 (渠道, 日期)。

        一个 (渠道,日期) 上只允许一种语义，否则标注会自相矛盾：
        比如同一天既有整天覆盖又有"明确不覆盖"，正确答案就说不清了。
        """
        taken = (self.delay_cover | self.fee_cover | set(self.scoped_delay)
                 | self.near_miss_delay | self.near_miss_fee | self.retracted)
        return [(c, d) for c in eligible for d in self.dates if (c, d) not in taken]

    # --------------------------------------------------------------- build
    def build(self) -> dict:
        channels = list(CHANNELS)
        delay_eligible = [c for c in channels if c in ("alipay", "wxpay")]
        fee_eligible = [c for c in channels if CHANNELS[c].bill_basis == "gross"]

        # 1) 整天覆盖
        for cover, texts, eligible in ((self.delay_cover, DELAY_NOTICES, delay_eligible),
                                       (self.fee_cover, FEE_ERROR_NOTICES, fee_eligible)):
            combos = self._free_combos(eligible)
            for i, (ch, d) in enumerate(self.rng.sample(combos,
                                                        k=min(self.cover_k, len(combos)))):
                title, body = texts[i % len(texts)]
                self._add(ch, d, title, body)
                cover.add((ch, d))

        # 2) 部分时段适用 —— 同一天窗内 D21、窗外 D01
        combos = self._free_combos(delay_eligible)
        for i, (ch, d) in enumerate(self.rng.sample(combos,
                                                    k=min(self.scoped_k, len(combos)))):
            lo, hi, label = WINDOWS[i % len(WINDOWS)]
            title, body = SCOPED_DELAY[i % len(SCOPED_DELAY)]
            self._add(ch, d, title, body, win=label)
            self.scoped_delay[(ch, d)] = (lo, hi)

        # 3) 近似但不覆盖
        for target, texts, eligible in ((self.near_miss_delay, NEAR_MISS_DELAY, delay_eligible),
                                        (self.near_miss_fee, NEAR_MISS_FEE, fee_eligible)):
            combos = self._free_combos(eligible)
            for i, (ch, d) in enumerate(self.rng.sample(combos,
                                                        k=min(self.near_miss_k, len(combos)))):
                title, body = texts[i % len(texts)]
                self._add(ch, d, title, body)
                target.add((ch, d))

        # 4) 后续收窄：从「整天覆盖」里挑几天补一条更正公告，把本渠道排除掉，
        #    并把它们**移出** delay_cover —— 当天读起来有覆盖性公告，
        #    但第二条把本渠道排除了，所以正确答案回到 D01。
        #
        #    ⚠️ 第一版是在 near_miss_delay 的日子上补一条整天公告再补更正，
        #       结果那些日子全被归成「更正收窄」，「近似延迟」这个场景
        #       一条任务都没生成出来。两类难点必须各占自己的日期。
        for ch, d in sorted(self.delay_cover)[:2]:
            other = next((c for c in delay_eligible if c != ch), "wxpay")
            self._add(ch, d, RETRACTION[0][0], RETRACTION[0][1],
                      other=CHANNELS[other].name)
            self.retracted.add((ch, d))
        self.delay_cover -= self.retracted

        # 5) 纯干扰
        for i, (title, body) in enumerate(DISTRACTOR_NOTICES):
            self._add(self.rng.choice(channels), self.rng.choice(self.dates), title, body)

        db.insert_many(self.conn, "channel_notices", self.rows)
        self.conn.commit()
        return {
            "notices": len(self.rows),
            "delay_cover": sorted(self.delay_cover),
            "fee_cover": sorted(self.fee_cover),
            "scoped_delay": {f"{c}/{d}": w for (c, d), w in sorted(self.scoped_delay.items())},
            "near_miss_delay": sorted(self.near_miss_delay),
            "near_miss_fee": sorted(self.near_miss_fee),
            "retracted": sorted(self.retracted),
            "distractors": len(DISTRACTOR_NOTICES),
        }

    # ----------------------------------------------------- 供注入器查询
    def scoped_window(self, channel_id: str, bill_date: str):
        return self.scoped_delay.get((channel_id, bill_date))

    def in_scoped_window(self, channel_id: str, bill_date: str, occurred_at: str) -> bool:
        w = self.scoped_window(channel_id, bill_date)
        if not w:
            return False
        return w[0] <= minute_of_day(occurred_at) < w[1]


def build_notices(conn, seed: int, dates: list[str], cover_k: int = 5) -> NoticeBoard:
    board = NoticeBoard(conn, seed, dates, cover_k=cover_k)
    board.build()
    return board
