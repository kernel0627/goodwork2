"""渠道公告 —— 自由文本证据层。

这一层的存在理由：**让任务集里出现规则做不到、模型能做到的东西。**

阶段 1 跑出的第一版基线是 100% —— 因为我先设计注入器和 SOP、再照着 SOP 写规则，
于是每类差错都有一条机械可查的判据，整个世界对确定性规则完全可解。
那样的话「agent 比规则强」就无从证明。

现实里最大的那块缺口是自由文本：渠道每天都在发延迟下发通知、费率调整通知、
系统维护公告。同一份可见的结构化证据，**在有没有覆盖性公告的情况下，
正确处置是不同的**：

  我方单边 + 无公告            -> D01  向渠道发起查询（CHANNEL_INQUIRY）
  我方单边 + 有延迟下发公告    -> D21  挂起等次日（HOLD_NEXT_BILL），不必去问

  手续费差异 + 无公告          -> D05  冲正我方手续费记账（REVERSAL，动账）
  手续费差异 + 有费率误用公告  -> D22  挂起等渠道次日更正（HOLD_NEXT_BILL，不动账）

第二组尤其关键：规则基线读不到公告，会去**动一笔本不该动的账**，
直接打在「错误动账」这个真实损失指标上。这不是人为制造的噪声，
这是对账工作里天天发生的事。

公告表里**故意没有 kind/type 这类结构化标签**，语义全在中文自由文本里。
一旦给了标签，规则关键词一匹配就绕过去了，整个设计就白做了。
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

from .. import db
from ..config import CHANNELS

# --------------------------------------------------------------------------
# 覆盖性公告（会改变正确处置）
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
# 干扰公告（读起来相关，但不改变任何处置）—— 防止「有公告就挂起」这种偷懒策略
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


# 供答案质量守卫使用：靠标题识别覆盖性公告。
# 求解方不允许用标题走捷径 —— 判据在正文里，而且干扰公告的标题同样"看起来相关"。
DELAY_TITLES = frozenset(t for t, _ in DELAY_NOTICES)
FEE_TITLES = frozenset(t for t, _ in FEE_ERROR_NOTICES)
COVERING_TITLES = DELAY_TITLES | FEE_TITLES


class NoticeBoard:
    """生成公告，并记住哪些 (渠道, 日期) 被覆盖性公告覆盖了。"""

    def __init__(self, conn, seed: int, dates: list[str]):
        self.conn = conn
        self.rng = random.Random(seed + 4242)
        self.dates = dates
        self.delay_cover: set[tuple[str, str]] = set()      # (channel_id, bill_date)
        self.fee_cover: set[tuple[str, str]] = set()
        self.rows: list[dict] = []

    def _next_day(self, d: str) -> str:
        return (datetime.strptime(d, "%Y-%m-%d").date() + timedelta(days=1)).isoformat()

    def _add(self, channel_id: str, date: str, title: str, body: str) -> None:
        nid = f"NT{len(self.rows) + 1:04d}"
        self.rows.append({
            "id": nid,
            "channel_id": channel_id,
            "published_at": f"{self._next_day(date)}T09:30:00",
            "effective_from": date,
            "effective_to": date,
            "title": title,
            "body": body.format(date=date, next_date=self._next_day(date)),
        })

    def build(self) -> dict:
        channels = list(CHANNELS)

        # 覆盖性公告的渠道必须和注入的前置条件对齐，否则会出现「公告发了、
        # 差错一条也注不出来」。曾经把两条费率公告都发给微信，而微信是 net 口径、
        # D22 只在 gross 口径注入，结果 D22 覆盖率为 0。
        delay_eligible = [c for c in channels if c in ("alipay", "wxpay")]   # 取流水量大的
        fee_eligible = [c for c in channels
                        if CHANNELS[c].bill_basis == "gross"]                # D22 的前置条件

        for cover, texts, eligible, k in (
            (self.delay_cover, DELAY_NOTICES, delay_eligible, 2),
            (self.fee_cover, FEE_ERROR_NOTICES, fee_eligible, 2),
        ):
            combos = [(c, d) for c in eligible for d in self.dates]
            picks = self.rng.sample(combos, k=min(k, len(combos)))
            for i, (ch, d) in enumerate(picks):
                title, body = texts[i % len(texts)]
                self._add(ch, d, title, body)
                cover.add((ch, d))

        # 干扰公告：数量明显多于覆盖性公告，逼求解方真的去读内容
        for i in range(len(DISTRACTOR_NOTICES)):
            ch = self.rng.choice(channels)
            d = self.rng.choice(self.dates)
            title, body = DISTRACTOR_NOTICES[i]
            self._add(ch, d, title, body)

        db.insert_many(self.conn, "channel_notices", self.rows)
        self.conn.commit()
        return {
            "notices": len(self.rows),
            "delay_cover": sorted(self.delay_cover),
            "fee_cover": sorted(self.fee_cover),
            "distractors": len(DISTRACTOR_NOTICES),
        }


def build_notices(conn, seed: int, dates: list[str]) -> NoticeBoard:
    board = NoticeBoard(conn, seed, dates)
    board.build()
    return board
