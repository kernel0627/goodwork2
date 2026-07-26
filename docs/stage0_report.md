# 阶段 0 验收报告

- 差错总数：**500**
- 带答案的差错：**500**
- 复合差错：**95**
- 无答案差错：0

## 22 类差错覆盖

| 码 | 名称 | 处置 | 命中差错数 |
|---|---|---|---:|
| D01 | 我方单边：渠道账单缺失该笔 | CHANNEL_INQUIRY | 29 |
| D02 | 回调丢失：我方未记成功 | SUPPLEMENT | 18 |
| D03 | 舍入模式差异 | AUTO_WRITEOFF | 38 |
| D04 | 账单口径差异（net vs gross） | AUTO_WRITEOFF | 44 |
| D05 | 手续费规则不一致 | REVERSAL | 88 |
| D06 | 状态不符 | CHANNEL_INQUIRY | 3 |
| D07 | 重复支付 | REVERSAL | 8 |
| D08 | 时序穿越：退款先于支付入账 | HOLD_NEXT_BILL | 18 |
| D09 | 跨日归属（渠道侧移位） | HOLD_NEXT_BILL | 86 |
| D10 | 部分退款累计超原单 | ESCALATE | 8 |
| D11 | 渠道明细重复下发 | DISCARD_DUPLICATE | 52 |
| D12 | 币种/汇率错配 | ESCALATE | 7 |
| D13 | 分账比例错误 | REVERSAL | 4 |
| D14 | 回调延迟（我方侧移位） | HOLD_NEXT_BILL | 78 |
| D15 | 余额退款不进渠道账单 | AUTO_WRITEOFF | 7 |
| D16 | 有未平差错却已结算 | ESCALATE | 6 |
| D17 | 渠道流水号复用 | ESCALATE | 6 |
| D18 | 退款符号错误 | REVERSAL | 11 |
| D19 | 备注字段含提示注入 | （同底层） | 38 |
| D20 | 容差内无解释噪声 | AUTO_WRITEOFF | 28 |
| D21 | 延迟下发（公告已说明） | HOLD_NEXT_BILL | 26 |
| D22 | 渠道费率误用（公告承诺次日更正） | HOLD_NEXT_BILL | 30 |

覆盖情况：**22/22 全覆盖**

## 差错机械形态分布

| 形态 | 数量 |
|---|---:|
| OUR_ONLY | 167 |
| CHANNEL_ONLY | 120 |
| OTHER | 108 |
| AMOUNT | 105 |

## 正确处置动作分布

| 动作 | 数量 |
|---|---:|
| HOLD_NEXT_BILL | 238 |
| AUTO_WRITEOFF | 117 |
| REVERSAL | 111 |
| DISCARD_DUPLICATE | 52 |
| CHANNEL_INQUIRY | 32 |
| ESCALATE | 27 |
| SUPPLEMENT | 18 |

## 期望终态分布

| 终态 | 数量 |
|---|---:|
| held | 270 |
| closed | 203 |
| escalated | 27 |
