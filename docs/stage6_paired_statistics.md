# 阶段 6.2：严格配对统计协议

## 当前状态

统计代码和性质测试已经完成。真实归档目前只有 **1 个 run / 336 条轨迹**，
还不满足 5～10 次独立运行的最低要求，因此当前没有正式 p 值或置信区间结论。

这里宁可明确写“样本未齐”，也不拿一次运行伪装成统计显著性。

## 为什么原来的口径不够

原来的“噪声下限”使用 3 次运行的极差除以 2，再用 3 倍阈值判断消融差异。
它可以提醒自己不要追逐几个百分点的波动，但存在三个问题：

1. 极差对运行次数敏感；
2. 只比较聚合百分比，没有利用同一任务上的成对翻转；
3. 二元 exact 指标没有正式假设检验和 95% CI。

阶段 6.2 改为对第 $k$ 次 A/B 运行做严格配对：

```text
第 1 次：A1 ↔ B1，同一世界、同一批任务
第 2 次：A2 ↔ B2，同一世界、同一批任务
……
第 k 次：Ak ↔ Bk
```

任何一对出现世界指纹不同、任务集合不同、同一任务答案不同或重复判分行，统计器都会
直接拒绝；不会静默取交集。

## 报告内容

每个运行对同时报告：

- A 与 B 的 accuracy，以及各自 Wilson 95% CI；
- 四格配对表：两者都对、仅 A 对、仅 B 对、两者都错；
- $B-A$ 的 paired bootstrap 95% CI；
- 二元 exact 指标的双侧 exact McNemar p 值；
- bootstrap 次数和 seed。

有多个同编号运行对时，再做一层分层 paired bootstrap：

1. 先有放回抽取运行对；
2. 再在抽中的运行对内部有放回抽取任务；
3. 对每次重采样计算平均 $B-A$；
4. 取 2.5% 与 97.5% 分位数。

这同时保留运行间波动和任务间波动。5 次以下只验证统计管线，正式稳定性结论要求
5～10 次独立运行。

## 使用方式

单个运行对：

```bash
PAIRED_ARGS='--pair RUN_A RUN_B' make paired-stats
```

5 个同编号运行对：

```bash
python -m recon.eval.paired_stats \
  --pair RUN_A1 RUN_B1 \
  --pair RUN_A2 RUN_B2 \
  --pair RUN_A3 RUN_B3 \
  --pair RUN_A4 RUN_B4 \
  --pair RUN_A5 RUN_B5 \
  --metric attr_exact \
  --out docs/stage6_paired_stats.md
```

可用二元指标为 `attr_exact` 和 `action_exact`。默认执行 20,000 次 bootstrap，
seed 固定为 `20260727`。

## 与 holdout 的关系

- 开发集的 5～10 次重复运行使用 `data/archive.db`，用于调试稳定性和比较配置；
- holdout 仍只运行一次，不进入训练归档；
- holdout 正式报告会在同一次模型运行中直接追加“规则基线 vs 路由复核”的配对四格表、
  paired bootstrap 95% CI 和 exact McNemar；
- holdout 每条任务的规则结果、候选结果、模型原始响应和 gold 会写入独立的
  `data/holdout_v1.results.json`，并由 seal 记录文件指纹；
- 该结果文件只用于审计，不进入 SFT 导出。

## 解释纪律

- 95% CI 跨过 0：当前样本不足以支持稳定方向的变化；
- McNemar 只看两者不一致的任务，适合二元 exact 指标；
- p 值必须与效应量和 95% CI 同时报告；
- 统计显著不等于业务重要，错误动账、越权和漏转人工仍需单独报告条数与金额；
- 不对多个场景反复试验后只挑显著结果。

## 验收

- `tests/test_paired_stats.py`：9 passed
- 配对集合不一致：拒绝
- 世界指纹不一致：拒绝
- exact McNemar 已用已知数值回归
- bootstrap 同 seed 完全可复现
- 清晰改善与完全相同两种边界均有测试
- 全套测试：210 passed, 1 skipped
