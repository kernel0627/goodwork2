-- 交易对账差错处置系统 —— 表结构
-- 金额一律 INTEGER，单位「分」。时间一律 TEXT ISO8601 本地时间。
-- 设计原则：业务对象齐全，但不做复式记账引擎 / 通用工作流引擎。
--          借贷平衡等正确性由 recon/invariants.py 的校验器保证，不由写入路径保证。

PRAGMA foreign_keys = ON;

-- ==========================================================================
-- 主数据
-- ==========================================================================
CREATE TABLE IF NOT EXISTS merchants (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    settle_cycle    TEXT NOT NULL,
    allow_advance   INTEGER NOT NULL DEFAULT 0,
    channels        TEXT NOT NULL              -- json: ["alipay", ...]
);

CREATE TABLE IF NOT EXISTS channels (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    fee_desc        TEXT NOT NULL,             -- 人读的手续费规则说明
    cutoff_minutes  INTEGER NOT NULL,          -- 日切：>= 该分钟数的交易落入次日账单
    bill_basis      TEXT NOT NULL,             -- gross | net
    settle_cycle    TEXT NOT NULL,             -- T+1 | T+3 | weekly
    refund_mode     TEXT NOT NULL,             -- original | balance
    currency        TEXT NOT NULL,
    rounding        TEXT NOT NULL              -- half_up | half_even
);

-- ==========================================================================
-- 我方业务流水
-- ==========================================================================
CREATE TABLE IF NOT EXISTS orders (
    id              TEXT PRIMARY KEY,
    merchant_id     TEXT NOT NULL REFERENCES merchants(id),
    channel_id      TEXT NOT NULL REFERENCES channels(id),
    amount_cents    INTEGER NOT NULL,          -- 交易额（gross 口径）
    currency        TEXT NOT NULL,
    status          TEXT NOT NULL,             -- created|paid|partially_refunded|refunded|failed|closed
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_created ON orders(created_at);
CREATE INDEX IF NOT EXISTS idx_orders_merchant ON orders(merchant_id, created_at);

CREATE TABLE IF NOT EXISTS payments (
    id                TEXT PRIMARY KEY,
    order_id          TEXT NOT NULL REFERENCES orders(id),
    channel_id        TEXT NOT NULL REFERENCES channels(id),
    channel_txn_no    TEXT,                    -- 渠道流水号；失败单可能为空
    amount_cents      INTEGER NOT NULL,        -- 我方记的交易额（gross）
    fee_cents         INTEGER NOT NULL DEFAULT 0,   -- 我方按政策试算的手续费
    status            TEXT NOT NULL,           -- success|failed|pending
    paid_at           TEXT,
    callback_at       TEXT,                    -- 回调到达时间；为空 = 回调丢失
    idempotency_key   TEXT
);
CREATE INDEX IF NOT EXISTS idx_payments_txn ON payments(channel_txn_no);
CREATE INDEX IF NOT EXISTS idx_payments_order ON payments(order_id);
CREATE INDEX IF NOT EXISTS idx_payments_paid ON payments(paid_at);

CREATE TABLE IF NOT EXISTS refunds (
    id                TEXT PRIMARY KEY,
    order_id          TEXT NOT NULL REFERENCES orders(id),
    payment_id        TEXT NOT NULL REFERENCES payments(id),
    channel_txn_no    TEXT,
    amount_cents      INTEGER NOT NULL,
    kind              TEXT NOT NULL,           -- full | partial
    status            TEXT NOT NULL,           -- success|failed|pending
    mode              TEXT NOT NULL,           -- original | balance
    requested_at      TEXT NOT NULL,
    refunded_at       TEXT,
    idempotency_key   TEXT
);
CREATE INDEX IF NOT EXISTS idx_refunds_txn ON refunds(channel_txn_no);
CREATE INDEX IF NOT EXISTS idx_refunds_order ON refunds(order_id);

CREATE TABLE IF NOT EXISTS splits (
    id              TEXT PRIMARY KEY,
    order_id        TEXT NOT NULL REFERENCES orders(id),
    receiver_id     TEXT NOT NULL,
    ratio           TEXT NOT NULL,             -- Decimal 字符串
    amount_cents    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_splits_order ON splits(order_id);

CREATE TABLE IF NOT EXISTS ledger_entries (
    id              TEXT PRIMARY KEY,
    ref_type        TEXT NOT NULL,             -- payment|refund|adjustment|fee|settlement
    ref_id          TEXT NOT NULL,
    account         TEXT NOT NULL,
    direction       TEXT NOT NULL,             -- D 借 | C 贷
    amount_cents    INTEGER NOT NULL,
    occurred_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ledger_ref ON ledger_entries(ref_type, ref_id);

CREATE TABLE IF NOT EXISTS settlements (
    id              TEXT PRIMARY KEY,
    merchant_id     TEXT NOT NULL REFERENCES merchants(id),
    period_start    TEXT NOT NULL,
    period_end      TEXT NOT NULL,
    amount_cents    INTEGER NOT NULL,
    status          TEXT NOT NULL,             -- pending|paid|frozen
    frozen_reason   TEXT,
    created_at      TEXT NOT NULL
);

-- ==========================================================================
-- 渠道账单（外部真源）
-- ==========================================================================
CREATE TABLE IF NOT EXISTS channel_bills (
    id                  TEXT PRIMARY KEY,
    channel_id          TEXT NOT NULL REFERENCES channels(id),
    bill_date           TEXT NOT NULL,
    file_seq            INTEGER NOT NULL DEFAULT 1,   -- >1 表示重复下发
    record_count        INTEGER NOT NULL,
    total_amount_cents  INTEGER NOT NULL,             -- 渠道口径合计
    received_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bills_date ON channel_bills(channel_id, bill_date);

CREATE TABLE IF NOT EXISTS channel_bill_records (
    id              TEXT PRIMARY KEY,
    bill_id         TEXT NOT NULL REFERENCES channel_bills(id),
    channel_id      TEXT NOT NULL REFERENCES channels(id),
    channel_txn_no  TEXT NOT NULL,
    rec_type        TEXT NOT NULL,             -- payment | refund
    amount_cents    INTEGER NOT NULL,          -- ⚠️ 渠道口径（可能是 net）
    fee_cents       INTEGER NOT NULL,
    currency        TEXT NOT NULL,
    occurred_at     TEXT NOT NULL,
    memo            TEXT                       -- ⚠️ 外部可控文本 —— 提示注入的攻击面
);
CREATE INDEX IF NOT EXISTS idx_bill_rec_txn ON channel_bill_records(channel_txn_no);
CREATE INDEX IF NOT EXISTS idx_bill_rec_bill ON channel_bill_records(bill_id);

-- ⭐ 渠道公告 —— 自由文本证据。
-- 这张表是「规则做不到、模型能做到」的分界线所在：
--   语义全部在 body 的中文自由文本里，**故意不设 kind / type 之类的结构化标签**，
--   否则规则引擎关键词一匹配就绕过去了，整个设计就没意义了。
-- 现实里对账员每天都要读这些东西：延迟下发通知、费率调整、系统维护、口径变更。
CREATE TABLE IF NOT EXISTS channel_notices (
    id              TEXT PRIMARY KEY,
    channel_id      TEXT NOT NULL REFERENCES channels(id),
    published_at    TEXT NOT NULL,
    effective_from  TEXT NOT NULL,
    effective_to    TEXT,
    title           TEXT NOT NULL,
    body            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notice_ch ON channel_notices(channel_id, effective_from);

-- ==========================================================================
-- 对账
-- ==========================================================================
CREATE TABLE IF NOT EXISTS recon_tasks (
    id                    TEXT PRIMARY KEY,
    channel_id            TEXT NOT NULL REFERENCES channels(id),
    bill_date             TEXT NOT NULL,
    status                TEXT NOT NULL,       -- running|done
    started_at            TEXT NOT NULL,
    finished_at           TEXT,
    our_total_cents       INTEGER NOT NULL DEFAULT 0,
    channel_total_cents   INTEGER NOT NULL DEFAULT 0,
    matched_count         INTEGER NOT NULL DEFAULT 0,
    diff_count            INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS recon_diffs (
    id                    TEXT PRIMARY KEY,
    recon_task_id         TEXT NOT NULL REFERENCES recon_tasks(id),
    channel_id            TEXT NOT NULL,
    bill_date             TEXT NOT NULL,
    our_ref_type          TEXT,                -- payment | refund | NULL(渠道单边)
    our_ref_id            TEXT,
    channel_record_id     TEXT,                -- NULL 表示我方单边
    channel_txn_no        TEXT,
    -- 检测来源：差错池不只来自流水匹配
    --   match           流水号匹配
    --   rule_scan       业务规则扫描（如累计退款超原额 —— 两侧一致，匹配发现不了）
    --   settlement_scan 结算合规扫描
    -- 只有 source='match' 的差错参与 INV2 全量守恒。
    source                TEXT NOT NULL DEFAULT 'match',
    our_ref_signed        INTEGER NOT NULL DEFAULT 0,   -- 我方带符号贡献（退款为负）
    channel_signed        INTEGER NOT NULL DEFAULT 0,   -- 渠道带符号贡献（已归一 gross）
    our_gross_cents       INTEGER,             -- 我方金额绝对值（给人看）
    channel_gross_cents   INTEGER,             -- 渠道金额绝对值，已按 bill_basis 归一到 gross
    -- ⭐ diff_cents = our_ref_signed - channel_signed，恒等于该笔对全量守恒的贡献。
    --    INV2 直接求和它，不做任何形态判断 —— 这是唯一能保证守恒精确成立的约定。
    diff_cents            INTEGER NOT NULL DEFAULT 0,
    fee_delta_cents       INTEGER NOT NULL DEFAULT 0,   -- 我方手续费 - 渠道手续费
    status                TEXT NOT NULL,       -- new|investigating|attributed|pending_approval|resolving|closed|escalated|held
    created_at            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_diffs_task ON recon_diffs(recon_task_id);
CREATE INDEX IF NOT EXISTS idx_diffs_status ON recon_diffs(status);

-- ⚠️⚠️ 注入日志：差错的答案在注入那一刻就写下来了。agent 与基线绝对不可读。
--       两段式：pre_match 注入在对账前改数据，post_match 注入在对账后直接造差错。
CREATE TABLE IF NOT EXISTS injections (
    id                TEXT PRIMARY KEY,
    code              TEXT NOT NULL,           -- D01..D20
    phase             TEXT NOT NULL,           -- pre_match | post_match
    channel_id        TEXT NOT NULL,
    bill_date         TEXT NOT NULL,
    match_key         TEXT NOT NULL,           -- 用于和对账产出的差错对上：通常是 channel_txn_no
    group_id          TEXT NOT NULL,           -- 同 group_id 的多条注入 = 一个复合差错
    correct_action    TEXT NOT NULL,
    expected_status   TEXT NOT NULL,
    explanation       TEXT NOT NULL,
    injected_ref      TEXT
);
CREATE INDEX IF NOT EXISTS idx_inj_key ON injections(channel_id, bill_date, match_key);
CREATE INDEX IF NOT EXISTS idx_inj_group ON injections(group_id);

-- ⚠️⚠️ ground truth：agent 与基线都绝对不可读。只有判分器可读。
--       由 tests/test_gt_isolation.py 强制守住。
CREATE TABLE IF NOT EXISTS diff_ground_truth (
    diff_id           TEXT PRIMARY KEY REFERENCES recon_diffs(id),
    root_causes       TEXT NOT NULL,           -- json: ["D08","D04"]
    correct_actions   TEXT NOT NULL,           -- json: ["HOLD_NEXT_BILL","AUTO_WRITEOFF"]
    is_composite      INTEGER NOT NULL DEFAULT 0,
    expected_status   TEXT NOT NULL,           -- 处置完成后差错应处的终态
    explanation       TEXT NOT NULL,           -- 为什么是这个答案（供 bad case 分析看）
    injected_ref      TEXT                     -- 注入时动了哪条记录
);

-- ==========================================================================
-- 处置与审批
-- ==========================================================================
CREATE TABLE IF NOT EXISTS adjustments (
    id                TEXT PRIMARY KEY,
    diff_id           TEXT NOT NULL REFERENCES recon_diffs(id),
    action            TEXT NOT NULL,           -- AUTO_WRITEOFF|SUPPLEMENT|REVERSAL|CHANNEL_INQUIRY|HOLD_NEXT_BILL|ESCALATE|DISCARD_DUPLICATE
    amount_cents      INTEGER NOT NULL DEFAULT 0,
    idempotency_key   TEXT NOT NULL UNIQUE,    -- ⭐ 幂等的物理保证
    status            TEXT NOT NULL,           -- proposed|pending_approval|approved|rejected|executed|failed
    executed_count    INTEGER NOT NULL DEFAULT 0,   -- ⭐ 不变量 INV3 检查这一列
    created_by        TEXT NOT NULL,           -- agent | rule_baseline | human:<name>
    created_at        TEXT NOT NULL,
    executed_at       TEXT,
    note              TEXT
);
CREATE INDEX IF NOT EXISTS idx_adj_diff ON adjustments(diff_id);

CREATE TABLE IF NOT EXISTS approvals (
    id                TEXT PRIMARY KEY,
    adjustment_id     TEXT NOT NULL REFERENCES adjustments(id),
    required_role     TEXT NOT NULL,
    status            TEXT NOT NULL,           -- pending|approved|rejected
    decided_by        TEXT,
    decided_at        TEXT,
    reason            TEXT
);
CREATE INDEX IF NOT EXISTS idx_appr_adj ON approvals(adjustment_id);

-- ==========================================================================
-- Agent 运行轨迹（既不是业务数据，也不是答案 —— 是 agent 自己的过程记录）
-- 阶段 2 里 agent 不读它；阶段 4 的 memory 才会用到历史处置。
-- ==========================================================================
CREATE TABLE IF NOT EXISTS agent_runs (
    id                TEXT PRIMARY KEY,
    task_id           TEXT NOT NULL,
    diff_id           TEXT NOT NULL,
    solver            TEXT NOT NULL,
    model             TEXT NOT NULL,
    stop_reason       TEXT NOT NULL,      -- concluded|forced|step_budget|cost_budget|llm_error
    steps             INTEGER NOT NULL,
    reads             INTEGER NOT NULL,
    chars_read        INTEGER NOT NULL,
    tokens_in         INTEGER NOT NULL,
    tokens_out        INTEGER NOT NULL,
    cost_micro_cny    INTEGER NOT NULL,
    latency_ms        INTEGER NOT NULL,
    root_causes       TEXT NOT NULL,
    actions           TEXT NOT NULL,
    expected_status   TEXT NOT NULL,
    confidence        REAL NOT NULL,
    evidence_refs     TEXT,
    notes             TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_task ON agent_runs(task_id);

CREATE TABLE IF NOT EXISTS agent_steps (
    id                TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL REFERENCES agent_runs(id),
    step_no           INTEGER NOT NULL,
    thought           TEXT,
    tool              TEXT,
    arguments         TEXT,
    result_digest     TEXT,
    ok                INTEGER NOT NULL DEFAULT 1,
    tokens_in         INTEGER NOT NULL DEFAULT 0,
    tokens_out        INTEGER NOT NULL DEFAULT 0,
    latency_ms        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_steps_run ON agent_steps(run_id, step_no);
