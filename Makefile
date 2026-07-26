# 解释器可覆盖：优先用环境里的 PY，其次找 conda 的 agent 环境，最后退回 python3。
# ⚠️ 原来硬编码成作者本机的绝对路径，别人 clone 下来一条命令都跑不了。
PY ?= $(shell command -v python >/dev/null 2>&1 && echo python || echo python3)
ifneq ($(wildcard $(HOME)/miniforge3/envs/agent/bin/python),)
PY := $(HOME)/miniforge3/envs/agent/bin/python
endif

SEED   ?= 42
START  ?= 2026-07-01
DAYS   ?= 7
ORDERS ?= 200
INJECT ?= 120

.PHONY: help build test check stats report case case-composite repro clean all \
        tasks baseline agent replay variance ablate route scenarios \
        archive-stats archive-export holdout-build holdout-check holdout-eval \
        paired-stats llm-config

help:
	@echo "make build            生成完整业务世界 + 带答案的差错池（一条命令）"
	@echo "make test             跑测试（含答案质量守卫）"
	@echo "make check            只跑不变量校验"
	@echo "make stats            差错池统计与 20 类覆盖"
	@echo "make report           统计并写入 docs/stage0_report.md"
	@echo "make case             随机看一条差错的完整证据面（带答案）"
	@echo "make case-composite   看一条复合差错"
	@echo "make tasks            任务集概览"
	@echo "make baseline         跑纯规则基线并判分（阶段 1 核心动作）"
	@echo "make agent            跑 Agent 并与规则基线同表对比（阶段 2 核心动作）"
	@echo "make replay           回放一条判错的 agent 轨迹"
	@echo "make llm-config       安全显示最终模型配置（不联网、不显示完整密钥）"
	@echo "make route            规则优先路由 + 单次复核（核心交付物）"
	@echo "make route-dry        只跑闸门，看要花多少钱"
	@echo "make variance         同配置重复跑，量方差与 pass^k"
	@echo "make ablate           消融阶梯"
	@echo "make repro            验证同 seed 可复现"
	@echo "make holdout-build    构建并封存阶段 6 holdout（不调用模型）"
	@echo "make holdout-check    核验 holdout 未被改动及一次性评测状态"
	@echo "make holdout-eval     一次性正式评测（需 CONFIRM_HOLDOUT=yes）"
	@echo "make paired-stats     对归档 run 做严格配对统计（需 PAIRED_ARGS）"
	@echo "make all              build + test + report"
	@echo ""
	@echo "参数：SEED=$(SEED) START=$(START) DAYS=$(DAYS) ORDERS=$(ORDERS) INJECT=$(INJECT)"

build:
	$(PY) -m recon.cli build --seed $(SEED) --start $(START) --days $(DAYS) \
		--orders-per-day $(ORDERS) --inject-per-day $(INJECT)

llm-config:
	$(PY) -m recon.cli llm-config

test:
	$(PY) -m pytest -q

check:
	$(PY) -m recon.cli check

stats:
	$(PY) -m recon.cli stats

report:
	$(PY) -m recon.cli stats --save

case:
	$(PY) -m recon.cli dump-case --show-answer

case-composite:
	$(PY) -m recon.cli dump-case --composite --show-answer

tasks:
	$(PY) -m recon.cli tasks

baseline:
	$(PY) -m recon.cli eval-baseline

AGENT_LIMIT   ?= 60
AGENT_WORKERS ?= 10

agent:
	$(PY) -m recon.cli eval-agent --limit $(AGENT_LIMIT) --workers $(AGENT_WORKERS)

replay:
	$(PY) -m recon.cli replay --failed-only

VAR_REPEAT ?= 3
SPLIT      ?= all

variance:
	$(PY) -m recon.cli variance --repeat $(VAR_REPEAT) --split $(SPLIT) --workers $(AGENT_WORKERS)

ablate:
	$(PY) -m recon.cli ablate --limit $(AGENT_LIMIT) --workers $(AGENT_WORKERS)

route:
	$(PY) -m recon.cli route --all-tasks --mode review --gate any --workers $(AGENT_WORKERS)

route-dry:
	$(PY) -m recon.cli route --all-tasks --dry-run

repro:
	$(PY) -m recon.cli verify-repro

# ---- 轨迹归档（只追加，跨世界重建存活）----
archive-stats:
	$(PY) -m recon.cli archive-stats

archive-export:
	$(PY) -m recon.cli archive-export --out data/sft.jsonl

# ---- 阶段 6 holdout（正式评测只能启动一次）----
holdout-build:
	$(PY) -m recon.cli holdout-build

holdout-check:
	$(PY) -m recon.cli holdout-check

holdout-eval:
	@if [ "$(CONFIRM_HOLDOUT)" != "yes" ]; then \
		echo "拒绝启动：这是一次性正式评测。确认后使用 CONFIRM_HOLDOUT=yes make holdout-eval"; \
		exit 2; \
	fi
	$(PY) -m recon.cli holdout-eval --confirm-once --workers $(AGENT_WORKERS)

PAIRED_ARGS   ?=
PAIRED_METRIC ?= attr_exact
PAIRED_OUT    ?= docs/stage6_paired_stats.md

paired-stats:
	@if [ -z "$(PAIRED_ARGS)" ]; then \
		echo "缺少运行对，例如：PAIRED_ARGS='--pair RUN_A RUN_B' make paired-stats"; \
		exit 2; \
	fi
	$(PY) -m recon.eval.paired_stats $(PAIRED_ARGS) --metric $(PAIRED_METRIC) \
		--out $(PAIRED_OUT)

all: build test report baseline

# ⚠️ 原来这里是 `rm -rf data`，会把 data/archive.db 一起删掉 ——
#    那是几十轮实验积累的轨迹，删了不可恢复，而世界库随时能重建。
#    所以只删可重建的东西；要删归档必须手动，见 clean-archive。
clean:
	rm -f data/recon.db data/recon.db-wal data/recon.db-shm
	rm -rf __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -exec rm -rf {} +
	@echo "已清理世界库与缓存。data/archive.db 保留（轨迹归档，不可重建）。"

clean-archive:
	@echo "这会永久删除 data/archive.db —— 所有历史轨迹，不可恢复。"
	@echo "确认请手动执行： rm -f data/archive.db*"
