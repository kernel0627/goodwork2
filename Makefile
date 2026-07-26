PY := /Users/traegang/miniforge3/envs/agent/bin/python

SEED   ?= 42
START  ?= 2026-07-01
DAYS   ?= 7
ORDERS ?= 200
INJECT ?= 120

.PHONY: help build test check stats report case case-composite repro clean all tasks baseline agent replay

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
	@echo "make repro            验证同 seed 可复现"
	@echo "make all              build + test + report"
	@echo ""
	@echo "参数：SEED=$(SEED) START=$(START) DAYS=$(DAYS) ORDERS=$(ORDERS) INJECT=$(INJECT)"

build:
	$(PY) -m recon.cli build --seed $(SEED) --start $(START) --days $(DAYS) \
		--orders-per-day $(ORDERS) --inject-per-day $(INJECT)

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

repro:
	$(PY) -m recon.cli verify-repro

all: build test report baseline

clean:
	rm -rf data __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -exec rm -rf {} +
