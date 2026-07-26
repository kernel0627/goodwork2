"""模型客户端 —— 强制 JSON 输出、token 与成本计量、重试。

两条纪律：
1. **密钥只从环境变量/.env 读，绝不进源码。** `.env` 已在 .gitignore 里。
2. **价格不编。** DeepSeek 的 usage 里给了 cache hit / miss / output 三档 token，
   单价必须由使用者从控制台填进环境变量；没填就如实报「未配置价格」，
   不拿一个猜的数字去算成本 —— 成本数字一旦是编的，整张成本-效果曲线就废了。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# 给足输出预算。v4-flash 的 reasoning token 也算在 completion 里，
# 预算不足时正文会返回空串（见 complete_json 的说明）。
DEFAULT_MAX_TOKENS = 4000
MAX_MAX_TOKENS = 12000

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def load_env(path: Path | None = None) -> None:
    """极简 .env 加载。不引入 python-dotenv。"""
    p = path or ENV_PATH
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


# --------------------------------------------------------------------------

@dataclass
class Usage:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cached_in: int = 0
    reasoning_out: int = 0
    latency_ms: int = 0
    cost_micro_cny: int = 0
    priced: bool = True          # 价格是否配置齐全

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.tokens_in += other.tokens_in
        self.tokens_out += other.tokens_out
        self.cached_in += other.cached_in
        self.reasoning_out += other.reasoning_out
        self.latency_ms += other.latency_ms
        self.cost_micro_cny += other.cost_micro_cny
        self.priced = self.priced and other.priced


@dataclass
class Pricing:
    """单价，单位：元 / 百万 token。全部来自环境变量，缺一项就视为未配置。"""
    in_miss: float | None = None
    in_hit: float | None = None
    out: float | None = None

    @classmethod
    def from_env(cls) -> "Pricing":
        def f(name: str) -> float | None:
            v = os.environ.get(name)
            try:
                return float(v) if v not in (None, "") else None
            except ValueError:
                return None
        return cls(f("RECON_PRICE_IN_MISS_PER_MTOK"),
                   f("RECON_PRICE_IN_HIT_PER_MTOK"),
                   f("RECON_PRICE_OUT_PER_MTOK"))

    @property
    def configured(self) -> bool:
        return None not in (self.in_miss, self.in_hit, self.out)

    def cost_micro_cny(self, miss: int, hit: int, out: int) -> int:
        if not self.configured:
            return 0
        yuan = (miss * self.in_miss + hit * self.in_hit + out * self.out) / 1_000_000
        return int(round(yuan * 1_000_000))


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    name: str

    def complete_json(self, messages: list[dict], *, max_tokens: int = 1200,
                      temperature: float = 0.0) -> tuple[dict, Usage]:
        ...


# --------------------------------------------------------------------------

class DeepSeekClient:
    """OpenAI SDK + base_url。不为每家模型再装一个 SDK。"""

    def __init__(self, model: str | None = None, *, api_key: str | None = None,
                 base_url: str | None = None, max_retries: int = 3,
                 pricing: Pricing | None = None):
        load_env()
        from openai import OpenAI
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise LLMError("缺少 DEEPSEEK_API_KEY（放在 .env 或环境变量里，不要写进源码）")
        self.model = model or os.environ.get("RECON_AGENT_MODEL", "deepseek-v4-flash")
        self.name = self.model
        self.max_retries = max_retries
        self.pricing = pricing or Pricing.from_env()
        self._client = OpenAI(
            api_key=key,
            base_url=base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=90.0, max_retries=0)      # 重试自己管，好统计

    def complete_json(self, messages: list[dict], *, max_tokens: int = DEFAULT_MAX_TOKENS,
                      temperature: float = 0.0) -> tuple[dict, Usage]:
        """强制 JSON。

        ⚠️ max_tokens 必须给足。v4-flash 会先产出 reasoning token，预算太小的话
           推理把额度吃光、正文返回空字符串，表现为 `Expecting value: line 1 column 1`。
           这不是限流也不是模型不听话，是调用方预算设错了 —— 60 条里挂了 5 条。
           所以：默认给足，且截断时自动加倍重试。
        """
        last: Exception | None = None
        budget = max_tokens
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                r = self._client.chat.completions.create(
                    model=self.model, messages=messages,
                    response_format={"type": "json_object"},
                    temperature=temperature, max_tokens=budget)
            except Exception as e:                       # 网络/限流：退避重试
                last = e
                time.sleep(min(2 ** attempt, 8))
                continue
            latency = int((time.time() - t0) * 1000)
            u = r.usage
            miss = getattr(u, "prompt_cache_miss_tokens", None)
            hit = getattr(u, "prompt_cache_hit_tokens", 0) or 0
            if miss is None:
                miss = u.prompt_tokens - hit
            details = getattr(u, "completion_tokens_details", None)
            reasoning = getattr(details, "reasoning_tokens", 0) or 0 if details else 0
            usage = Usage(
                calls=1, tokens_in=u.prompt_tokens, tokens_out=u.completion_tokens,
                cached_in=hit, reasoning_out=reasoning, latency_ms=latency,
                cost_micro_cny=self.pricing.cost_micro_cny(miss, hit, u.completion_tokens),
                priced=self.pricing.configured)

            choice = r.choices[0]
            text = choice.message.content or ""
            if not text.strip():
                # 正文为空：几乎总是推理把 max_tokens 吃光了。加倍预算重试。
                last = LLMError(
                    f"响应正文为空（finish_reason={choice.finish_reason}，"
                    f"输出 {u.completion_tokens} token 其中推理 {reasoning}，"
                    f"预算 {budget}）—— 预算不足，加倍重试")
                budget = min(budget * 2, MAX_MAX_TOKENS)
                continue
            try:
                return json.loads(text), usage
            except json.JSONDecodeError as e:
                # JSON 坏了：把错误回灌给模型再试一次，这是 harness 的一部分
                last = e
                messages = messages + [
                    {"role": "assistant", "content": text[:2000]},
                    {"role": "user", "content":
                        f"上一条输出不是合法 JSON（{e}）。只输出一个 JSON 对象，不要任何其它文字。"},
                ]
        raise LLMError(f"模型调用失败（重试 {self.max_retries} 次）：{last}")


# --------------------------------------------------------------------------

class FakeLLM:
    """离线假模型 —— 测试绝不联网。

    按脚本依次返回预设决策；脚本用完后返回一个兜底的 CONCLUDE。
    """

    name = "fake"

    def __init__(self, script: list[dict] | None = None, *,
                 fallback: dict | None = None, tokens: int = 100):
        self.script = list(script or [])
        self.fallback = fallback or {
            "thought": "脚本已用完，兜底转人工",
            "next_action": {"type": "CONCLUDE"},
            "conclusion": {"root_causes": ["UNKNOWN"], "actions": ["ESCALATE"],
                           "expected_status": "escalated", "confidence": 0.1,
                           "evidence_refs": [], "reasoning": "fake fallback"},
        }
        self.tokens = tokens
        self.calls = 0
        self.seen: list[list[dict]] = []

    def complete_json(self, messages: list[dict], *, max_tokens: int = 1200,
                      temperature: float = 0.0) -> tuple[dict, Usage]:
        self.calls += 1
        self.seen.append(messages)
        out = self.script.pop(0) if self.script else self.fallback
        return out, Usage(calls=1, tokens_in=self.tokens, tokens_out=self.tokens // 2,
                          latency_ms=1, cost_micro_cny=0, priced=False)
