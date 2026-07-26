"""OpenAI-compatible 模型客户端 —— 配置解析、JSON、自适应兼容与成本计量。

三条纪律：
1. **密钥只从环境变量/.env 读，绝不进源码。** `.env` 已在 .gitignore 里。
2. **供应商不绑死。** 新的通用变量优先，同时兼容 OpenAI / DeepSeek 旧变量和
   Ollama、vLLM 等本地 OpenAI-compatible 端点。
3. **价格不编。** usage 可能给 cache hit / miss / output 三档 token，
   单价必须由使用者从控制台填进环境变量；没填就如实报「未配置价格」，
   不拿一个猜的数字去算成本 —— 成本数字一旦是编的，整张成本-效果曲线就废了。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

# 给足输出预算。v4-flash 的 reasoning token 也算在 completion 里，
# 预算不足时正文会返回空串（见 complete_json 的说明）。
DEFAULT_MAX_TOKENS = 4000
MAX_MAX_TOKENS = 12000

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def _env_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def load_env(path: Path | None = None) -> None:
    """加载项目 `.env`，但绝不覆盖 shell / CI 已显式设置的变量。"""
    p = path or ENV_PATH
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), _env_value(v))


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    low = value.lower()
    if low in {"changeme", "your-api-key", "your_api_key", "not-set", "none"}:
        return None
    compact = low.replace("-", "").replace("_", "")
    if compact.startswith("sk") and len(set(compact[2:])) <= 2:
        return None
    return value


def _first(env: dict[str, str] | os._Environ[str], *names: str,
           ) -> tuple[str | None, str | None]:
    for name in names:
        value = _clean(env.get(name))
        if value is not None:
            return value, name
    return None, None


def _is_local_url(value: str | None) -> bool:
    if not value:
        return False
    host = (urlparse(value).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1", "host.docker.internal"}


def _normalise_local_base(provider: str, base_url: str | None) -> str | None:
    if provider != "ollama" or not base_url:
        return base_url
    value = base_url.rstrip("/")
    return value if value.endswith("/v1") else value + "/v1"


def _choice(env, name: str, default: str, allowed: set[str]) -> str:
    value = (_clean(env.get(name)) or default).lower()
    if value not in allowed:
        raise LLMError(
            f"{name}={value!r} 不合法，可选：{', '.join(sorted(allowed))}")
    return value


def _number(env, name: str, default: float, *, integer: bool = False):
    raw = _clean(env.get(name))
    if raw is None:
        return int(default) if integer else default
    try:
        return int(raw) if integer else float(raw)
    except ValueError as exc:
        raise LLMError(f"{name} 必须是数字，收到 {raw!r}") from exc


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    api_key: str
    base_url: str | None
    json_mode: str = "auto"        # auto | required | off
    token_param: str = "auto"      # auto | max_tokens | max_completion_tokens
    timeout_seconds: float = 90.0
    max_retries: int = 3
    source: str = "generic"

    @property
    def identity(self) -> str:
        endpoint = (urlparse(self.base_url).netloc if self.base_url
                    else "official")
        return f"{self.provider}:{endpoint}:{self.model}"

    def safe_dict(self) -> dict[str, Any]:
        key = self.api_key
        masked = "(local/no-key)" if key == "local-no-key" else "(configured)"
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url or "(provider default)",
            "api_key": masked,
            "json_mode": self.json_mode,
            "token_param": self.token_param,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "source": self.source,
            "identity": self.identity,
        }


def resolve_model_config(model: str | None = None, *,
                         api_key: str | None = None,
                         base_url: str | None = None,
                         env: dict[str, str] | None = None) -> ModelConfig:
    """按确定性优先级解析配置，不探网、不猜模型。

    优先级：显式参数 > RECON_LLM_* > OPENAI_* > DEEPSEEK_*（旧配置）>
    Ollama/vLLM 本地变量。旧的 RECON_AGENT_MODEL 仍作为模型名后备。
    """
    if env is None:
        load_env()
        values = os.environ
    else:
        values = env

    requested = (_clean(values.get("RECON_LLM_PROVIDER")) or "auto").lower()
    if requested not in {"auto", "openai", "deepseek", "ollama", "vllm",
                         "compatible"}:
        raise LLMError(
            "RECON_LLM_PROVIDER 可选 auto/openai/deepseek/ollama/vllm/compatible")

    generic_key = _clean(api_key) or _first(values, "RECON_LLM_API_KEY")[0]
    generic_base = _clean(base_url) or _first(values, "RECON_LLM_BASE_URL")[0]
    generic_model = (_clean(model)
                     or _first(values, "RECON_LLM_MODEL",
                               "RECON_AGENT_MODEL")[0])
    has_new_config = any(_clean(values.get(name)) for name in (
        "RECON_LLM_API_KEY", "RECON_LLM_BASE_URL", "RECON_LLM_MODEL",
        "RECON_LLM_PROVIDER")) or any((api_key, base_url, model))
    has_new_routing = bool(
        _clean(api_key) or _clean(base_url)
        or _clean(values.get("RECON_LLM_API_KEY"))
        or _clean(values.get("RECON_LLM_BASE_URL"))
        or requested != "auto")

    provider = requested
    source = "RECON_LLM_*" if has_new_config else "auto"
    key = generic_key
    endpoint = generic_base
    selected_model = generic_model

    if provider == "auto" and has_new_routing:
        if endpoint and _is_local_url(endpoint) and key is None:
            provider = "compatible"
        elif endpoint and "deepseek.com" in endpoint:
            provider = "deepseek"
        elif endpoint and "openai.com" in endpoint:
            provider = "openai"
        else:
            provider = "compatible"

    if provider == "auto":
        openai_key, _ = _first(values, "OPENAI_API_KEY")
        deepseek_key, _ = _first(values, "DEEPSEEK_API_KEY")
        ollama_model, _ = _first(values, "OLLAMA_MODEL")
        vllm_base, _ = _first(values, "VLLM_BASE_URL")
        if openai_key:
            provider, source, key = "openai", "OPENAI_*", openai_key
            endpoint = _first(values, "OPENAI_BASE_URL")[0]
            selected_model = selected_model or _first(values, "OPENAI_MODEL")[0]
        elif deepseek_key:
            provider, source, key = "deepseek", "DEEPSEEK_* (legacy)", deepseek_key
            endpoint = (_first(values, "DEEPSEEK_BASE_URL")[0]
                        or "https://api.deepseek.com")
            selected_model = selected_model or "deepseek-v4-flash"
        elif ollama_model:
            provider, source, key = "ollama", "OLLAMA_*", "local-no-key"
            endpoint = (_first(values, "OLLAMA_BASE_URL", "OLLAMA_HOST")[0]
                        or "http://localhost:11434/v1")
            selected_model = selected_model or ollama_model
        elif vllm_base:
            provider, source, key = "vllm", "VLLM_*", "local-no-key"
            endpoint = vllm_base
            selected_model = selected_model or _first(values, "VLLM_MODEL")[0]
        else:
            raise LLMError(
                "没有找到模型配置。推荐填写 RECON_LLM_MODEL / "
                "RECON_LLM_BASE_URL / RECON_LLM_API_KEY；也兼容 OPENAI_*、"
                "DEEPSEEK_*、OLLAMA_*、VLLM_*。")

    if provider == "openai":
        key = key or _first(values, "OPENAI_API_KEY")[0]
        endpoint = endpoint or _first(values, "OPENAI_BASE_URL")[0]
        selected_model = selected_model or _first(values, "OPENAI_MODEL")[0]
    elif provider == "deepseek":
        key = key or _first(values, "DEEPSEEK_API_KEY")[0]
        endpoint = (endpoint or _first(values, "DEEPSEEK_BASE_URL")[0]
                    or "https://api.deepseek.com")
        selected_model = selected_model or "deepseek-v4-flash"
    elif provider == "ollama":
        key = key or "local-no-key"
        endpoint = (endpoint
                    or _first(values, "OLLAMA_BASE_URL", "OLLAMA_HOST")[0]
                    or "http://localhost:11434/v1")
        selected_model = selected_model or _first(values, "OLLAMA_MODEL")[0]
    elif provider == "vllm":
        endpoint = endpoint or _first(values, "VLLM_BASE_URL")[0]
        selected_model = selected_model or _first(values, "VLLM_MODEL")[0]
        if endpoint and _is_local_url(endpoint):
            key = key or "local-no-key"

    endpoint = _normalise_local_base(provider, endpoint)
    if key is None and _is_local_url(endpoint):
        key = "local-no-key"
    if key is None:
        raise LLMError(
            f"{provider} 配置缺少 API key；远程端点必须设置 RECON_LLM_API_KEY")
    if selected_model is None:
        raise LLMError(
            f"{provider} 配置缺少模型名；设置 RECON_LLM_MODEL 或对应供应商的 MODEL")

    json_mode = _choice(
        values, "RECON_LLM_JSON_MODE", "auto", {"auto", "required", "off"})
    token_param = _choice(
        values, "RECON_LLM_TOKEN_PARAM", "auto",
        {"auto", "max_tokens", "max_completion_tokens"})
    timeout = _number(values, "RECON_LLM_TIMEOUT_SECONDS", 90.0)
    retries = _number(values, "RECON_LLM_MAX_RETRIES", 3, integer=True)
    if timeout <= 0 or retries <= 0:
        raise LLMError("timeout 和 max_retries 必须大于 0")
    return ModelConfig(
        provider=provider, model=selected_model, api_key=key,
        base_url=endpoint, json_mode=json_mode, token_param=token_param,
        timeout_seconds=float(timeout), max_retries=int(retries), source=source)


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
        load_env()

        def f(*names: str) -> float | None:
            v, _ = _first(os.environ, *names)
            try:
                return float(v) if v is not None else None
            except ValueError:
                return None
        return cls(
            f("RECON_PRICE_INPUT_PER_MTOK",
              "RECON_PRICE_IN_MISS_PER_MTOK"),
            f("RECON_PRICE_CACHED_INPUT_PER_MTOK",
              "RECON_PRICE_IN_HIT_PER_MTOK"),
            f("RECON_PRICE_OUTPUT_PER_MTOK",
              "RECON_PRICE_OUT_PER_MTOK"))

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


class LLMFatalError(LLMError):
    """永久性错误：余额不足、密钥无效、无权限。

    ⚠️ 这类错误重试是纯浪费，而且会掩盖问题：
       账户余额耗尽时，59 条任务 × 9 轮全部退化成 UNKNOWN，
       报表上看起来像「模型能力差」，实际是账户没钱了。
       必须快速失败并中止整批，而不是安静地产出一堆兜底答案。
    """


# HTTP 状态码里属于「重试没用」的那些
FATAL_STATUS = {400, 401, 402, 403, 404, 422}


def _is_fatal(exc: Exception) -> bool:
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(code, int) and code in FATAL_STATUS:
        return True
    text = str(exc).lower()
    return any(k in text for k in (
        "insufficient balance", "invalid api key", "authentication",
        "quota exceeded", "permission denied", "error code: 402",
        "error code: 401", "error code: 403"))


class LLMClient(Protocol):
    name: str

    def complete_json(self, messages: list[dict], *, max_tokens: int = 1200,
                      temperature: float = 0.0) -> tuple[dict, Usage]:
        ...


# --------------------------------------------------------------------------

def _mentions_unsupported(exc: Exception, parameter: str) -> bool:
    text = str(exc).lower()
    parameter = parameter.lower()
    return parameter in text and any(word in text for word in (
        "unsupported", "not supported", "unknown", "unrecognized",
        "not permitted", "extra inputs", "invalid parameter", "does not support"))


def _decode_json_object(text: str) -> dict:
    """兼容纯 JSON、fenced JSON 和对象前后的少量说明文字。"""
    value = text.strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as first:
        parsed = None
        if value.startswith("```") and value.endswith("```"):
            lines = value.splitlines()
            fenced = "\n".join(lines[1:-1]).strip()
            try:
                parsed = json.loads(fenced)
            except json.JSONDecodeError:
                pass
        if parsed is None:
            start = value.find("{")
            if start >= 0:
                try:
                    parsed, _ = json.JSONDecoder().raw_decode(value[start:])
                except json.JSONDecodeError:
                    pass
        if parsed is None:
            raise first
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("顶层必须是 JSON 对象", value, 0)
    return parsed


def _usage_from_response(response, pricing: Pricing, latency_ms: int) -> Usage:
    raw = getattr(response, "usage", None)
    prompt = int(getattr(raw, "prompt_tokens", 0) or 0)
    completion = int(getattr(raw, "completion_tokens", 0) or 0)
    hit = getattr(raw, "prompt_cache_hit_tokens", None)
    if hit is None:
        details = getattr(raw, "prompt_tokens_details", None)
        hit = getattr(details, "cached_tokens", 0) if details else 0
    hit = int(hit or 0)
    miss_raw = getattr(raw, "prompt_cache_miss_tokens", None)
    miss = int(miss_raw if miss_raw is not None else max(0, prompt - hit))
    details = getattr(raw, "completion_tokens_details", None)
    reasoning = int(
        getattr(details, "reasoning_tokens", 0) or 0 if details else 0)
    return Usage(
        calls=1, tokens_in=prompt, tokens_out=completion,
        cached_in=hit, reasoning_out=reasoning, latency_ms=latency_ms,
        cost_micro_cny=pricing.cost_micro_cny(miss, hit, completion),
        priced=pricing.configured)


class OpenAICompatibleClient:
    """一个客户端覆盖 OpenAI、DeepSeek、兼容网关及本地 Ollama/vLLM。"""

    def __init__(self, model: str | None = None, *, api_key: str | None = None,
                 base_url: str | None = None, max_retries: int | None = None,
                 pricing: Pricing | None = None, config: ModelConfig | None = None,
                 client: Any | None = None):
        self.config = config or resolve_model_config(
            model, api_key=api_key, base_url=base_url)
        self.model = self.config.model
        self.provider = self.config.provider
        self.identity = self.config.identity
        self.name = self.model
        self.max_retries = max_retries or self.config.max_retries
        self.pricing = pricing or Pricing.from_env()
        self.compatibility_fallbacks: list[str] = []
        self._json_enabled = self.config.json_mode != "off"
        self._token_param = (
            "max_tokens" if self.config.token_param == "auto"
            else self.config.token_param)
        self._send_temperature = True

        if client is not None:
            self._client = client
        else:
            from openai import OpenAI
            kwargs: dict[str, Any] = {
                "api_key": self.config.api_key,
                "timeout": self.config.timeout_seconds,
                "max_retries": 0,
            }
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = OpenAI(**kwargs)

    def safe_config(self) -> dict[str, Any]:
        return self.config.safe_dict()

    def _request(self, messages: list[dict], budget: int, temperature: float):
        """只对明确的能力不兼容降级；鉴权等普通 400 仍然快速失败。"""
        while True:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                self._token_param: budget,
            }
            if self._json_enabled:
                kwargs["response_format"] = {"type": "json_object"}
            if self._send_temperature:
                kwargs["temperature"] = temperature
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                if (self.config.json_mode == "auto" and self._json_enabled
                        and _mentions_unsupported(exc, "response_format")):
                    self._json_enabled = False
                    self.compatibility_fallbacks.append(
                        "不支持 response_format，退回提示词约束 JSON")
                    continue
                if (self.config.token_param == "auto"
                        and self._token_param == "max_tokens"
                        and _mentions_unsupported(exc, "max_tokens")):
                    self._token_param = "max_completion_tokens"
                    self.compatibility_fallbacks.append(
                        "不支持 max_tokens，改用 max_completion_tokens")
                    continue
                if (self._send_temperature
                        and _mentions_unsupported(exc, "temperature")):
                    self._send_temperature = False
                    self.compatibility_fallbacks.append(
                        "不支持 temperature，已省略该参数")
                    continue
                raise

    def complete_json(self, messages: list[dict], *, max_tokens: int = DEFAULT_MAX_TOKENS,
                      temperature: float = 0.0) -> tuple[dict, Usage]:
        """优先使用原生 JSON mode；不支持时退回提示词约束并严格解析。"""
        last: Exception | None = None
        budget = max_tokens
        request_messages = list(messages)
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                response = self._request(
                    request_messages, budget, temperature)
            except Exception as exc:
                if _is_fatal(exc):
                    raise LLMFatalError(
                        f"永久性错误，重试无意义：{exc}") from exc
                last = exc
                time.sleep(min(2 ** attempt, 8))
                continue
            latency = int((time.time() - t0) * 1000)
            usage = _usage_from_response(response, self.pricing, latency)
            choices = getattr(response, "choices", None) or []
            if not choices:
                last = LLMError("响应没有 choices")
                continue
            choice = choices[0]
            content = getattr(getattr(choice, "message", None), "content", "")
            text = content if isinstance(content, str) else str(content or "")
            if not text.strip():
                last = LLMError(
                    f"响应正文为空（finish_reason={getattr(choice, 'finish_reason', None)}，"
                    f"输出 {usage.tokens_out} token 其中推理 {usage.reasoning_out}，"
                    f"预算 {budget}）—— 预算不足，加倍重试")
                if budget < MAX_MAX_TOKENS:
                    budget = min(budget * 2, MAX_MAX_TOKENS)
                continue
            try:
                return _decode_json_object(text), usage
            except json.JSONDecodeError as exc:
                last = exc
                request_messages = request_messages + [
                    {"role": "assistant", "content": text[:2000]},
                    {"role": "user", "content":
                        f"上一条输出不是合法 JSON（{exc}）。"
                        "只输出一个 JSON 对象，不要任何其它文字。"},
                ]
        raise LLMError(f"模型调用失败（重试 {self.max_retries} 次）：{last}")


# 旧代码和外部使用者继续可用；新代码统一使用通用名字。
DeepSeekClient = OpenAICompatibleClient


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
