"""通用 LLM 配置和 OpenAI-compatible 能力降级测试；全部离线。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from recon.agent.llm import (
    LLMError,
    LLMFatalError,
    ModelConfig,
    OpenAICompatibleClient,
    Pricing,
    resolve_model_config,
)


def _config(**overrides) -> ModelConfig:
    values = {
        "provider": "compatible",
        "model": "test-model",
        "api_key": "test-secret-key",
        "base_url": "https://example.test/v1",
        "json_mode": "auto",
        "token_param": "auto",
        "timeout_seconds": 10.0,
        "max_retries": 2,
        "source": "test",
    }
    values.update(overrides)
    return ModelConfig(**values)


def _response(content: str, *, prompt: int = 0, completion: int = 0):
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason="stop",
    )
    return SimpleNamespace(usage=usage, choices=[choice])


class _Completions:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _fake_client(*script):
    completions = _Completions(script)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions))
    return client, completions


def test_generic_config_has_highest_environment_priority():
    cfg = resolve_model_config(env={
        "RECON_LLM_API_KEY": "generic-key",
        "RECON_LLM_BASE_URL": "https://gateway.example/v1",
        "RECON_LLM_MODEL": "gateway/model",
        "OPENAI_API_KEY": "openai-key",
        "OPENAI_MODEL": "openai-model",
        "DEEPSEEK_API_KEY": "deepseek-key",
    })
    assert cfg.provider == "compatible"
    assert cfg.api_key == "generic-key"
    assert cfg.base_url == "https://gateway.example/v1"
    assert cfg.model == "gateway/model"


def test_openai_standard_variables_win_over_legacy_deepseek():
    cfg = resolve_model_config(env={
        "OPENAI_API_KEY": "openai-key",
        "OPENAI_MODEL": "gpt-test",
        "DEEPSEEK_API_KEY": "deepseek-key",
        "RECON_AGENT_MODEL": "shared-model-override",
    })
    assert cfg.provider == "openai"
    assert cfg.api_key == "openai-key"
    assert cfg.model == "shared-model-override"


def test_legacy_deepseek_configuration_still_works():
    cfg = resolve_model_config(env={
        "DEEPSEEK_API_KEY": "deepseek-key",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.example",
        "RECON_AGENT_MODEL": "legacy-model",
    })
    assert cfg.provider == "deepseek"
    assert cfg.source == "DEEPSEEK_* (legacy)"
    assert cfg.model == "legacy-model"


def test_placeholder_key_is_ignored():
    with pytest.raises(LLMError, match="没有找到模型配置"):
        resolve_model_config(env={
            "DEEPSEEK_API_KEY": "sk-xxxxxxxxxxxxxxxxxxxxxxxx",
            "RECON_AGENT_MODEL": "model",
        })


def test_local_ollama_needs_no_fake_user_key_and_normalises_v1():
    cfg = resolve_model_config(env={
        "OLLAMA_MODEL": "qwen-local",
        "OLLAMA_HOST": "http://localhost:11434",
    })
    assert cfg.provider == "ollama"
    assert cfg.api_key == "local-no-key"
    assert cfg.base_url == "http://localhost:11434/v1"


def test_generic_local_endpoint_needs_no_key():
    cfg = resolve_model_config(env={
        "RECON_LLM_BASE_URL": "http://127.0.0.1:8000/v1",
        "RECON_LLM_MODEL": "local-model",
    })
    assert cfg.provider == "compatible"
    assert cfg.api_key == "local-no-key"


def test_invalid_endpoint_gets_a_clear_error():
    with pytest.raises(LLMError, match=r"http\(s\) URL"):
        resolve_model_config(env={
            "RECON_LLM_BASE_URL": "gateway.example/v1",
            "RECON_LLM_API_KEY": "key",
            "RECON_LLM_MODEL": "model",
        })


def test_safe_config_never_exposes_full_key():
    cfg = _config(api_key="super-secret-api-key")
    safe = cfg.safe_dict()
    assert "super-secret-api-key" not in str(safe)
    assert safe["api_key"] == "(configured)"
    assert safe["identity"] == "compatible:example.test:test-model"


def test_json_mode_falls_back_for_compatible_endpoint():
    fake, calls = _fake_client(
        RuntimeError("unsupported parameter: response_format"),
        _response("```json\n{\"ok\": true}\n```", prompt=12, completion=4),
    )
    client = OpenAICompatibleClient(
        config=_config(), client=fake, pricing=Pricing())
    value, usage = client.complete_json(
        [{"role": "user", "content": "return json"}])

    assert value == {"ok": True}
    assert usage.tokens_in == 12 and usage.tokens_out == 4
    assert "response_format" in calls.calls[0]
    assert "response_format" not in calls.calls[1]
    assert client.compatibility_fallbacks


def test_token_and_temperature_parameters_adapt():
    fake, calls = _fake_client(
        RuntimeError("unsupported parameter: max_tokens; use max_completion_tokens"),
        RuntimeError("temperature is not supported by this model"),
        _response("{\"ok\": true}"),
    )
    client = OpenAICompatibleClient(
        config=_config(), client=fake, pricing=Pricing())
    value, _ = client.complete_json(
        [{"role": "user", "content": "json"}], max_tokens=321)

    assert value["ok"] is True
    assert calls.calls[-1]["max_completion_tokens"] == 321
    assert "max_tokens" not in calls.calls[-1]
    assert "temperature" not in calls.calls[-1]


def test_missing_usage_is_tolerated_but_not_claimed_as_priced():
    response = SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=[{"text": "{\"ok\": true}"}]),
            finish_reason="stop")],
    )
    fake, _ = _fake_client(response)
    client = OpenAICompatibleClient(
        config=_config(), client=fake, pricing=Pricing(1, 1, 1))
    value, usage = client.complete_json(
        [{"role": "user", "content": "json"}])

    assert value == {"ok": True}
    assert usage.tokens_in == usage.tokens_out == 0
    assert usage.priced is False


def test_authentication_error_does_not_get_disguised_as_compatibility_fallback():
    error = RuntimeError("invalid api key")
    error.status_code = 401
    fake, calls = _fake_client(error)
    client = OpenAICompatibleClient(
        config=_config(), client=fake, pricing=Pricing())

    with pytest.raises(LLMFatalError, match="永久性错误"):
        client.complete_json([{"role": "user", "content": "json"}])
    assert len(calls.calls) == 1
    assert client.compatibility_fallbacks == []


def test_new_price_names_and_legacy_names_are_both_supported(monkeypatch):
    for name in (
        "RECON_PRICE_INPUT_PER_MTOK",
        "RECON_PRICE_CACHED_INPUT_PER_MTOK",
        "RECON_PRICE_OUTPUT_PER_MTOK",
        "RECON_PRICE_IN_MISS_PER_MTOK",
        "RECON_PRICE_IN_HIT_PER_MTOK",
        "RECON_PRICE_OUT_PER_MTOK",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECON_PRICE_INPUT_PER_MTOK", "1.2")
    monkeypatch.setenv("RECON_PRICE_CACHED_INPUT_PER_MTOK", "0.3")
    monkeypatch.setenv("RECON_PRICE_OUTPUT_PER_MTOK", "4.8")
    assert Pricing.from_env() == Pricing(1.2, 0.3, 4.8)

    monkeypatch.delenv("RECON_PRICE_INPUT_PER_MTOK")
    monkeypatch.delenv("RECON_PRICE_CACHED_INPUT_PER_MTOK")
    monkeypatch.delenv("RECON_PRICE_OUTPUT_PER_MTOK")
    monkeypatch.setenv("RECON_PRICE_IN_MISS_PER_MTOK", "2")
    monkeypatch.setenv("RECON_PRICE_IN_HIT_PER_MTOK", "1")
    monkeypatch.setenv("RECON_PRICE_OUT_PER_MTOK", "8")
    assert Pricing.from_env() == Pricing(2.0, 1.0, 8.0)


def test_llm_config_command_is_offline_and_masks_secret(monkeypatch):
    from click.testing import CliRunner

    from recon.cli import cli

    monkeypatch.setenv("RECON_LLM_PROVIDER", "compatible")
    monkeypatch.setenv("RECON_LLM_API_KEY", "command-secret-key")
    monkeypatch.setenv("RECON_LLM_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("RECON_LLM_MODEL", "command-model")
    result = CliRunner().invoke(cli, ["llm-config"])

    assert result.exit_code == 0
    assert "command-model" in result.output
    assert "未发送网络请求" in result.output
    assert "command-secret-key" not in result.output
