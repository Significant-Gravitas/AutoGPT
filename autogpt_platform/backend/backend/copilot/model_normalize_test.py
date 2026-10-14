"""Unit tests for the shared model-slug normalizer."""

import pytest

from .config import ChatConfig
from .model_normalize import normalize_model_for_transport


def _make_cfg(platform_route: str, **kwargs) -> ChatConfig:
    return ChatConfig(platform_route=platform_route, **kwargs)


class TestNormalizeModelForTransport:
    def test_openrouter_keeps_prefix(self):
        cfg = _make_cfg("openrouter", openrouter_api_key="or-key")
        assert (
            normalize_model_for_transport("anthropic/claude-sonnet-4.6", cfg)
            == "anthropic/claude-sonnet-4.6"
        )

    def test_direct_anthropic_strips_prefix_and_dots(self):
        cfg = _make_cfg("direct_anthropic")
        assert (
            normalize_model_for_transport("anthropic/claude-sonnet-4.6", cfg)
            == "claude-sonnet-4-6"
        )

    def test_direct_anthropic_rejects_non_anthropic_vendor(self):
        cfg = _make_cfg("direct_anthropic")
        with pytest.raises(ValueError, match="serves Anthropic models only"):
            normalize_model_for_transport("openai/gpt-4o-mini", cfg)

    def test_subscription_strips_prefix(self):
        cfg = _make_cfg("subscription")
        assert (
            normalize_model_for_transport("anthropic/claude-opus-4.7", cfg)
            == "claude-opus-4-7"
        )

    def test_unprefixed_claude_slug_passes(self):
        # ``claude-sonnet-4-6`` (no vendor prefix) is the
        # Anthropic Messages / OpenAI-compat form — must pass through.
        cfg = _make_cfg("direct_anthropic")
        assert (
            normalize_model_for_transport("claude-sonnet-4-6", cfg)
            == "claude-sonnet-4-6"
        )

    def test_unprefixed_non_anthropic_slug_rejected(self):
        # Bare ``gpt-4o-mini`` (no slash, no claude- prefix) would fail
        # at runtime against Anthropic with an opaque model_not_found —
        # raise here so the misconfig surfaces near its source.
        cfg = _make_cfg("direct_anthropic")
        with pytest.raises(ValueError, match="serves Anthropic models only"):
            normalize_model_for_transport("gpt-4o-mini", cfg)

    def test_local_route_passes_bare_slug_unchanged(self):
        # Local backends (Ollama, vLLM, …) use operator-chosen slugs
        # like ``llama3.1:8b-instruct-q4_K_M`` that don't fit the
        # ``vendor/model`` convention and must not be rewritten — the
        # Anthropic-prefix rule applies only to the Anthropic routes.
        cfg = _make_cfg(
            "local",
            local_base_url="http://ollama:11434/v1",
            local_adapter="openai_chat",
            local_context_tokens=131_072,
            standard_model="llama3.1:8b-instruct-q4_K_M",
            advanced_model="llama3.1:8b-instruct-q4_K_M",
        )
        assert (
            normalize_model_for_transport("llama3.1:8b-instruct-q4_K_M", cfg)
            == "llama3.1:8b-instruct-q4_K_M"
        )

    def test_disabled_route_passes_the_name_through(self):
        # No platform route: nothing sends the name anywhere, so nothing
        # rewrites or refuses it.
        cfg = _make_cfg("disabled")
        assert normalize_model_for_transport("openai/gpt-4o-mini", cfg) == (
            "openai/gpt-4o-mini"
        )
