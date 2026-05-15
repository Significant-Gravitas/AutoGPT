"""Unit tests for the shared model-slug normalizer."""

import pytest

from .config import ChatConfig
from .model_normalize import normalize_model_for_transport


def _make_cfg(**kwargs) -> ChatConfig:
    defaults: dict = {
        "thinking_standard_model": "anthropic/claude-sonnet-4-6",
        "thinking_advanced_model": "anthropic/claude-opus-4-7",
        # Aux key satisfies ``_validate_aux_client_for_direct_main`` —
        # these tests target normalize behavior, not the aux check.
        "aux_api_key": "or-aux-key",
    }
    defaults.update(kwargs)
    return ChatConfig(**defaults)


class TestNormalizeModelForTransport:
    def test_openrouter_keeps_prefix(self):
        cfg = _make_cfg(
            use_openrouter=True,
            use_claude_code_subscription=False,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
        )
        assert (
            normalize_model_for_transport("anthropic/claude-sonnet-4.6", cfg)
            == "anthropic/claude-sonnet-4.6"
        )

    def test_direct_anthropic_strips_prefix_and_dots(self):
        cfg = _make_cfg(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=False,
        )
        assert (
            normalize_model_for_transport("anthropic/claude-sonnet-4.6", cfg)
            == "claude-sonnet-4-6"
        )

    def test_direct_anthropic_rejects_non_anthropic_vendor(self):
        cfg = _make_cfg(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=False,
        )
        with pytest.raises(ValueError, match="requires an Anthropic model"):
            normalize_model_for_transport("openai/gpt-4o-mini", cfg)

    def test_subscription_strips_prefix(self):
        cfg = _make_cfg(use_claude_code_subscription=True)
        assert (
            normalize_model_for_transport("anthropic/claude-opus-4.7", cfg)
            == "claude-opus-4-7"
        )

    def test_unprefixed_claude_slug_passes(self):
        # ``claude-sonnet-4-20250514`` (no vendor prefix) is the
        # Anthropic Messages / OpenAI-compat form — must pass through.
        cfg = _make_cfg(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=False,
        )
        assert (
            normalize_model_for_transport("claude-sonnet-4-20250514", cfg)
            == "claude-sonnet-4-20250514"
        )

    def test_unprefixed_non_anthropic_slug_rejected(self):
        # Bare ``gpt-4o-mini`` (no slash, no claude- prefix) would fail
        # at runtime against Anthropic with an opaque model_not_found —
        # raise here so the misconfig surfaces near its source.
        cfg = _make_cfg(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=False,
        )
        with pytest.raises(ValueError, match="Anthropic model slug"):
            normalize_model_for_transport("gpt-4o-mini", cfg)
