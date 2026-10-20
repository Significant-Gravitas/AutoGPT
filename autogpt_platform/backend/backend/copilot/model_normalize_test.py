"""Unit tests for the shared model-slug normalizer."""

from typing import Any

import pytest

from .config import ChatConfig
from .model_normalize import (
    normalize_model_for_anthropic,
    normalize_model_for_transport,
)


def _make_cfg(**kwargs: Any) -> ChatConfig:
    """A config built from these settings alone: no ``.env`` file
    (``_env_file=None``), and the models and transport flags the
    normalizer reads pinned, so neither a developer's ``.env`` nor a
    ``CHAT_*`` variable in the environment can pick the transport or
    model under test. Explicit arguments outrank the environment."""
    settings: dict[str, Any] = {
        "_env_file": None,
        "fast_standard_model": "anthropic/claude-sonnet-4-6",
        "fast_advanced_model": "anthropic/claude-opus-4-7",
        "thinking_standard_model": "anthropic/claude-sonnet-4-6",
        "thinking_advanced_model": "anthropic/claude-opus-4-7",
        "use_local": False,
        "use_openrouter": False,
        "use_claude_code_subscription": False,
        # Aux key satisfies ``_validate_aux_client_for_direct_main`` —
        # these tests target normalize behavior, not the aux check.
        "aux_api_key": "or-aux-key",
        **kwargs,
    }
    return ChatConfig(**settings)


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
        # ``claude-sonnet-4-6`` (no vendor prefix) is the
        # Anthropic Messages / OpenAI-compat form — must pass through.
        cfg = _make_cfg(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=False,
        )
        assert (
            normalize_model_for_transport("claude-sonnet-4-6", cfg)
            == "claude-sonnet-4-6"
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

    def test_direct_anthropic_error_names_the_transport(self):
        cfg = _make_cfg(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=False,
        )
        with pytest.raises(ValueError, match="'direct_anthropic' transport requires"):
            normalize_model_for_transport("openai/gpt-4o-mini", cfg)

    def test_local_transport_passes_bare_slug_unchanged(self):
        # Local backends (Ollama, vLLM, …) use operator-chosen slugs
        # like ``llama3.1:8b-instruct-q4_K_M`` that don't fit the
        # ``vendor/model`` convention and must not be rewritten — the
        # Anthropic-prefix rule applies only to cloud transports.
        cfg = _make_cfg(
            use_local=True,
            api_key="ollama",
            base_url="http://ollama:11434/v1",
            fast_standard_model="llama3.1:8b-instruct-q4_K_M",
        )
        assert (
            normalize_model_for_transport("llama3.1:8b-instruct-q4_K_M", cfg)
            == "llama3.1:8b-instruct-q4_K_M"
        )


class TestNormalizeModelForAnthropic:
    """The native-Anthropic rule on its own: no config read, so it holds
    under the OpenRouter chat transport too (the dream batch path)."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("anthropic/claude-opus-5.5", "claude-opus-5-5"),
            ("anthropic/claude-sonnet-5", "claude-sonnet-5"),
            ("claude-sonnet-4.6", "claude-sonnet-4-6"),
            ("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
        ],
    )
    def test_strips_prefix_and_dots(self, raw, expected):
        assert normalize_model_for_anthropic(raw) == expected

    def test_is_idempotent(self):
        once = normalize_model_for_anthropic("anthropic/claude-opus-5.5")
        assert normalize_model_for_anthropic(once) == once

    def test_rejects_non_anthropic_vendor(self):
        with pytest.raises(
            ValueError, match="The native Anthropic API requires an Anthropic model"
        ):
            normalize_model_for_anthropic("openai/gpt-4.1-mini")

    def test_rejects_bare_non_claude_slug(self):
        with pytest.raises(ValueError, match="Anthropic model slug"):
            normalize_model_for_anthropic("gpt-4o-mini")
