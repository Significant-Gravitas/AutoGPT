"""Tests for the prompt-caching markers (ex-baseline helpers).

Salvaged from ``baseline/service_unit_test.py`` (``TestApplyPromptCacheMarkers``
+ ``TestSupportsPromptCacheMarkers``) when the baseline engine was deleted:
only the cases covering the moved helpers came along — cases for
baseline-only machinery (``_mark_system_message_with_cache_control``, the
LLM-caller memoisation) died with the package.
"""

import pytest

from backend.copilot import cache_markers as cm
from backend.copilot.cache_markers import (
    _build_cached_system_message,
    _fresh_anthropic_caching_headers,
    _fresh_ephemeral_cache_control,
    _is_anthropic_model,
    _mark_tools_with_cache_control,
    _supports_prompt_cache_markers,
)


class TestMarkToolsWithCacheControl:
    def test_last_tool_gets_cache_control(self):
        tools = [
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}},
        ]

        cached_tools = _mark_tools_with_cache_control(tools)

        assert "cache_control" not in cached_tools[0]
        assert cached_tools[-1]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "1h",
        }
        # Last tool's other fields preserved.
        assert cached_tools[-1]["function"] == {"name": "b"}

    def test_does_not_mutate_input(self):
        tools = [{"type": "function", "function": {"name": "a"}}]

        _mark_tools_with_cache_control(tools)

        assert tools == [{"type": "function", "function": {"name": "a"}}]

    def test_empty_tools_safe(self):
        assert _mark_tools_with_cache_control([]) == []


class TestIsAnthropicModel:
    def test_matches_claude_and_anthropic_prefix(self):
        assert _is_anthropic_model("anthropic/claude-sonnet-4-6")
        assert _is_anthropic_model("claude-3-5-sonnet-20241022")
        assert _is_anthropic_model("anthropic.claude-3-5-sonnet-20241022-v2:0")
        assert _is_anthropic_model("ANTHROPIC/Claude-Opus")  # case insensitive

    def test_rejects_other_providers(self):
        assert not _is_anthropic_model("openai/gpt-4o")
        assert not _is_anthropic_model("openai/gpt-5")
        assert not _is_anthropic_model("google/gemini-2.5-pro")
        assert not _is_anthropic_model("xai/grok-4")
        assert not _is_anthropic_model("meta-llama/llama-3.3-70b-instruct")

    def test_rejects_kimi_routes(self):
        """Regression guard: Kimi K2.6 is a reasoning route (reasoning
        extra_body is sent) but NOT an Anthropic route — Moonshot does
        its own auto prompt caching, so ``cache_control`` markers must
        NOT be applied. OpenRouter silently drops them today, but if
        they ever start failing fast we'd want the gate tight."""
        assert not _is_anthropic_model("moonshotai/kimi-k2.6")
        assert not _is_anthropic_model("moonshotai/kimi-k2-thinking")
        assert not _is_anthropic_model("kimi-k2-instruct")


class TestFreshHelpers:
    def test_cache_control_uses_configured_ttl(self, monkeypatch):
        """TTL comes from ChatConfig.prompt_cache_ttl — defaults
        to 1h so the static prefix (system + tools) stays warm across
        workspace users past the 5-min default window."""
        assert cm.config.prompt_cache_ttl == "1h"
        cc = cm._fresh_ephemeral_cache_control()
        assert cc == {"type": "ephemeral", "ttl": "1h"}
        monkeypatch.setattr(cm.config, "prompt_cache_ttl", "5m")
        assert cm._fresh_ephemeral_cache_control() == {
            "type": "ephemeral",
            "ttl": "5m",
        }

    def test_fresh_helpers_return_distinct_objects(self):
        """Regression guard: the `_fresh_*` helpers must return a NEW dict
        on every call.  A future refactor returning a module-level constant
        would silently reintroduce the shared-mutable-state bug flagged
        during earlier review cycles."""
        assert _fresh_ephemeral_cache_control() is not _fresh_ephemeral_cache_control()
        assert (
            _fresh_anthropic_caching_headers() is not _fresh_anthropic_caching_headers()
        )


class TestBuildCachedSystemMessage:
    def test_applies_cache_control(self):
        """The single-message helper wraps the string content in a text block
        with an ephemeral cache_control marker."""
        out = _build_cached_system_message({"role": "system", "content": "hi"})
        assert out["role"] == "system"
        assert out["content"] == [
            {
                "type": "text",
                "text": "hi",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ]

    def test_preserves_extra_fields(self):
        """Unknown keys (e.g. ``name``) survive the transformation."""
        out = _build_cached_system_message(
            {"role": "system", "content": "sys", "name": "dev"}
        )
        assert out["name"] == "dev"
        assert out["role"] == "system"

    def test_non_string_passthrough(self):
        """Pre-marked list content is returned as-is (shallow-copied)."""
        pre_marked = [
            {
                "type": "text",
                "text": "sys",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ]
        out = _build_cached_system_message({"role": "system", "content": pre_marked})
        assert out["content"] is pre_marked


class TestSupportsPromptCacheMarkers:
    """``_supports_prompt_cache_markers`` is the widened gate for
    emitting ``cache_control`` markers on message content.  It's a
    superset of ``_is_anthropic_model`` that ALSO admits Moonshot
    (whose Anthropic-compat endpoint honours the marker) while keeping
    the False answer for OpenAI / Grok / Gemini (which 400 on the
    unknown field)."""

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-sonnet-4-6",
            "claude-3-5-sonnet-20241022",
            "anthropic.claude-3-5-sonnet",
            "ANTHROPIC/Claude-Opus",
        ],
    )
    def test_anthropic_routes_are_supported(self, model):
        assert _supports_prompt_cache_markers(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "moonshotai/kimi-k2.6",
            "moonshotai/kimi-k2-thinking",
            "moonshotai/kimi-k2.5",
            "moonshotai/kimi-k3.0",  # future SKU
        ],
    )
    def test_moonshot_routes_are_supported(self, model):
        """The whole reason this predicate exists — Moonshot must be
        True even though ``_is_anthropic_model`` is False for it."""
        assert _supports_prompt_cache_markers(model) is True
        # Verify this is strictly wider than the anthropic-only check.
        assert _is_anthropic_model(model) is False

    @pytest.mark.parametrize(
        "model",
        [
            "openai/gpt-4o",
            "google/gemini-2.5-pro",
            "xai/grok-4",
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-v3",
        ],
    )
    def test_other_providers_still_rejected(self, model):
        """Regression guard: OpenAI/Grok/Gemini still 400 on
        ``cache_control``, so the widened gate must keep them out."""
        assert _supports_prompt_cache_markers(model) is False
