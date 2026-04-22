"""Tests for the LD-aware model resolver."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.model_router import _FLAG_BY_CELL, _config_default, resolve_model


def _make_config() -> ChatConfig:
    """Build a config with the canonical defaults so tests read naturally."""
    return ChatConfig(
        fast_standard_model="anthropic/claude-sonnet-4-6",
        fast_advanced_model="anthropic/claude-opus-4.7",
        thinking_standard_model="anthropic/claude-sonnet-4-6",
        thinking_advanced_model="anthropic/claude-opus-4.7",
    )


class TestConfigDefault:
    def test_fast_standard(self):
        cfg = _make_config()
        assert _config_default(cfg, "fast", "standard") == cfg.fast_standard_model

    def test_fast_advanced(self):
        cfg = _make_config()
        assert _config_default(cfg, "fast", "advanced") == cfg.fast_advanced_model

    def test_thinking_standard(self):
        cfg = _make_config()
        assert (
            _config_default(cfg, "thinking", "standard") == cfg.thinking_standard_model
        )

    def test_thinking_advanced(self):
        cfg = _make_config()
        assert (
            _config_default(cfg, "thinking", "advanced") == cfg.thinking_advanced_model
        )


class TestResolveModel:
    @pytest.mark.asyncio
    async def test_missing_user_returns_fallback(self):
        """Without user_id there's no LD context — skip the lookup entirely."""
        cfg = _make_config()
        with patch("backend.copilot.model_router.get_feature_flag_value") as mock_flag:
            result = await resolve_model("fast", "standard", None, config=cfg)
        assert result == cfg.fast_standard_model
        mock_flag.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_user_strips_whitespace_from_fallback(self):
        """Sentry MEDIUM: the anonymous-user branch returned an unstripped
        config value.  If ``CHAT_*_MODEL`` env carries trailing whitespace
        the downstream ``resolved == tier_default`` check in
        ``_resolve_sdk_model_for_request`` would diverge from the
        whitespace-stripped LD side, bypassing subscription mode for
        every anonymous request.  Strip at the source."""
        cfg = ChatConfig(
            fast_standard_model="anthropic/claude-sonnet-4-6  ",  # trailing ws
            fast_advanced_model="anthropic/claude-opus-4.7",
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            thinking_advanced_model="anthropic/claude-opus-4.7",
        )
        result = await resolve_model("fast", "standard", None, config=cfg)
        assert result == "anthropic/claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_ld_string_override_wins(self):
        """LD-returned model string replaces the config default."""
        cfg = _make_config()
        with patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(return_value="moonshotai/kimi-k2.6"),
        ):
            result = await resolve_model("fast", "standard", "user-1", config=cfg)
        assert result == "moonshotai/kimi-k2.6"

    @pytest.mark.asyncio
    async def test_whitespace_is_stripped(self):
        cfg = _make_config()
        with patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(return_value="  xai/grok-4  "),
        ):
            result = await resolve_model("thinking", "advanced", "user-1", config=cfg)
        assert result == "xai/grok-4"

    @pytest.mark.asyncio
    async def test_non_string_value_falls_back_with_type_in_warning(self, caplog):
        """LD misconfigured as a boolean flag — don't try to use ``True`` as a
        model name; return the config default.  Warning must say
        'non-string' (not 'empty string') so the LD operator knows the
        flag type is wrong, not just missing a value."""
        import logging

        cfg = _make_config()
        with caplog.at_level(logging.WARNING, logger="backend.copilot.model_router"):
            with patch(
                "backend.copilot.model_router.get_feature_flag_value",
                new=AsyncMock(return_value=True),
            ):
                result = await resolve_model("fast", "advanced", "user-1", config=cfg)
        assert result == cfg.fast_advanced_model
        assert any("non-string" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_empty_string_falls_back_with_empty_in_warning(self, caplog):
        """When LD returns ``""`` the warning must say 'empty string' —
        not 'non-string' — so the operator doesn't chase a type bug
        when the flag is simply unset to an empty value."""
        import logging

        cfg = _make_config()
        with caplog.at_level(logging.WARNING, logger="backend.copilot.model_router"):
            with patch(
                "backend.copilot.model_router.get_feature_flag_value",
                new=AsyncMock(return_value=""),
            ):
                result = await resolve_model("fast", "standard", "user-1", config=cfg)
        assert result == cfg.fast_standard_model
        messages = [r.message for r in caplog.records]
        assert any("empty string" in m for m in messages)
        assert not any("non-string" in m for m in messages)

    @pytest.mark.asyncio
    async def test_ld_exception_falls_back(self):
        """LD client throws (network blip, SDK init race) — serve the default
        instead of failing the whole request."""
        cfg = _make_config()
        with patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(side_effect=RuntimeError("LD down")),
        ):
            result = await resolve_model("fast", "standard", "user-1", config=cfg)
        assert result == cfg.fast_standard_model

    @pytest.mark.asyncio
    async def test_all_four_cells_hit_distinct_flags(self):
        """Each (mode, tier) cell must route to its own flag — regression
        guard against copy-paste bugs in the _FLAG_BY_CELL map."""
        cfg = _make_config()
        calls: list[str] = []

        async def _capture(flag_key, user_id, default):
            calls.append(flag_key)
            return default

        with patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(side_effect=_capture),
        ):
            await resolve_model("fast", "standard", "u", config=cfg)
            await resolve_model("fast", "advanced", "u", config=cfg)
            await resolve_model("thinking", "standard", "u", config=cfg)
            await resolve_model("thinking", "advanced", "u", config=cfg)

        assert calls == [
            _FLAG_BY_CELL[("fast", "standard")].value,
            _FLAG_BY_CELL[("fast", "advanced")].value,
            _FLAG_BY_CELL[("thinking", "standard")].value,
            _FLAG_BY_CELL[("thinking", "advanced")].value,
        ]
        assert len(set(calls)) == 4
