"""Tests for the LD-aware model resolver."""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.model_router import _config_default, resolve_model


def _make_config() -> ChatConfig:
    """Build a config with the canonical defaults so tests read naturally."""
    return ChatConfig(
        fast_standard_model="anthropic/claude-sonnet-4-6",
        fast_advanced_model="anthropic/claude-opus-4.7",
        thinking_standard_model="anthropic/claude-sonnet-4-6",
        thinking_advanced_model="anthropic/claude-opus-4.7",
    )


_FULL_PAYLOAD = {
    "fast": {
        "standard": "fast-standard-model",
        "advanced": "fast-advanced-model",
    },
    "thinking": {
        "standard": "thinking-standard-model",
        "advanced": "thinking-advanced-model",
    },
}


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
    async def test_payload_none_falls_back(self):
        """LD unset / serving ``None`` → ChatConfig default for every cell."""
        cfg = _make_config()
        with patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(return_value=None),
        ):
            assert (
                await resolve_model("fast", "standard", "u", config=cfg)
                == cfg.fast_standard_model
            )
            assert (
                await resolve_model("fast", "advanced", "u", config=cfg)
                == cfg.fast_advanced_model
            )
            assert (
                await resolve_model("thinking", "standard", "u", config=cfg)
                == cfg.thinking_standard_model
            )
            assert (
                await resolve_model("thinking", "advanced", "u", config=cfg)
                == cfg.thinking_advanced_model
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "mode, tier, expected",
        [
            ("fast", "standard", "fast-standard-model"),
            ("fast", "advanced", "fast-advanced-model"),
            ("thinking", "standard", "thinking-standard-model"),
            ("thinking", "advanced", "thinking-advanced-model"),
        ],
    )
    async def test_full_payload_routes_each_cell(self, mode, tier, expected):
        """Full JSON with all 4 cells → each cell returns its mapped value."""
        cfg = _make_config()
        with patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(return_value=_FULL_PAYLOAD),
        ):
            result = await resolve_model(mode, tier, "user-1", config=cfg)
        assert result == expected

    @pytest.mark.asyncio
    async def test_partial_payload_missing_mode_falls_back(self):
        """Only ``fast`` provided → present cells returned, missing mode falls back."""
        cfg = _make_config()
        payload = {
            "fast": {
                "standard": "fast-standard-override",
                "advanced": "fast-advanced-override",
            }
        }
        with patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(return_value=payload),
        ):
            assert (
                await resolve_model("fast", "standard", "u", config=cfg)
                == "fast-standard-override"
            )
            assert (
                await resolve_model("fast", "advanced", "u", config=cfg)
                == "fast-advanced-override"
            )
            assert (
                await resolve_model("thinking", "standard", "u", config=cfg)
                == cfg.thinking_standard_model
            )
            assert (
                await resolve_model("thinking", "advanced", "u", config=cfg)
                == cfg.thinking_advanced_model
            )

    @pytest.mark.asyncio
    async def test_partial_payload_missing_tier_falls_back(self):
        """Only ``fast.standard`` set → that cell returned, fast.advanced falls back."""
        cfg = _make_config()
        payload = {"fast": {"standard": "fast-standard-override"}}
        with patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(return_value=payload),
        ):
            assert (
                await resolve_model("fast", "standard", "u", config=cfg)
                == "fast-standard-override"
            )
            assert (
                await resolve_model("fast", "advanced", "u", config=cfg)
                == cfg.fast_advanced_model
            )

    @pytest.mark.asyncio
    async def test_whitespace_is_stripped(self):
        cfg = _make_config()
        payload = {"thinking": {"advanced": "  xai/grok-4  "}}
        with patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(return_value=payload),
        ):
            result = await resolve_model("thinking", "advanced", "user-1", config=cfg)
        assert result == "xai/grok-4"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bogus_payload",
        [
            "anthropic/claude-sonnet-4-6",  # raw string (legacy shape)
            ["anthropic/claude-sonnet-4-6"],
            42,
            True,
        ],
    )
    async def test_non_dict_payload_falls_back_with_warning(
        self, caplog, bogus_payload
    ):
        """Non-dict payload → all cells fall back + warning logged."""
        cfg = _make_config()
        with caplog.at_level(logging.WARNING, logger="backend.copilot.model_router"):
            with patch(
                "backend.copilot.model_router.get_feature_flag_value",
                new=AsyncMock(return_value=bogus_payload),
            ):
                result = await resolve_model("fast", "standard", "user-1", config=cfg)
        assert result == cfg.fast_standard_model
        assert any("expected a JSON object" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "value",
        [42, ["x"], None, True, {"nested": "dict"}],
    )
    async def test_non_string_cell_value_falls_back_with_warning(self, caplog, value):
        """LD misconfigured cell value (number, list, bool, dict) — don't try
        to use it as a model name; return the config default.  Warning
        must say 'non-string' (skipped for ``None`` since that means the
        cell is simply unset, not misconfigured)."""
        cfg = _make_config()
        payload = {"fast": {"advanced": value}}
        with caplog.at_level(logging.WARNING, logger="backend.copilot.model_router"):
            with patch(
                "backend.copilot.model_router.get_feature_flag_value",
                new=AsyncMock(return_value=payload),
            ):
                result = await resolve_model("fast", "advanced", "user-1", config=cfg)
        assert result == cfg.fast_advanced_model
        if value is None:
            # ``None`` is a missing cell, not a misconfiguration — no warning.
            assert not any(
                "non-string" in r.message or "empty string" in r.message
                for r in caplog.records
            )
        else:
            assert any("non-string" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_empty_string_cell_falls_back_with_empty_in_warning(self, caplog):
        """When LD returns ``""`` for a cell the warning must say 'empty
        string' — not 'non-string' — so the operator doesn't chase a
        type bug when the flag is simply unset to an empty value."""
        cfg = _make_config()
        payload = {"fast": {"standard": ""}}
        with caplog.at_level(logging.WARNING, logger="backend.copilot.model_router"):
            with patch(
                "backend.copilot.model_router.get_feature_flag_value",
                new=AsyncMock(return_value=payload),
            ):
                result = await resolve_model("fast", "standard", "user-1", config=cfg)
        assert result == cfg.fast_standard_model
        messages = [r.message for r in caplog.records]
        assert any("empty string" in m for m in messages)
        assert not any("non-string" in m for m in messages)

    @pytest.mark.asyncio
    async def test_mode_cell_not_dict_falls_back_silently(self):
        """LD payload has ``"fast": "claude"`` (string instead of dict) —
        treat the whole mode as missing and fall back without spamming
        a warning per cell (the non-dict-payload branch already warns
        once for the top-level shape issue when applicable)."""
        cfg = _make_config()
        payload = {"fast": "anthropic/claude-sonnet-4-6"}
        with patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(return_value=payload),
        ):
            assert (
                await resolve_model("fast", "standard", "u", config=cfg)
                == cfg.fast_standard_model
            )
            assert (
                await resolve_model("fast", "advanced", "u", config=cfg)
                == cfg.fast_advanced_model
            )

    @pytest.mark.asyncio
    async def test_ld_exception_falls_back_with_warning(self, caplog):
        """LD client throws (network blip, SDK init race) — serve the default
        instead of failing the whole request, and log with ``exc_info``."""
        cfg = _make_config()
        with caplog.at_level(logging.WARNING, logger="backend.copilot.model_router"):
            with patch(
                "backend.copilot.model_router.get_feature_flag_value",
                new=AsyncMock(side_effect=RuntimeError("LD down")),
            ):
                result = await resolve_model("fast", "standard", "user-1", config=cfg)
        assert result == cfg.fast_standard_model
        records = [r for r in caplog.records if "LD lookup failed" in r.message]
        assert records, "expected an LD-failure warning"
        assert records[0].exc_info is not None

    @pytest.mark.asyncio
    async def test_single_ld_call_per_resolve(self):
        """Each ``resolve_model`` call hits the single JSON flag exactly once
        — regression guard against accidentally re-introducing per-cell
        flag fan-out."""
        cfg = _make_config()
        calls: list[str] = []

        async def _capture(flag_key, user_id, default):
            calls.append(flag_key)
            return _FULL_PAYLOAD

        with patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(side_effect=_capture),
        ):
            await resolve_model("fast", "standard", "u", config=cfg)
            await resolve_model("fast", "advanced", "u", config=cfg)
            await resolve_model("thinking", "standard", "u", config=cfg)
            await resolve_model("thinking", "advanced", "u", config=cfg)

        assert calls == ["copilot-model-routing"] * 4
