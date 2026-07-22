"""Tests for the LD-aware model resolver."""

import logging
import textwrap
from unittest.mock import AsyncMock, patch

import pytest

import backend.data.llm_registry.registry as reg
from backend.copilot.config import ChatConfig
from backend.copilot.model_router import _config_default, resolve_model_route


def _function_calls(module_name: str, obj_name: str, callee: str) -> bool:
    """AST-verified: *obj_name* in *module_name* contains a real CALL of
    *callee* — a commented-out call cannot pass (unlike source grep)."""
    import ast
    import importlib
    import inspect

    module = importlib.import_module(module_name)
    src = textwrap.dedent(inspect.getsource(getattr(module, obj_name)))
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name == callee:
                return True
    return False


async def resolve_model(mode, tier, user_id, *, config):
    """Test-local stand-in for the deleted back-compat wrapper: these tests
    assert on resolution VALUES; the routing source is covered elsewhere."""
    return (await resolve_model_route(mode, tier, user_id, config=config)).model


@pytest.fixture(autouse=True)
def _empty_catalog_by_default():
    """These tests assert the resolver's ungated (pre-catalog) behavior, which
    requires an EMPTY registry — but other test modules (registry_test) load
    the real catalog into the module globals and legitimately leave it there.
    Snapshot, clear, restore, so suite order can never change outcomes.
    TestRegistryGating's own fixture layers its populated state on top."""
    old = (reg._dynamic_models, reg._date_stripped_models, reg._routes)
    reg._dynamic_models, reg._date_stripped_models, reg._routes = {}, {}, {}
    yield
    reg._dynamic_models, reg._date_stripped_models, reg._routes = old


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


class TestRegistryGating:
    """LD → registry cell → env resolution with serve-time slug validation."""

    @pytest.fixture(autouse=True)
    def registry_state(self, mocker):
        import backend.data.llm_registry.registry as reg
        from backend.data.llm_registry.registry import (
            RegistryModel,
            RegistryModelMetadata,
        )

        def make(slug, *, enabled=True, visibility="GA"):
            return RegistryModel(
                slug=slug,
                display_name=slug,
                metadata=RegistryModelMetadata(
                    provider="open_router",
                    context_window=100000,
                    max_output_tokens=8192,
                    display_name=slug,
                    provider_name="OpenRouter",
                    creator_name="Test",
                    price_tier=2,
                ),
                provider_display_name="OpenRouter",
                is_enabled=enabled,
                visibility=visibility,
            )

        self.reg = reg
        self.make = make
        old_models, old_routes = reg._dynamic_models, reg._routes
        reg._dynamic_models = {
            "known/model": make("known/model"),
            "disabled/model": make("disabled/model", enabled=False),
            "hidden/model": make("hidden/model", visibility="HIDDEN"),
            "cell/model": make("cell/model"),
            # The env-floor defaults resolve through the gate too (served
            # regardless, but logged) — register them so refusal
            # assertions only count the layer under test.
            "claude-sonnet-4-6": make("claude-sonnet-4-6"),
            "claude-opus-4-7": make("claude-opus-4-7"),
        }
        reg._routes = {}
        # Mirror load_catalog's derived index for the seeded models.
        from backend.data.llm_registry.llm_models import MODEL_DATE_SUFFIX_RE

        reg._date_stripped_models = {
            stripped: m
            for slug, m in reg._dynamic_models.items()
            if (stripped := MODEL_DATE_SUFFIX_RE.sub("", slug)) != slug
        }
        # Cell tests exercise the cloud path — the test env's default
        # behave_as is LOCAL, which (correctly) skips cells entirely.
        import backend.copilot.model_router as router_mod
        from backend.util.settings import BehaveAs

        mocker.patch.object(router_mod.settings.config, "behave_as", BehaveAs.CLOUD)
        router_mod._sentry_reported.clear()
        self.sentry = mocker.patch(
            "backend.copilot.model_router.sentry_sdk.capture_message"
        )
        yield
        reg._dynamic_models, reg._routes = old_models, old_routes

    def _ld(self, mocker, slug):
        mocker.patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(return_value={"fast": {"standard": slug}}),
        )

    @pytest.mark.asyncio
    async def test_known_ld_slug_serves_with_ld_source(self, mocker):
        from backend.copilot.model_router import resolve_model_route

        self._ld(mocker, "known/model")
        resolved = await resolve_model_route(
            "fast", "standard", "user-1", config=_make_config()
        )
        assert resolved == ("known/model", "ld")

    @pytest.mark.asyncio
    async def test_unknown_ld_slug_refused_and_falls_through(self, mocker):
        from backend.copilot.model_router import resolve_model_route

        self._ld(mocker, "typo/model")
        resolved = await resolve_model_route(
            "fast", "standard", "user-1", config=_make_config()
        )
        assert resolved.source == "env"
        self.sentry.assert_called_once()
        assert "'typo/model'" in self.sentry.call_args.args[0]

    @pytest.mark.asyncio
    async def test_sentry_fires_once_per_refused_slug(self, mocker):
        """A bad LD slug refuses on every turn; Sentry hears about it once
        per process, the log hears about it every time."""
        from backend.copilot.model_router import resolve_model_route

        self._ld(mocker, "typo/model")
        for _ in range(3):
            await resolve_model_route(
                "fast", "standard", "user-1", config=_make_config()
            )
        self.sentry.assert_called_once()

    @pytest.mark.asyncio
    async def test_ld_date_suffixed_anthropic_slug_serves(self, mocker):
        """OpenRouter drops the -YYYYMMDD snapshot suffix Anthropic's API
        slugs carry; an LD experiment routing to the 4.5 family must match
        the suffixed catalog slug, not be refused as unknown."""
        from backend.copilot.model_router import resolve_model_route

        self.reg._dynamic_models["claude-haiku-4-5-20251001"] = self.make(
            "claude-haiku-4-5-20251001"
        )
        self.reg._date_stripped_models["claude-haiku-4-5"] = self.reg._dynamic_models[
            "claude-haiku-4-5-20251001"
        ]
        self._ld(mocker, "anthropic/claude-haiku-4-5")
        resolved = await resolve_model_route(
            "fast", "standard", "user-1", config=_make_config()
        )
        assert resolved == ("anthropic/claude-haiku-4-5", "ld")
        self.sentry.assert_not_called()

    @pytest.mark.asyncio
    async def test_ld_vendor_prefixed_openai_slug_serves(self, mocker):
        """LD may route with ANY vendor prefix (openai/gpt-5.4), while the
        catalog stores the bare (possibly date-suffixed) slug — the gate
        must match the model, not the vendor spelling.

        The OpenAI snapshot suffix is -YYYY-MM-DD (dashed), which the
        transport-spelling fallbacks and the -YYYYMMDD date-stripped index
        both miss; resolution here rides the enum's exact alias map
        (openai/gpt-5.4 -> gpt-5.4-2026-03-05). The index below is built
        the same way load_catalog derives it, so this slug is deliberately
        absent from it — proving the enum-alias gate, not a hand-seeded
        index entry, is what resolves the slug.
        """
        self.reg._dynamic_models["gpt-5.4-2026-03-05"] = self.make("gpt-5.4-2026-03-05")
        from backend.data.llm_registry.llm_models import MODEL_DATE_SUFFIX_RE

        self.reg._date_stripped_models = {
            stripped: m
            for slug, m in self.reg._dynamic_models.items()
            if (stripped := MODEL_DATE_SUFFIX_RE.sub("", slug)) != slug
        }
        assert "gpt-5.4" not in self.reg._date_stripped_models
        self._ld(mocker, "openai/gpt-5.4")
        resolved = await resolve_model_route(
            "fast", "standard", "user-1", config=_make_config()
        )
        assert resolved == ("openai/gpt-5.4", "ld")
        self.sentry.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_ld_slug_refused_kill_switch(self, mocker):
        from backend.copilot.model_router import resolve_model_route

        self._ld(mocker, "disabled/model")
        resolved = await resolve_model_route(
            "fast", "standard", "user-1", config=_make_config()
        )
        assert resolved.source == "env"
        self.sentry.assert_called_once()

    @pytest.mark.asyncio
    async def test_hidden_ld_slug_serves(self, mocker):
        """HIDDEN = registered + routable, just not shown in pickers."""
        from backend.copilot.model_router import resolve_model_route

        self._ld(mocker, "hidden/model")
        resolved = await resolve_model_route(
            "fast", "standard", "user-1", config=_make_config()
        )
        assert resolved == ("hidden/model", "ld")
        self.sentry.assert_not_called()

    @pytest.mark.asyncio
    async def test_db_cell_used_when_no_ld(self, mocker):
        from backend.copilot.model_router import resolve_model_route

        mocker.patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(return_value=None),
        )
        self.reg._routes = {("copilot", "fast", "standard"): "cell/model"}
        resolved = await resolve_model_route(
            "fast", "standard", "user-1", config=_make_config()
        )
        assert resolved == ("cell/model", "catalog")

    @pytest.mark.asyncio
    async def test_ld_beats_db_cell(self, mocker):
        from backend.copilot.model_router import resolve_model_route

        self._ld(mocker, "known/model")
        self.reg._routes = {("copilot", "fast", "standard"): "cell/model"}
        resolved = await resolve_model_route(
            "fast", "standard", "user-1", config=_make_config()
        )
        assert resolved == ("known/model", "ld")

    @pytest.mark.asyncio
    async def test_stale_db_cell_refused_falls_to_env(self, mocker):
        from backend.copilot.model_router import resolve_model_route

        mocker.patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(return_value=None),
        )
        self.reg._routes = {("copilot", "fast", "standard"): "disabled/model"}
        cfg = _make_config()
        resolved = await resolve_model_route("fast", "standard", "user-1", config=cfg)
        assert resolved == (cfg.fast_standard_model, "env")
        assert "refused catalog slug" in self.sentry.call_args.args[0]

    @pytest.mark.asyncio
    async def test_no_user_skips_ld_but_uses_db_cell(self):
        from backend.copilot.model_router import resolve_model_route

        self.reg._routes = {("copilot", "fast", "standard"): "cell/model"}
        resolved = await resolve_model_route(
            "fast", "standard", None, config=_make_config()
        )
        assert resolved == ("cell/model", "catalog")

    @pytest.mark.asyncio
    async def test_empty_registry_gates_nothing(self, mocker):
        """Dormant-registry installs keep exact pre-registry behavior."""
        from backend.copilot.model_router import resolve_model_route

        self.reg._dynamic_models = {}
        self._ld(mocker, "totally/unregistered")
        resolved = await resolve_model_route(
            "fast", "standard", "user-1", config=_make_config()
        )
        assert resolved == ("totally/unregistered", "ld")
        self.sentry.assert_not_called()

    @pytest.mark.asyncio
    async def test_ld_openrouter_spelling_matches_bare_catalog_slug(self, mocker):
        """LD sends anthropic/claude-opus-4.6-style slugs; the catalog holds
        bare dashed enum slugs. The gate must match the model, not the
        spelling."""
        from backend.copilot.model_router import resolve_model_route

        self.reg._dynamic_models["claude-opus-4-6"] = self.make("claude-opus-4-6")
        self._ld(mocker, "anthropic/claude-opus-4.6")
        resolved = await resolve_model_route(
            "fast", "standard", "user-1", config=_make_config()
        )
        assert resolved == ("anthropic/claude-opus-4.6", "ld")
        self.sentry.assert_not_called()

    @pytest.mark.asyncio
    async def test_cell_value_is_returned_verbatim(self, mocker):
        """Cells carry transport-ready spellings and are returned untouched;
        the slug-tolerant gate maps them to catalog identity."""
        from backend.copilot.model_router import resolve_model_route

        mocker.patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(return_value=None),
        )
        self.reg._dynamic_models["claude-sonnet-4-6"] = self.make("claude-sonnet-4-6")
        self.reg._routes = {
            ("copilot", "fast", "standard"): "anthropic/claude-sonnet-4.6"
        }
        resolved = await resolve_model_route(
            "fast", "standard", "user-1", config=_make_config()
        )
        assert resolved == ("anthropic/claude-sonnet-4.6", "catalog")

    def test_catalog_cell_values_normalize_on_every_transport(self, mocker):
        """The REAL catalog routing cells must be transport-ready: OpenRouter
        sends them verbatim (so they must be genuine OR slugs — the dot
        forms), and the direct-Anthropic family strips the vendor prefix and
        dedots without raising."""
        from unittest.mock import PropertyMock

        from backend.copilot.config import ChatConfig
        from backend.copilot.model_normalize import normalize_model_for_transport
        from backend.data.llm_registry import get_catalog

        # Cells ship empty (claiming one is an explicit catalog PR), so this
        # governs any real cells that exist PLUS the canonical spellings the
        # runbook tells operators to use — the transport property must hold
        # before anyone claims a cell.
        cells = [
            slug
            for modes in get_catalog().routing.values()
            for tiers in modes.values()
            for slug in tiers.values()
        ] + [
            "anthropic/claude-sonnet-4.6",
            "anthropic/claude-opus-4.7",
        ]

        def cfg_for(transport: str) -> ChatConfig:
            cfg = ChatConfig()
            mocker.patch.object(
                type(cfg),
                "effective_transport",
                new_callable=PropertyMock,
                return_value=transport,
            )
            return cfg

        for slug in cells:
            # OpenRouter passes cells through verbatim — the cell IS the wire slug.
            assert normalize_model_for_transport(slug, cfg_for("openrouter")) == slug
            # Direct-Anthropic strips the vendor prefix and dedots.
            direct = normalize_model_for_transport(slug, cfg_for("subscription"))
            assert "/" not in direct and "." not in direct
            assert direct.startswith("claude-")


class TestLocalTransportSkipsCells:
    """Catalog cells hold cloud slugs; local transports must ignore them."""

    @pytest.mark.asyncio
    async def test_local_transport_resolves_env_not_cell(self, mocker):
        from unittest.mock import PropertyMock

        import backend.data.llm_registry.registry as reg
        from backend.copilot.model_router import resolve_model_route

        old_routes = reg._routes
        reg._routes = {("copilot", "fast", "standard"): "claude-sonnet-4-6"}
        mocker.patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(return_value=None),
        )
        cfg = _make_config()
        mocker.patch.object(
            type(cfg),
            "baseline_provider",
            new_callable=PropertyMock,
            return_value="local",
        )
        try:
            resolved = await resolve_model_route(
                "fast", "standard", "user-1", config=cfg
            )
        finally:
            reg._routes = old_routes

        assert resolved == (cfg.fast_standard_model, "env")


class TestExecutorCatalogLoad:
    """The copilot executor process must load the catalog (turns run there,
    not in the rest API process — an unloaded registry no-ops all routing)."""

    def test_loader_calls_load_catalog(self, mocker):
        from backend.copilot.executor import manager

        load = mocker.patch.object(manager.backend.data.llm_registry, "load_catalog")
        manager._load_catalog()
        load.assert_called_once()

    def test_loader_fails_hard_on_broken_catalog(self, mocker):
        """Post-cutover the catalog is load-bearing — a load failure must
        stop the process, not silently disable routing cells and gating."""
        from backend.copilot.executor import manager

        mocker.patch.object(
            manager.backend.data.llm_registry,
            "load_catalog",
            side_effect=RuntimeError("bad catalog"),
        )
        with pytest.raises(RuntimeError, match="bad catalog"):
            manager._load_catalog()

    def test_executor_run_invokes_the_loader(self):
        assert _function_calls(
            "backend.copilot.executor.manager", "CoPilotExecutor", "_load_catalog"
        ), (
            "CoPilotExecutor.run must load the LLM catalog — turns execute in "
            "this process, and without the load every routing cell and "
            "serve-time gate silently no-ops"
        )


class TestSelfHostedSkipsCells:
    """Cells are cloud deployment config: behave_as != CLOUD ignores them."""

    @pytest.mark.asyncio
    async def test_self_hosted_ld_slug_not_catalog_gated(self, mocker):
        """A self-hosted operator's LD flag may route to their own model
        (e.g. a custom Ollama tag) that the shipped catalog has never heard
        of — the catalog must not veto it. Gating is cloud-only."""
        import backend.data.llm_registry.registry as reg
        from backend.copilot.model_router import resolve_model_route

        old_models, old_routes = reg._dynamic_models, reg._routes
        mocker.patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(return_value={"fast": {"standard": "qwen3:8b-custom"}}),
        )
        try:
            resolved = await resolve_model_route(
                "fast", "standard", "user-1", config=_make_config()
            )
        finally:
            reg._dynamic_models, reg._routes = old_models, old_routes

        assert resolved == ("qwen3:8b-custom", "ld")

    @pytest.mark.asyncio
    async def test_self_hosted_cloud_transport_resolves_env_not_cell(self, mocker):
        import backend.data.llm_registry.registry as reg
        from backend.copilot.model_router import resolve_model_route

        old_models, old_routes = reg._dynamic_models, reg._routes
        # behave_as defaults to LOCAL in the test env — that IS the case
        # under test; no patch needed. Transport stays cloud (openrouter).
        reg._routes = {("copilot", "fast", "standard"): "anthropic/claude-sonnet-4.6"}
        mocker.patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(return_value=None),
        )
        cfg = _make_config()
        try:
            resolved = await resolve_model_route(
                "fast", "standard", "user-1", config=cfg
            )
        finally:
            reg._dynamic_models, reg._routes = old_models, old_routes

        assert resolved == (cfg.fast_standard_model, "env")


class TestEnvFloorIncidentPath:
    """The env default is served even when the catalog refuses it (the last
    layer cannot fall through), but the refusal must be LOUD."""

    @pytest.mark.asyncio
    async def test_killed_env_default_serves_but_screams(self, mocker):
        import backend.copilot.model_router as router_mod
        import backend.data.llm_registry.registry as reg
        from backend.copilot.model_router import resolve_model_route
        from backend.util.settings import BehaveAs

        mocker.patch.object(router_mod.settings.config, "behave_as", BehaveAs.CLOUD)
        router_mod._sentry_reported.clear()
        sentry = mocker.patch("backend.copilot.model_router.sentry_sdk.capture_message")
        mocker.patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(return_value=None),
        )

        cfg = _make_config()
        # Register the env default as KILL-SWITCHED
        from backend.data.llm_registry.registry import (
            RegistryModel,
            RegistryModelMetadata,
        )

        def _entry(slug, enabled):
            return RegistryModel(
                slug=slug,
                display_name=slug,
                metadata=RegistryModelMetadata(
                    provider="anthropic",
                    context_window=1000,
                    max_output_tokens=None,
                    display_name=slug,
                    provider_name="Anthropic",
                    creator_name="Anthropic",
                    price_tier=1,
                ),
                provider_display_name="Anthropic",
                is_enabled=enabled,
            )

        old = (reg._dynamic_models, reg._routes)
        reg._dynamic_models = {"claude-sonnet-4-6": _entry("claude-sonnet-4-6", False)}
        reg._routes = {}
        try:
            resolved = await resolve_model_route(
                "fast", "standard", "user-1", config=cfg
            )
        finally:
            reg._dynamic_models, reg._routes = old

        # Serves anyway (last resort) but Sentry heard about it.
        assert resolved == (cfg.fast_standard_model, "env")
        sentry.assert_called_once()


class TestUnloadedRegistryWiring:
    """Empty-because-never-loaded is a wiring bug and must log loudly once,
    distinct from a legitimately dormant empty registry."""

    @pytest.mark.asyncio
    async def test_never_loaded_process_logs_wiring_error_once(self, mocker, caplog):
        import logging

        import backend.copilot.model_router as router_mod
        import backend.data.llm_registry.registry as reg
        from backend.copilot.model_router import _registry_refuses

        old_models, old_loaded = reg._dynamic_models, reg._loaded
        reg._dynamic_models, reg._loaded = {}, False
        router_mod._unloaded_reported = False
        try:
            with caplog.at_level(logging.ERROR):
                assert await _registry_refuses("x/y", "ld") is None
                assert await _registry_refuses("x/y", "ld") is None
        finally:
            reg._dynamic_models, reg._loaded = old_models, old_loaded
            router_mod._unloaded_reported = False

        wiring_errors = [r for r in caplog.records if "never called" in r.message]
        assert len(wiring_errors) == 1


class TestRestApiCatalogLoad:
    """The REST API lifespan must load the catalog (same wiring guarantee
    the executor gets)."""

    def test_lifespan_invokes_the_loader(self):
        assert _function_calls(
            "backend.api.rest_api", "lifespan_context", "load_catalog"
        ), (
            "rest_api's lifespan must load the LLM catalog — without it "
            "routing cells and serve-time gating silently no-op in the "
            "API process"
        )
