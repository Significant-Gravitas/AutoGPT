"""Tests for Graphiti client management — derive_group_id and evict_client."""

from unittest.mock import MagicMock

import pytest

from backend.copilot.config import ChatConfig

from . import client as client_mod
from .client import derive_group_id, evict_client, make_flex_graphiti_client


class TestDeriveGroupId:
    def test_empty_user_id_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            derive_group_id("")

    def test_all_invalid_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="empty group_id after sanitization"):
            derive_group_id("!!!")

    def test_user_id_with_stripped_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid characters"):
            derive_group_id("abc.def")

    def test_valid_uuid_passthrough(self) -> None:
        uid = "883cc9da-fe37-4863-839b-acba022bf3ef"
        result = derive_group_id(uid)
        assert result == f"user_{uid}"

    def test_simple_alphanumeric_id(self) -> None:
        result = derive_group_id("user123")
        assert result == "user_user123"

    def test_hyphens_and_underscores_allowed(self) -> None:
        result = derive_group_id("a-b_c")
        assert result == "user_a-b_c"


class TestEvictClient:
    @pytest.mark.asyncio
    async def test_evict_nonexistent_group_id_does_not_raise(self) -> None:
        await evict_client("no-such-group-id")


class TestHyphenInGroupIdRegression:
    """Regression coverage for upstream Graphiti issue #1483.

    `AutoGPTFalkorDriver.build_fulltext_query` interpolates the group_id
    into a Redisearch tag filter as ``(@group_id:user_abc-def)``. In
    Redisearch query syntax ``-`` means NOT — so the literal hyphen
    inside a UUID-derived group_id is interpreted as a negation, which
    causes silent search misses on every user whose UUID contains
    hyphens (i.e. every user).

    Upstream tracking: https://github.com/getzep/graphiti/issues/1483

    The xfail test below documents what we *want* the contract to be.
    Once upstream resolves the bug — or we mitigate locally by
    converting hyphens to underscores in ``derive_group_id`` — flip
    the marker off and the regression suite catches a re-introduction.
    """

    def test_derive_group_id_preserves_hyphens(self) -> None:
        """Sanity check: hyphenated UUIDs round-trip through derivation."""
        uid = "a1b2c3d4-e5f6-7890-1234-567890abcdef"
        result = derive_group_id(uid)
        assert result == f"user_{uid}"
        assert "-" in result

    @pytest.mark.xfail(
        reason=(
            "Graphiti #1483: AutoGPTFalkorDriver.build_fulltext_query produces a "
            "Redisearch tag filter that interprets hyphens in the group_id as NOT. "
            "Mitigation deferred — either upstream fix or sanitize in derive_group_id."
        ),
        strict=False,
    )
    def test_build_fulltext_query_escapes_hyphens_in_group_id(self) -> None:
        from .falkordb_driver import AutoGPTFalkorDriver

        # Instantiate without connecting — we only need build_fulltext_query.
        driver = AutoGPTFalkorDriver.__new__(AutoGPTFalkorDriver)
        result = driver.build_fulltext_query(
            query="alice",
            group_ids=["user_a1b2c3d4-e5f6-7890-1234-567890abcdef"],
        )

        # The contract we want: hyphens are escaped (Redisearch backtick
        # form) or otherwise rendered safe for tag-filter matching.
        # Today's output is the raw interpolation, which Redisearch treats
        # as a NOT — hence xfail.
        # Acceptable forms include backtick-escaped values
        #   (@group_id:`user_…-…`)
        # or substituting hyphens with underscores in the filter.
        unsafe_pattern = "user_a1b2c3d4-e5f6-7890"  # raw hyphenated form
        assert unsafe_pattern not in result, (
            "Hyphenated group_id is interpolated raw into the Redisearch tag "
            "filter — Redisearch treats `-` as NOT, causing silent search misses."
        )


def _patch_chat_cfg(monkeypatch: pytest.MonkeyPatch, cfg: ChatConfig) -> None:
    """Swap the ``copilot.sdk.env.config`` singleton for a per-test
    transport. Mirrors the pattern used by ``transport_routing_test``
    and ``graphiti.config_test``."""
    from backend.copilot.sdk import env

    monkeypatch.setattr(env, "config", cfg)


class TestMakeFlexGraphitiClient:
    """The flex client is the seam where transport identity actually
    matters at runtime — the OpenAI ``service_tier="flex"`` parameter
    delivers a ~50% discount via OpenRouter's pass-through but blows
    up against Ollama (no Responses API). Pin the transport gate so a
    local install doesn't 404 on its weekly community rebuild."""

    @pytest.mark.asyncio
    async def test_returns_regular_openaiclient_when_flex_unsupported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``CHAT_USE_LOCAL=true`` → ``transport.supports_flex_tier``
        is False → fall back to the cached non-flex client so the
        rebuild runs at sync price instead of 404-ing."""
        _patch_chat_cfg(
            monkeypatch,
            ChatConfig(
                use_local=True,
                api_key="ollama-placeholder",
                base_url="http://localhost:11434/v1",
            ),
        )
        captured: dict = {}

        def _fake_build_graphiti(group_id: str, llm_client):
            captured["llm_client"] = llm_client
            captured["group_id"] = group_id
            return MagicMock(name="fake-graphiti")

        # Both clients have heavy constructor side effects (network
        # connection, validator runs); patch them to lightweight
        # sentinels so we can inspect which one was instantiated.
        flex_sentinel = MagicMock(name="FlexOpenAIClient")
        regular_sentinel = MagicMock(name="OpenAIClient")
        monkeypatch.setattr(client_mod, "_build_graphiti", _fake_build_graphiti)
        monkeypatch.setattr(
            client_mod, "_build_llm_config", lambda: MagicMock(name="LLMConfig")
        )

        # ``make_flex_graphiti_client`` imports both classes lazily;
        # patch them on their actual modules so the import inside the
        # function picks up the sentinel.
        import graphiti_core.llm_client

        from . import flex_client as flex_module

        monkeypatch.setattr(flex_module, "FlexOpenAIClient", flex_sentinel)
        monkeypatch.setattr(graphiti_core.llm_client, "OpenAIClient", regular_sentinel)

        await make_flex_graphiti_client("user_abc")

        # The regular (non-flex) client was constructed.
        assert regular_sentinel.called, "expected fallback to regular OpenAIClient"
        assert (
            not flex_sentinel.called
        ), "flex client must not be constructed under local transport"
        # And the constructed instance was passed into _build_graphiti.
        assert captured["llm_client"] is regular_sentinel.return_value

    @pytest.mark.asyncio
    async def test_returns_flex_client_when_transport_supports_flex(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenRouter transport keeps the flex tier — the gate only
        kicks in for non-flex-capable transports."""
        _patch_chat_cfg(
            monkeypatch,
            ChatConfig(
                use_openrouter=True,
                api_key="or-key",
                base_url="https://openrouter.ai/api/v1",
            ),
        )
        captured: dict = {}

        def _fake_build_graphiti(group_id: str, llm_client):
            captured["llm_client"] = llm_client
            return MagicMock(name="fake-graphiti")

        flex_sentinel = MagicMock(name="FlexOpenAIClient")
        regular_sentinel = MagicMock(name="OpenAIClient")
        monkeypatch.setattr(client_mod, "_build_graphiti", _fake_build_graphiti)
        monkeypatch.setattr(
            client_mod, "_build_llm_config", lambda: MagicMock(name="LLMConfig")
        )

        import graphiti_core.llm_client

        from . import flex_client as flex_module

        monkeypatch.setattr(flex_module, "FlexOpenAIClient", flex_sentinel)
        monkeypatch.setattr(graphiti_core.llm_client, "OpenAIClient", regular_sentinel)

        await make_flex_graphiti_client("user_abc")

        assert flex_sentinel.called, "expected FlexOpenAIClient under openrouter"
        assert not regular_sentinel.called
        assert captured["llm_client"] is flex_sentinel.return_value
