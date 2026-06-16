"""Tests for LocalLLMRouter.should_route.

Each gate is tested in isolation. The router is pure (no I/O beyond LD
lookups), so we patch ``is_feature_enabled`` / ``get_feature_flag_value``
directly rather than spinning up an LD client.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.tools.local_llm_router import (
    _DEFAULT_TIER_PREFERENCES,
    LocalLLMRouter,
)


def _make_executor(
    *,
    capabilities: list[str] | None = None,
    local_llm_models: list[str] | None = None,
):
    """Build a duck-typed stand-in for LocalPCShim. The router only reads
    ``capabilities`` and ``local_llm_models`` so we can skip the WS dance."""
    return SimpleNamespace(
        capabilities=(
            capabilities
            if capabilities is not None
            else ["shell", "files", "local_llm"]
        ),
        local_llm_models=(
            local_llm_models if local_llm_models is not None else ["llama3.2:3b"]
        ),
    )


def _make_config(policy: str = "prefer_for_fast") -> ChatConfig:
    # ChatConfig has many required defaults; instantiation with no
    # overrides picks the defaults. policy is the only one we care about.
    cfg = ChatConfig()
    cfg.local_llm_policy = policy  # type: ignore[assignment]
    return cfg


@pytest.mark.asyncio
async def test_no_executor_returns_none() -> None:
    router = LocalLLMRouter(_make_config())
    decision = await router.should_route(
        user_id="u1", mode="fast", tier="fast", executor=None
    )
    assert decision is None


@pytest.mark.asyncio
async def test_missing_capability_returns_none() -> None:
    router = LocalLLMRouter(_make_config())
    executor = _make_executor(capabilities=["shell", "files"])
    decision = await router.should_route(
        user_id="u1", mode="fast", tier="fast", executor=executor
    )
    assert decision is None


@pytest.mark.asyncio
async def test_empty_models_returns_none() -> None:
    router = LocalLLMRouter(_make_config())
    executor = _make_executor(local_llm_models=[])
    decision = await router.should_route(
        user_id="u1", mode="fast", tier="fast", executor=executor
    )
    assert decision is None


@pytest.mark.asyncio
async def test_no_user_id_returns_none() -> None:
    router = LocalLLMRouter(_make_config())
    decision = await router.should_route(
        user_id=None, mode="fast", tier="fast", executor=_make_executor()
    )
    assert decision is None


@pytest.mark.asyncio
async def test_ld_flag_off_returns_none() -> None:
    router = LocalLLMRouter(_make_config())
    with patch(
        "backend.copilot.tools.local_llm_router.is_feature_enabled",
        new=AsyncMock(return_value=False),
    ):
        decision = await router.should_route(
            user_id="u1", mode="fast", tier="fast", executor=_make_executor()
        )
    assert decision is None


@pytest.mark.asyncio
async def test_policy_never_returns_none() -> None:
    router = LocalLLMRouter(_make_config(policy="never"))
    with patch(
        "backend.copilot.tools.local_llm_router.is_feature_enabled",
        new=AsyncMock(return_value=True),
    ):
        decision = await router.should_route(
            user_id="u1", mode="fast", tier="fast", executor=_make_executor()
        )
    assert decision is None


@pytest.mark.asyncio
async def test_policy_prefer_for_fast_skips_thinking_mode() -> None:
    router = LocalLLMRouter(_make_config(policy="prefer_for_fast"))
    with patch(
        "backend.copilot.tools.local_llm_router.is_feature_enabled",
        new=AsyncMock(return_value=True),
    ):
        decision = await router.should_route(
            user_id="u1",
            mode="thinking",
            tier="fast",
            executor=_make_executor(),
        )
    assert decision is None


@pytest.mark.asyncio
async def test_policy_prefer_for_fast_routes_fast_mode_with_match() -> None:
    router = LocalLLMRouter(_make_config(policy="prefer_for_fast"))
    with (
        patch(
            "backend.copilot.tools.local_llm_router.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.tools.local_llm_router.get_feature_flag_value",
            new=AsyncMock(return_value=None),  # fall through to defaults
        ),
    ):
        decision = await router.should_route(
            user_id="u1",
            mode="fast",
            tier="fast",
            executor=_make_executor(local_llm_models=["llama3.2:3b"]),
        )
    assert decision == "llama3.2:3b"


@pytest.mark.asyncio
async def test_policy_always_routes_thinking_when_tier_has_model() -> None:
    router = LocalLLMRouter(_make_config(policy="always"))
    with (
        patch(
            "backend.copilot.tools.local_llm_router.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.tools.local_llm_router.get_feature_flag_value",
            new=AsyncMock(
                return_value={
                    "fast": ["llama3.2:3b"],
                    "thinking": ["llama3.1:8b"],
                }
            ),
        ),
    ):
        decision = await router.should_route(
            user_id="u1",
            mode="thinking",
            tier="thinking",
            executor=_make_executor(local_llm_models=["llama3.1:8b", "llama3.2:3b"]),
        )
    assert decision == "llama3.1:8b"


@pytest.mark.asyncio
async def test_default_thinking_tier_is_empty_so_skips_routing() -> None:
    """Per LOCAL_LLM.md, thinking tier defaults to empty list — never route."""
    router = LocalLLMRouter(_make_config(policy="always"))
    with (
        patch(
            "backend.copilot.tools.local_llm_router.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.tools.local_llm_router.get_feature_flag_value",
            new=AsyncMock(return_value=None),
        ),
    ):
        decision = await router.should_route(
            user_id="u1",
            mode="thinking",
            tier="thinking",
            executor=_make_executor(local_llm_models=["llama3.1:8b"]),
        )
    assert decision is None


@pytest.mark.asyncio
async def test_no_tier_model_overlap_falls_back_to_cloud() -> None:
    router = LocalLLMRouter(_make_config(policy="always"))
    with (
        patch(
            "backend.copilot.tools.local_llm_router.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.tools.local_llm_router.get_feature_flag_value",
            new=AsyncMock(return_value=None),
        ),
    ):
        decision = await router.should_route(
            user_id="u1",
            mode="fast",
            tier="fast",
            # Shim has a model, but it's not in the preference list.
            executor=_make_executor(local_llm_models=["never-heard-of-this:8b"]),
        )
    assert decision is None


@pytest.mark.asyncio
async def test_tier_preferences_picks_first_match_in_order() -> None:
    """When the LD payload lists models in order, the first one that's
    also advertised wins — even if a later one is also available."""
    router = LocalLLMRouter(_make_config(policy="always"))
    with (
        patch(
            "backend.copilot.tools.local_llm_router.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.tools.local_llm_router.get_feature_flag_value",
            new=AsyncMock(
                return_value={"fast": ["llama3.2:1b", "llama3.2:3b", "mistral:7b"]}
            ),
        ),
    ):
        decision = await router.should_route(
            user_id="u1",
            mode="fast",
            tier="fast",
            executor=_make_executor(local_llm_models=["llama3.2:3b", "mistral:7b"]),
        )
    # 1b not in advertised → skip; 3b is → win
    assert decision == "llama3.2:3b"


@pytest.mark.asyncio
async def test_malformed_tier_payload_falls_back_to_defaults() -> None:
    router = LocalLLMRouter(_make_config(policy="always"))
    with (
        patch(
            "backend.copilot.tools.local_llm_router.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.tools.local_llm_router.get_feature_flag_value",
            new=AsyncMock(return_value="not a dict"),
        ),
    ):
        decision = await router.should_route(
            user_id="u1",
            mode="fast",
            tier="fast",
            executor=_make_executor(local_llm_models=["llama3.2:3b"]),
        )
    # Default preferences include llama3.2:3b for fast tier
    assert decision == "llama3.2:3b"
    assert "llama3.2:3b" in _DEFAULT_TIER_PREFERENCES["fast"]
