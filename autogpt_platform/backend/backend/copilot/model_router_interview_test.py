"""The onboarding interview's own route: catalog cell, then env, and never the
copilot's per-user flag."""

from unittest.mock import AsyncMock

import pytest

import backend.copilot.model_router as router_mod
import backend.data.llm_registry.registry as reg
from backend.copilot.config import ChatConfig
from backend.copilot.model_router import (
    ROUTE_SURFACE_ONBOARDING_INTERVIEW,
    resolve_onboarding_interview_model,
)
from backend.data.llm_registry.catalog import get_catalog
from backend.data.llm_registry.registry import RegistryModel, RegistryModelMetadata
from backend.util.settings import BehaveAs

HAIKU = "anthropic/claude-haiku-5.5"


def _model(slug: str, *, enabled: bool = True) -> RegistryModel:
    return RegistryModel(
        slug=slug,
        display_name=slug,
        metadata=RegistryModelMetadata(
            provider="anthropic",
            context_window=100000,
            max_output_tokens=8192,
            display_name=slug,
            provider_name="Anthropic",
            creator_name="Anthropic",
            price_tier=2,
        ),
        provider_display_name="Anthropic",
        is_enabled=enabled,
    )


@pytest.fixture
def registry(mocker):
    old = (reg._dynamic_models, reg._date_stripped_models, reg._routes)
    reg._dynamic_models = {
        "claude-haiku-5-5": _model("claude-haiku-5-5"),
        "claude-sonnet-5-5": _model("claude-sonnet-5-5"),
        "claude-opus-5-5": _model("claude-opus-5-5", enabled=False),
    }
    reg._date_stripped_models = {}
    reg._routes = {}
    mocker.patch.object(router_mod.settings.config, "behave_as", BehaveAs.CLOUD)
    mocker.patch("backend.copilot.model_router.sentry_sdk.capture_message")
    router_mod._sentry_reported.clear()
    # The copilot's per-user flag is never the interview's to read.
    mocker.patch(
        "backend.copilot.model_router.get_feature_flag_value",
        new=AsyncMock(side_effect=AssertionError("the flag was read")),
    )
    yield reg
    reg._dynamic_models, reg._date_stripped_models, reg._routes = old


def _cell(slug: str) -> None:
    reg._routes = {(ROUTE_SURFACE_ONBOARDING_INTERVIEW, "fast", "standard"): slug}


@pytest.mark.asyncio
async def test_the_catalog_cell_picks_the_interview_s_model(registry):
    _cell("anthropic/claude-sonnet-5.5")

    route = await resolve_onboarding_interview_model(config=ChatConfig())

    assert (route.model, route.source) == ("anthropic/claude-sonnet-5.5", "catalog")


@pytest.mark.asyncio
async def test_without_a_cell_the_env_floor_is_haiku(registry):
    route = await resolve_onboarding_interview_model(config=ChatConfig())

    assert (route.model, route.source) == (HAIKU, "env")


@pytest.mark.asyncio
async def test_a_killed_cell_falls_through_to_the_env_floor(registry):
    _cell("anthropic/claude-opus-5.5")

    route = await resolve_onboarding_interview_model(config=ChatConfig())

    assert (route.model, route.source) == (HAIKU, "env")


@pytest.mark.asyncio
async def test_the_copilot_s_own_cells_never_reach_the_interview(registry):
    reg._routes = {("copilot", "fast", "standard"): "anthropic/claude-sonnet-5.5"}

    route = await resolve_onboarding_interview_model(config=ChatConfig())

    assert route.model == HAIKU


@pytest.mark.asyncio
async def test_a_self_hosted_install_reads_the_env_value_only(registry, mocker):
    mocker.patch.object(router_mod.settings.config, "behave_as", BehaveAs.LOCAL)
    _cell("anthropic/claude-sonnet-5.5")

    route = await resolve_onboarding_interview_model(
        config=ChatConfig(onboarding_interview_model="my/own-model")
    )

    assert (route.model, route.source) == ("my/own-model", "env")


def test_the_shipped_catalog_routes_the_interview_to_haiku_5_5() -> None:
    catalog = get_catalog()
    cell = catalog.routing[ROUTE_SURFACE_ONBOARDING_INTERVIEW]["fast"]["standard"]

    assert cell == HAIKU
    haiku = next(m for m in catalog.models if m.slug == "claude-haiku-5-5")
    assert haiku.is_enabled
