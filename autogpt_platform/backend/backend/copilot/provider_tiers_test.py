"""Describing a provider's tiers without asking whether the user may use it."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_mock

from backend.copilot import provider_tiers
from backend.copilot.provider_tiers import describe_provider_tiers, display_name
from backend.data.llm_registry import registry

USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"


@pytest.fixture
def real_catalog():
    """Resolution is only observable with the catalog actually loaded, and it
    is process state, so it is put back afterwards."""
    old = (
        registry._dynamic_models,
        registry._date_stripped_models,
        registry._routes,
        registry._loaded,
    )
    registry.load_catalog()
    yield
    (
        registry._dynamic_models,
        registry._date_stripped_models,
        registry._routes,
        registry._loaded,
    ) = old


@pytest.fixture(autouse=True)
def engine(mocker: pytest_mock.MockerFixture):
    mocker.patch.object(
        provider_tiers, "resolve_use_sdk", new=AsyncMock(return_value=False)
    )
    mocker.patch.object(
        provider_tiers,
        "resolve_model_route",
        new=AsyncMock(
            side_effect=lambda mode, tier, user_id, *, config: SimpleNamespace(
                model=f"{mode}-{tier}-model", source="config"
            )
        ),
    )


@pytest.mark.asyncio
async def test_describes_chatgpt_without_the_user_having_it(
    real_catalog,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """The whole reason this exists: the connect dialog and the plan cards
    describe ChatGPT to someone who has not connected it, and the offers list
    deliberately omits a connection nobody can select."""
    mocker.patch.object(
        provider_tiers.llm_registry,
        "get_route",
        side_effect=lambda surface, mode, tier: {
            "standard": "gpt-5.6-terra",
            "advanced": "gpt-5.6-sol",
        }[tier],
    )

    providers = await describe_provider_tiers(USER_ID)

    chatgpt = next(p for p in providers if p.provider_family == "openai")
    # Named, not slugged: the catalog knows both, which is the point.
    assert [(t.label, t.display_model) for t in chatgpt.tiers] == [
        ("Balanced", "GPT-5.6 Terra"),
        ("Advanced", "GPT-5.6 Sol"),
    ]


@pytest.mark.asyncio
async def test_describes_the_platform_too(real_catalog) -> None:
    providers = await describe_provider_tiers(USER_ID)

    platform = next(p for p in providers if p.provider_family == "autogpt")
    assert [t.label for t in platform.tiers] == ["Balanced", "Advanced"]
    assert [t.tier for t in platform.tiers] == ["standard", "advanced"]


@pytest.mark.asyncio
async def test_says_nothing_about_access(real_catalog) -> None:
    """A tier here carries no ``selectable`` and no ``lock_reason`` -- those
    are answers about a user, and this is a statement about the catalog."""
    providers = await describe_provider_tiers(USER_ID)

    tier = providers[0].tiers[0]
    assert not hasattr(tier, "selectable")
    assert not hasattr(tier, "lock_reason")


def test_names_a_model_configured_in_transport_spelling(real_catalog) -> None:
    assert display_name("anthropic/claude-sonnet-5") == "Claude Sonnet 5"


def test_falls_back_to_the_slug_tail_for_an_unlisted_model(real_catalog) -> None:
    assert display_name("acme/a-model-nobody-has-heard-of") == (
        "a-model-nobody-has-heard-of"
    )


def test_has_no_name_for_no_model(real_catalog) -> None:
    assert display_name(None) is None
