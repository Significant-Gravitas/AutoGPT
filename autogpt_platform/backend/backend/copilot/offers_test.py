"""The server-owned description of an AI connection."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_mock

from backend.copilot import offers, transports
from backend.copilot.offers import get_connection_offers, offer_id_for
from backend.copilot.transports import ChatTransportResponse
from backend.util.settings import BehaveAs

USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"


def _transport(
    auth_provider: str = "platform",
    credential_id: str | None = None,
    label: str = "AutoGPT Platform",
    available: bool = True,
    default: bool = False,
) -> ChatTransportResponse:
    return ChatTransportResponse(
        auth_provider=auth_provider,  # type: ignore[arg-type]
        credential_id=credential_id,
        label=label,
        available=available,
        default=default,
    )


@pytest.fixture(autouse=True)
def hosted(mocker: pytest_mock.MockerFixture):
    mocker.patch.object(offers.settings.config, "behave_as", BehaveAs.CLOUD)
    mocker.patch.object(transports.settings.config, "behave_as", BehaveAs.CLOUD)
    mocker.patch.object(offers, "resolve_use_sdk", new=AsyncMock(return_value=False))
    mocker.patch.object(
        offers,
        "resolve_model_route",
        new=AsyncMock(
            side_effect=lambda mode, tier, user_id, *, config: SimpleNamespace(
                model=f"{mode}-{tier}-model", source="config"
            )
        ),
    )


def _mock_transports(
    mocker: pytest_mock.MockerFixture, transport_list: list[ChatTransportResponse]
) -> None:
    mocker.patch.object(
        offers,
        "get_chat_transports",
        new=AsyncMock(return_value=transport_list),
    )


@pytest.mark.asyncio
async def test_an_offer_says_what_backs_a_run(
    mocker: pytest_mock.MockerFixture,
) -> None:
    _mock_transports(
        mocker,
        [_transport(default=True), _transport("codex", "cred-1", "ChatGPT")],
    )

    platform, chatgpt = await get_connection_offers(USER_ID)

    assert platform.backed_by_label == "Your AutoGPT plan"
    assert "spend AutoGPT credits" in platform.description
    assert chatgpt.backed_by_label == "Your ChatGPT plan"
    assert "spend no AutoGPT credits" in chatgpt.description


@pytest.mark.asyncio
async def test_self_host_does_not_deny_credits_it_never_had(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(offers.settings.config, "behave_as", BehaveAs.LOCAL)
    _mock_transports(mocker, [_transport(label="Self-hosted chat", default=True)])

    (offer,) = await get_connection_offers(USER_ID)

    assert offer.backed_by_label == "This server's chat provider"
    assert "credits" not in offer.description


@pytest.mark.asyncio
async def test_provider_family_groups_without_becoming_the_credential(
    mocker: pytest_mock.MockerFixture,
) -> None:
    _mock_transports(mocker, [_transport(), _transport("codex", "cred-1", "ChatGPT")])

    platform, chatgpt = await get_connection_offers(USER_ID)

    assert platform.provider_family == "autogpt"
    assert chatgpt.provider_family == "openai"
    assert chatgpt.auth_method == "chatgpt_oauth"
    assert chatgpt.credential_id == "cred-1"


@pytest.mark.asyncio
async def test_offer_ids_are_stable_and_per_account(
    mocker: pytest_mock.MockerFixture,
) -> None:
    first = _transport("codex", "cred-1", "ChatGPT")
    second = _transport("codex", "cred-2", "ChatGPT")
    _mock_transports(mocker, [_transport(), first, second])

    ids = [offer.offer_id for offer in await get_connection_offers(USER_ID)]

    assert ids == ["platform:deployment", "codex:cred-1", "codex:cred-2"]
    assert offer_id_for(first) == offer_id_for(first)


@pytest.mark.asyncio
async def test_tiers_name_the_model_they_resolve_to(
    mocker: pytest_mock.MockerFixture,
) -> None:
    _mock_transports(mocker, [_transport(default=True)])

    (offer,) = await get_connection_offers(USER_ID)

    assert [tier.label for tier in offer.tiers] == ["Balanced", "Advanced"]
    assert [tier.display_model for tier in offer.tiers] == [
        "fast-standard-model",
        "fast-advanced-model",
    ]


@pytest.mark.asyncio
async def test_tiers_follow_the_engine_the_user_will_actually_run_on(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """The engine is the server's decision, so it is knowable before a turn."""
    mocker.patch.object(offers, "resolve_use_sdk", new=AsyncMock(return_value=True))
    _mock_transports(mocker, [_transport(default=True)])

    (offer,) = await get_connection_offers(USER_ID)

    assert [tier.display_model for tier in offer.tiers] == [
        "thinking-standard-model",
        "thinking-advanced-model",
    ]


@pytest.mark.asyncio
async def test_chatgpt_tiers_are_not_named(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Naming them means leasing a runtime per credential to render a list."""
    _mock_transports(mocker, [_transport("codex", "cred-1", "ChatGPT")])

    (offer,) = await get_connection_offers(USER_ID)

    assert all(tier.display_model is None for tier in offer.tiers)
    assert [tier.label for tier in offer.tiers] == ["Balanced", "Advanced"]


@pytest.mark.asyncio
async def test_an_unresolvable_tier_is_described_without_a_name(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(
        offers,
        "resolve_model_route",
        new=AsyncMock(side_effect=RuntimeError("registry down")),
    )
    _mock_transports(mocker, [_transport(default=True)])

    (offer,) = await get_connection_offers(USER_ID)

    assert all(tier.display_model is None for tier in offer.tiers)
    assert offer.state == "ready"


@pytest.mark.asyncio
async def test_an_unavailable_connection_is_not_selectable(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(offers.settings.config, "behave_as", BehaveAs.LOCAL)
    chat_config = MagicMock()
    chat_config.test_mode = False
    chat_config.use_claude_code_subscription = False
    chat_config.main_client_credentials = (None, "https://api.anthropic.com/v1/")
    mocker.patch.object(transports, "config", chat_config)
    _mock_transports(mocker, [_transport(label="Self-hosted chat", available=False)])

    (offer,) = await get_connection_offers(USER_ID)

    assert offer.state == "unavailable"
    assert offer.selectable is False
    assert all(tier.selectable is False for tier in offer.tiers)
    assert "No chat provider is configured on this server yet." in offer.limitations


@pytest.mark.asyncio
async def test_chatgpt_states_the_edge_a_user_can_actually_hit(
    mocker: pytest_mock.MockerFixture,
) -> None:
    _mock_transports(mocker, [_transport("codex", "cred-1", "ChatGPT")])

    (offer,) = await get_connection_offers(USER_ID)

    assert offer.limitations == [
        "The agent builder's chat panel always runs on AutoGPT."
    ]


@pytest.mark.asyncio
async def test_exactly_one_offer_is_the_default(
    mocker: pytest_mock.MockerFixture,
) -> None:
    _mock_transports(
        mocker,
        [_transport(default=True), _transport("codex", "cred-1", "ChatGPT")],
    )

    offer_list = await get_connection_offers(USER_ID)

    assert [offer.is_default for offer in offer_list].count(True) == 1
