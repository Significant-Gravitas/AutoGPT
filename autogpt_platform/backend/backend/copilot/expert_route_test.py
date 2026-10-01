"""The route an unattended turn addressed to an expert runs on."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_mock

from backend.copilot import expert_route
from backend.copilot.expert_route import resolve_expert_chat_route
from backend.copilot.transports import ChatTransportResponse

USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"


def _transport(auth_provider: str, credential_id: str | None) -> ChatTransportResponse:
    return ChatTransportResponse(
        auth_provider=auth_provider,  # type: ignore[arg-type]
        credential_id=credential_id,
        label=auth_provider,
        available=True,
        default=False,
    )


@pytest.fixture
def expert(mocker: pytest_mock.MockerFixture) -> MagicMock:
    """An expert pinned to ChatGPT, read through the db accessor."""
    row = MagicMock()
    row.llm_auth_provider = "codex"
    row.llm_credential_id = "cred-expert"
    db = MagicMock()
    db.get_expert = AsyncMock(return_value=row)
    mocker.patch.object(expert_route, "experts_db", return_value=db)
    return row


@pytest.fixture
def routes(mocker: pytest_mock.MockerFixture):
    pinned = mocker.patch.object(
        expert_route,
        "resolve_pinned_chat_route",
        new=AsyncMock(return_value=_transport("codex", "cred-expert")),
    )
    default = mocker.patch.object(
        expert_route,
        "resolve_default_chat_route",
        new=AsyncMock(return_value=("platform", None)),
    )
    return pinned, default


@pytest.mark.asyncio
async def test_an_experts_pin_wins_over_the_account_default(expert, routes) -> None:
    pinned, default = routes

    assert await resolve_expert_chat_route(USER_ID, "expert-1") == (
        "codex",
        "cred-expert",
    )
    pinned.assert_awaited_once_with(USER_ID, "codex", "cred-expert", unattended=True)
    default.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_pin_that_no_longer_resolves_falls_back_to_the_account_default(
    expert, routes
) -> None:
    pinned, default = routes
    pinned.return_value = None
    default.return_value = ("codex", "cred-account")

    assert await resolve_expert_chat_route(USER_ID, "expert-1") == (
        "codex",
        "cred-account",
    )
    default.assert_awaited_once_with(USER_ID)


@pytest.mark.asyncio
async def test_an_unpinned_expert_follows_the_account_default(expert, routes) -> None:
    expert.llm_auth_provider = None
    expert.llm_credential_id = None
    pinned, default = routes
    pinned.return_value = None

    assert await resolve_expert_chat_route(USER_ID, "expert-1") == ("platform", None)
    pinned.assert_awaited_once_with(USER_ID, None, None, unattended=True)


@pytest.mark.asyncio
async def test_a_missing_expert_follows_the_account_default(
    mocker: pytest_mock.MockerFixture, routes
) -> None:
    db = MagicMock()
    db.get_expert = AsyncMock(return_value=None)
    mocker.patch.object(expert_route, "experts_db", return_value=db)
    pinned, default = routes
    pinned.return_value = None

    assert await resolve_expert_chat_route(USER_ID, "expert-gone") == (
        "platform",
        None,
    )
    pinned.assert_awaited_once_with(USER_ID, None, None, unattended=True)


@pytest.mark.asyncio
async def test_a_broken_expert_lookup_never_fails_the_turn(
    mocker: pytest_mock.MockerFixture, routes
) -> None:
    db = MagicMock()
    db.get_expert = AsyncMock(side_effect=RuntimeError("db manager down"))
    mocker.patch.object(expert_route, "experts_db", return_value=db)
    pinned, default = routes
    pinned.return_value = None

    assert await resolve_expert_chat_route(USER_ID, "expert-1") == ("platform", None)
