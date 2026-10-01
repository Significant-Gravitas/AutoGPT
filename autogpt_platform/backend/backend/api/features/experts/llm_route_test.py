"""How an expert's pinned AI connection reads back to its owner."""

from unittest.mock import AsyncMock

import pytest
import pytest_mock

from backend.api.features.experts import llm_route
from backend.api.features.experts.llm_route import (
    annotate_llm_routes,
    known_auth_provider,
)
from backend.api.features.experts.models import PROTECTED_SOUL_RULES, Expert
from backend.copilot.transports import ChatTransportResponse

USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"


def _expert(**overrides) -> Expert:
    values = {
        "id": "expert-1",
        "name": "Maria",
        "avatar_url": None,
        "role": "Marketing",
        "tagline": None,
        "bio": None,
        "skills": [],
        "identity": "You are Maria.",
        "voice_preferences": "",
        "boundaries": "",
        "protected_soul_rules": list(PROTECTED_SOUL_RULES),
        "is_template": False,
        "source_template_id": None,
        "is_archived": False,
        "workflows": [],
    }
    values.update(overrides)
    return Expert(**values)


def _transports(*codex_credential_ids: str) -> list[ChatTransportResponse]:
    return [
        ChatTransportResponse(
            auth_provider="platform",
            credential_id=None,
            label="AutoGPT Platform",
            available=True,
            default=True,
        ),
        *(
            ChatTransportResponse(
                auth_provider="codex",
                credential_id=credential_id,
                label="ChatGPT",
                available=True,
                default=False,
            )
            for credential_id in codex_credential_ids
        ),
    ]


def test_known_auth_provider_accepts_every_route_this_server_can_run() -> None:
    assert known_auth_provider("platform") == "platform"
    assert known_auth_provider("codex") == "codex"
    assert known_auth_provider("microsoft_365_copilot") == "microsoft_365_copilot"


def test_known_auth_provider_reads_anything_else_as_unpinned() -> None:
    assert known_auth_provider(None) is None
    assert known_auth_provider("gemini") is None


@pytest.mark.asyncio
async def test_unpinned_experts_cost_no_lookup(
    mocker: pytest_mock.MockerFixture,
) -> None:
    lookup = mocker.patch.object(llm_route, "get_chat_transports", new=AsyncMock())
    experts = [_expert(), _expert(id="expert-2")]

    assert await annotate_llm_routes(USER_ID, experts) == experts
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_live_pin_carries_the_transports_label(
    mocker: pytest_mock.MockerFixture,
) -> None:
    lookup = mocker.patch.object(
        llm_route,
        "get_chat_transports",
        new=AsyncMock(return_value=_transports("cred-1", "cred-2")),
    )
    experts = [
        _expert(llm_auth_provider="codex", llm_credential_id="cred-1"),
        _expert(id="expert-2", llm_auth_provider="codex", llm_credential_id="cred-2"),
        _expert(id="expert-3"),
    ]

    pinned_one, pinned_two, unpinned = await annotate_llm_routes(USER_ID, experts)

    assert (pinned_one.llm_route_label, pinned_one.llm_route_available) == (
        "ChatGPT",
        True,
    )
    assert pinned_two.llm_route_available is True
    assert (unpinned.llm_route_label, unpinned.llm_route_available) == (None, True)
    # One lookup for the whole list, not one per expert.
    lookup.assert_awaited_once_with(USER_ID)


@pytest.mark.asyncio
async def test_a_pin_to_an_unlinked_account_reads_as_missing_by_name(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(
        llm_route,
        "get_chat_transports",
        new=AsyncMock(return_value=_transports("cred-other")),
    )

    (expert,) = await annotate_llm_routes(
        USER_ID, [_expert(llm_auth_provider="codex", llm_credential_id="cred-gone")]
    )

    assert expert.llm_route_available is False
    # Still named, so the warning can say which connection to relink.
    assert expert.llm_route_label == "ChatGPT"
    # The pin itself is left for the owner to see and change.
    assert (expert.llm_auth_provider, expert.llm_credential_id) == (
        "codex",
        "cred-gone",
    )


@pytest.mark.asyncio
async def test_a_failed_lookup_does_not_raise_a_warning_on_every_expert(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(
        llm_route,
        "get_chat_transports",
        new=AsyncMock(side_effect=RuntimeError("credential store down")),
    )
    pinned = _expert(
        llm_auth_provider="codex", llm_credential_id="cred-1", llm_route_label="ChatGPT"
    )

    assert await annotate_llm_routes(USER_ID, [pinned]) == [pinned]
