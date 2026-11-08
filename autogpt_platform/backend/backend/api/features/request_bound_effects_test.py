"""The writes and routes that enforce a bound, driven through the caller.

Each test here covers a check whose own helper is tested elsewhere: what it
pins is that the write or route actually calls it, and what a client gets back
when it refuses.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import fastapi
import pydantic
import pytest
from autogpt_libs.auth.models import RequestContext

from backend.api.features.chat.routes import (
    MAX_CONTEXT_ENTRIES,
    QueuePendingMessageRequest,
    StreamChatRequest,
)
from backend.api.features.library.db import MAX_PRESET_NAME_LENGTH
from backend.api.features.orgs.invitation_routes import create_invitation
from backend.util.exceptions import InvalidInputError

ORG_ID = "org-1"
TEAM_ID = "team-1"
USER_ID = "user-1"


@pytest.mark.asyncio
async def test_updating_a_preset_refuses_an_oversized_name(mocker):
    from backend.api.features.library import db as library_db

    mocker.patch.object(
        library_db,
        "get_preset",
        AsyncMock(return_value=MagicMock(expert_id=None, name="before")),
    )

    with pytest.raises(InvalidInputError, match="Preset name is too long"):
        await library_db.update_preset(
            user_id=USER_ID,
            preset_id="preset-1",
            name="x" * (MAX_PRESET_NAME_LENGTH + 1),
        )


@pytest.mark.asyncio
async def test_creating_credentials_answers_400_when_the_store_refuses(mocker):
    from backend.api.features.integrations import router

    mocker.patch.object(
        router.creds_manager,
        "create",
        AsyncMock(side_effect=InvalidInputError("too large")),
    )

    with pytest.raises(fastapi.HTTPException) as caught:
        await router.create_credentials(
            user_id=USER_ID,
            provider=router.ProviderName.GITHUB,
            credentials=_api_key_credentials(),
        )

    assert caught.value.status_code == 400
    assert "too large" in str(caught.value.detail)


@pytest.mark.asyncio
async def test_the_external_api_answers_400_when_the_store_refuses(mocker):
    from backend.api.external.v1 import integrations as external_integrations

    mocker.patch.object(
        external_integrations.creds_manager,
        "create",
        AsyncMock(side_effect=InvalidInputError("too large")),
    )

    with pytest.raises(fastapi.HTTPException) as caught:
        await external_integrations.create_credential(
            provider="github",
            request=external_integrations.CreateAPIKeyCredentialRequest(
                type="api_key", api_key="a-key", title="a title"
            ),
            auth=MagicMock(user_id=USER_ID),
        )

    assert caught.value.status_code == 400
    assert "too large" in str(caught.value.detail)


@pytest.mark.asyncio
async def test_an_invitation_stores_each_team_once(mocker):
    from backend.api.features.orgs import invitation_routes

    prisma_client = MagicMock()
    prisma_client.team.find_many = AsyncMock(
        return_value=[SimpleNamespace(id=TEAM_ID, orgId=ORG_ID, archivedAt=None)]
    )
    prisma_client.orginvitation.find_first = AsyncMock(return_value=None)
    prisma_client.orgmember.find_first = AsyncMock(return_value=None)
    prisma_client.orginvitation.create = AsyncMock(
        return_value=MagicMock(teamIds=[TEAM_ID])
    )
    mocker.patch.object(invitation_routes, "prisma", prisma_client)
    mocker.patch.object(
        invitation_routes.InvitationCreateResponse,
        "from_db",
        MagicMock(return_value=MagicMock()),
    )

    await create_invitation(
        org_id=ORG_ID,
        request=invitation_routes.CreateInvitationRequest(
            email="someone@example.com", team_ids=[TEAM_ID, TEAM_ID, TEAM_ID]
        ),
        ctx=_owner_context(),
    )

    stored = prisma_client.orginvitation.create.await_args.kwargs["data"]["teamIds"]
    assert stored == [TEAM_ID]


@pytest.mark.parametrize("model", [StreamChatRequest, QueuePendingMessageRequest])
def test_a_chat_turn_refuses_more_context_entries_than_the_cap(model):
    too_many = {
        f"https://example.com/{n}": "page" for n in range(MAX_CONTEXT_ENTRIES + 1)
    }

    with pytest.raises(pydantic.ValidationError):
        model(message="hello", context=too_many)

    at_the_cap = {
        f"https://example.com/{n}": "page" for n in range(MAX_CONTEXT_ENTRIES)
    }
    assert model(message="hello", context=at_the_cap).context == at_the_cap


def _api_key_credentials():
    from pydantic import SecretStr

    from backend.data.model import APIKeyCredentials

    return APIKeyCredentials(
        provider="github", api_key=SecretStr("a-key"), title="a title"
    )


def _owner_context() -> RequestContext:
    return RequestContext(
        user_id=USER_ID,
        org_id=ORG_ID,
        team_id=None,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=True,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )
