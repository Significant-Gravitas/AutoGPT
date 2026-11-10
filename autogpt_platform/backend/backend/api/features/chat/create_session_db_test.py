"""A new chat, created against a real database.

``POST /sessions`` looks the expert up and resolves the connection side by
side, then creates the session, or adopts it when a retry names the same id.
Every input here is a real row (the user's plan, their stored ChatGPT
connection and saved default, the expert and who owns it) and so is the
session it answers with.
"""

from uuid import UUID, uuid4

import pytest
from autogpt_libs.auth.models import RequestContext
from fastapi import HTTPException
from prisma.enums import SubscriptionTier
from prisma.models import ChatSession, Expert, User
from pytest_mock import MockerFixture

from backend.api.features.chat import routes as chat_routes
from backend.api.features.chat.routes import CreateSessionRequest, create_session
from backend.api.features.orgs.db import get_user_default_team
from backend.copilot import transports
from backend.copilot.transports_test import _codex_credentials
from backend.data.user import get_or_create_user, set_user_default_chat_route
from backend.util import entitlements
from backend.util.settings import BehaveAs
from backend.util.test import SpinTestServer

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.fixture(autouse=True)
def hosted(mocker: MockerFixture) -> None:
    """The hosted platform: ChatGPT takes a Max plan, read from the user's row."""
    for settings in (chat_routes.settings, transports.settings, entitlements.settings):
        mocker.patch.object(settings.config, "behave_as", BehaveAs.CLOUD)


async def test_a_new_chat_starts_on_the_saved_connection_and_a_retry_adopts_it(
    server: SpinTestServer,
) -> None:
    user_id, ctx = await _user(SubscriptionTier.MAX)
    credential_id = await _chatgpt_connection(user_id)
    await set_user_default_chat_route(user_id, "codex", credential_id)
    request = CreateSessionRequest(session_id=uuid4())

    first = await create_session(user_id, ctx, request)
    retry = await create_session(user_id, ctx, request)

    assert retry.id == first.id == str(request.session_id)
    for response in (first, retry):
        assert response.metadata.llm_auth_provider == "codex"
        assert response.metadata.llm_credential_id == credential_id
    assert await _sessions(request.session_id) == 1


async def test_an_experts_new_chat_runs_as_the_owners_expert(
    server: SpinTestServer,
) -> None:
    user_id, ctx = await _user()
    expert_id = await _expert(user_id)
    request = CreateSessionRequest(expert_id=expert_id, session_id=uuid4())

    first = await create_session(user_id, ctx, request)
    retry = await create_session(user_id, ctx, request)

    assert retry.id == first.id == str(request.session_id)
    assert first.expert_id == retry.expert_id == expert_id
    assert first.metadata.llm_auth_provider == "platform"
    row = await ChatSession.prisma().find_unique_or_raise(where={"id": first.id})
    assert (row.userId, row.expertId) == (user_id, expert_id)


@pytest.mark.parametrize("expert", ["someone else's", "archived"])
async def test_an_expert_the_user_cannot_chat_with_is_not_found_before_the_plan(
    server: SpinTestServer, expert: str
) -> None:
    """Both lookups fail: the expert isn't the user's to chat with, and
    ChatGPT isn't on their plan. The expert's 404 answers, and no session is
    made."""
    user_id, ctx = await _user()
    owner = (await _user())[0] if expert == "someone else's" else user_id
    expert_id = await _expert(owner, archived=expert == "archived")
    request = _chatgpt_request(expert_id)

    with pytest.raises(HTTPException) as error:
        await create_session(user_id, ctx, request)

    assert (error.value.status_code, error.value.detail) == (404, "Expert not found")
    assert await _sessions(request.session_id) == 0


async def test_a_found_expert_still_gets_the_connections_refusal(
    server: SpinTestServer,
) -> None:
    """The expert is the user's, and ChatGPT isn't on their plan: the plan's
    refusal answers, and no session is made."""
    user_id, ctx = await _user()
    request = _chatgpt_request(await _expert(user_id))

    with pytest.raises(HTTPException) as error:
        await create_session(user_id, ctx, request)

    assert error.value.status_code == 402
    assert await _sessions(request.session_id) == 0


async def _user(
    tier: SubscriptionTier = SubscriptionTier.NO_TIER,
) -> tuple[str, RequestContext]:
    """A signed-up user on *tier*, with their personal workspace as context."""
    user_id = str(uuid4())
    await get_or_create_user({"sub": user_id, "email": f"{user_id}@example.com"})
    await User.prisma().update(where={"id": user_id}, data={"subscriptionTier": tier})
    org_id, team_id = await get_user_default_team(user_id)
    assert org_id is not None
    return user_id, RequestContext(
        user_id=user_id,
        org_id=org_id,
        team_id=team_id,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=True,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )


async def _chatgpt_connection(user_id: str) -> str:
    credentials = _codex_credentials(str(uuid4()))
    await transports.credentials_manager.store.add_creds(user_id, credentials)
    return credentials.id


async def _expert(owner_id: str, *, archived: bool = False) -> str:
    expert = await Expert.prisma().create(
        data={
            "ownerUserId": owner_id,
            "name": "Scout",
            "role": "Research",
            "identity": "Finds things out.",
            "isArchived": archived,
        }
    )
    return expert.id


def _chatgpt_request(expert_id: str) -> CreateSessionRequest:
    return CreateSessionRequest(
        expert_id=expert_id,
        session_id=uuid4(),
        llm_auth_provider="codex",
        llm_credential_id=str(uuid4()),
    )


async def _sessions(session_id: UUID | None) -> int:
    return await ChatSession.prisma().count(where={"id": str(session_id)})
