from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from autogpt_libs.auth import OrgAction, TeamAction
from autogpt_libs.auth.models import RequestContext

from backend.api.features.chat.share import _live_chat_share_action


@pytest.mark.asyncio
async def test_owner_share_uses_the_owner_filtering_session_lookup(mocker):
    lookup = mocker.patch(
        "backend.api.features.chat.share.get_chat_session_metadata",
        autospec=True,
        return_value=SimpleNamespace(
            organization_id="org-1",
            team_id="team-1",
        ),
    )

    @asynccontextmanager
    async def allowed(*_args, **_kwargs):
        yield True

    mocker.patch(
        "backend.api.features.chat.share.live_resource_permission_barrier",
        new=allowed,
    )
    mocker.patch(
        "backend.api.features.chat.share.require_exact_chat_session_scope",
        new_callable=AsyncMock,
    )
    ctx = RequestContext(
        user_id="user-1",
        org_id="org-1",
        team_id="team-1",
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=True,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )

    async with _live_chat_share_action(
        "session-1",
        "user-1",
        ctx,
        OrgAction.VIEW_RESOURCES,
        TeamAction.VIEW_AGENTS,
    ) as scope:
        assert scope == ("org-1", "team-1")

    lookup.assert_awaited_once_with("session-1", "user-1")
