"""Real-database coverage for a user's first workspace: the calls that race
to create it all get one row, and later calls read that row."""

import asyncio
import uuid

import prisma.models
import pytest

from backend.data.user import get_or_create_user
from backend.data.workspace import get_or_create_workspace
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_first_calls_racing_to_create_a_workspace_all_get_one(
    server: SpinTestServer,
):
    user_id = await _user()

    workspaces = await asyncio.gather(
        *(get_or_create_workspace(user_id) for _ in range(5))
    )

    assert len({workspace.id for workspace in workspaces}) == 1
    assert (
        await prisma.models.UserWorkspace.prisma().count(where={"userId": user_id}) == 1
    )
    later = await get_or_create_workspace(user_id)
    assert later.id == workspaces[0].id


async def _user() -> str:
    suffix = uuid.uuid4().hex[:8]
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"first-workspace-{suffix}@example.com",
            "name": "First Caller",
        }
    )
    return user.id
