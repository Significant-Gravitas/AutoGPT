"""``is_auth_session_live`` against the real database."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from prisma.models import AuthSession, AuthUser

from backend.data.auth.session import is_auth_session_live
from backend.util.test import SpinTestServer


async def _auth_user() -> str:
    user_id = str(uuid4())
    await AuthUser.prisma().create(
        data={"id": user_id, "name": "Session test", "email": f"{user_id}@x.com"}
    )
    return user_id


async def _session(user_id: str, expires_in: timedelta) -> str:
    session_id = str(uuid4())
    await AuthSession.prisma().create(
        data={
            "id": session_id,
            "token": str(uuid4()),
            "userId": user_id,
            "expiresAt": datetime.now(timezone.utc) + expires_in,
        }
    )
    return session_id


@pytest.mark.asyncio(loop_scope="session")
async def test_only_an_unexpired_session_of_the_same_user_is_live(
    server: SpinTestServer,
):
    owner = await _auth_user()
    other = await _auth_user()
    try:
        live = await _session(owner, timedelta(days=1))
        expired = await _session(owner, -timedelta(minutes=1))
        deleted = await _session(owner, timedelta(days=1))
        await AuthSession.prisma().delete(where={"id": deleted})

        assert await is_auth_session_live(live, owner)
        assert not await is_auth_session_live(expired, owner)
        assert not await is_auth_session_live(deleted, owner)
        assert not await is_auth_session_live(live, other)
        assert not await is_auth_session_live(str(uuid4()), owner)
    finally:
        await AuthUser.prisma().delete_many(where={"id": {"in": [owner, other]}})
