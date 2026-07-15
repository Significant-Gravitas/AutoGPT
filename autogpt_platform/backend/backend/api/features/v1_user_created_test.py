import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Response

from backend.api.features.v1 import get_or_create_user_route
from backend.data.model import User
from backend.data.user import UserCreationResult


@pytest.fixture(autouse=True)
def graph_cleanup():
    """Keep this route unit test independent of the integration server."""
    yield


@pytest.mark.asyncio
async def test_new_user_response_reports_creation_and_starts_tally_population():
    now = datetime.now(timezone.utc)
    user = User(
        id="user-new",
        email="new@example.com",
        created_at=now,
        updated_at=now,
    )
    response = Response()

    with (
        patch(
            "backend.api.features.v1.get_or_create_user_with_status",
            new=AsyncMock(return_value=UserCreationResult(user=user, was_created=True)),
        ),
        patch(
            "backend.api.features.v1.populate_understanding_from_tally",
            new_callable=AsyncMock,
        ) as populate_from_tally,
    ):
        body = await get_or_create_user_route(
            response,
            user_data={"sub": user.id, "email": user.email},
        )
        await asyncio.sleep(0)

    assert response.headers["X-AutoGPT-User-Created"] == "true"
    assert body == user.model_dump()
    populate_from_tally.assert_awaited_once_with(user.id, user.email)
