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
async def test_new_user_response_reports_creation():
    now = datetime(2020, 1, 1, tzinfo=timezone.utc)
    user = User(
        id="user-new",
        email="new@example.com",
        created_at=now,
        updated_at=now,
    )
    response = Response()

    with patch(
        "backend.api.features.v1.get_or_create_user_with_status",
        new=AsyncMock(return_value=UserCreationResult(user=user, was_created=True)),
    ):
        body = await get_or_create_user_route(
            response,
            user_data={"sub": user.id, "email": user.email},
        )

    assert response.headers["X-AutoGPT-User-Created"] == "true"
    assert body == user.model_dump()
