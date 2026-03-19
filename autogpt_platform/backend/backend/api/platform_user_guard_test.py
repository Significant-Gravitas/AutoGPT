from unittest.mock import AsyncMock, Mock

import pytest
import pytest_mock

from backend.util.exceptions import NotAuthorizedError

from .platform_user_guard import ensure_platform_user, should_enforce_platform_user


def test_should_enforce_platform_user() -> None:
    assert should_enforce_platform_user("/api/credits")
    assert not should_enforce_platform_user("/metrics")
    assert not should_enforce_platform_user("/api/auth/check-invite")
    assert not should_enforce_platform_user("/api/auth/user")
    assert not should_enforce_platform_user("/api/public/shared/token")


@pytest.mark.asyncio
async def test_ensure_platform_user_skips_existing_user(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.platform_user_guard.get_user_by_id",
        new=AsyncMock(return_value=Mock()),
    )
    activate_user = mocker.patch(
        "backend.api.platform_user_guard.get_or_activate_user",
        new=AsyncMock(),
    )

    await ensure_platform_user({"sub": "user-1", "email": "user@example.com"})

    activate_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_platform_user_activates_missing_user(
    mocker: pytest_mock.MockerFixture,
) -> None:
    payload = {"sub": "user-1", "email": "user@example.com"}
    mocker.patch(
        "backend.api.platform_user_guard.get_user_by_id",
        new=AsyncMock(side_effect=ValueError("User not found")),
    )
    activate_user = mocker.patch(
        "backend.api.platform_user_guard.get_or_activate_user",
        new=AsyncMock(),
    )

    await ensure_platform_user(payload)

    activate_user.assert_awaited_once_with(payload)


@pytest.mark.asyncio
async def test_ensure_platform_user_propagates_activation_denials(
    mocker: pytest_mock.MockerFixture,
) -> None:
    payload = {"sub": "user-1", "email": "user@example.com"}
    mocker.patch(
        "backend.api.platform_user_guard.get_user_by_id",
        new=AsyncMock(side_effect=ValueError("User not found")),
    )
    mocker.patch(
        "backend.api.platform_user_guard.get_or_activate_user",
        new=AsyncMock(side_effect=NotAuthorizedError("Access denied")),
    )

    with pytest.raises(NotAuthorizedError, match="Access denied"):
        await ensure_platform_user(payload)
