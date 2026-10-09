from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.errors import ClientNotConnectedError

from backend.api.features.onboarding import routes
from backend.data.onboarding_role import OnboardingRole, UserOnboarding
from backend.data.understanding import BusinessUnderstandingInput


@pytest.mark.asyncio
async def test_profile_routes_role_writes_through_rpc_without_prisma(mocker):
    rpc = MagicMock()
    rpc.save_onboarding_role = AsyncMock()
    rpc.queue_onboarding_role = AsyncMock()
    mocker.patch("backend.data.db.is_connected", return_value=False)
    mocker.patch(
        "backend.util.clients.get_database_manager_async_client", return_value=rpc
    )
    local_write = mocker.patch.object(
        type(UserOnboarding.prisma()),
        "upsert",
        new=AsyncMock(side_effect=ClientNotConnectedError()),
    )
    mocker.patch.object(
        routes,
        "extract_business_understanding",
        new=AsyncMock(return_value=BusinessUnderstandingInput.model_construct()),
    )
    mocker.patch.object(routes, "upsert_business_understanding", new=AsyncMock())
    mocker.patch.object(routes, "set_onboarding_role")

    result = await routes.submit_onboarding_profile(
        routes.OnboardingProfileRequest(user_name="Nick", user_role="Marketing"),
        "authenticated-user",
    )

    assert result == {"status": "ok"}
    role = OnboardingRole(choice="Marketing")
    rpc.save_onboarding_role.assert_awaited_once_with("authenticated-user", role)
    rpc.queue_onboarding_role.assert_awaited_once_with("authenticated-user", role)
    local_write.assert_not_awaited()
