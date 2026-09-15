import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.api.features.local_executor.consent import (
    get_computer_use_consent,
    is_computer_use_approved,
    set_computer_use_consent,
)


@pytest.mark.asyncio
async def test_consent_is_scoped_to_owner_and_session() -> None:
    redis = AsyncMock()
    redis.get.return_value = json.dumps(
        {
            "state": "approved",
            "machine_id": "machine-1",
            "features_coarse": ["screenshot"],
            "features": [],
        }
    ).encode()

    with patch(
        "backend.api.features.local_executor.consent.get_redis_async",
        return_value=redis,
    ):
        assert (
            await get_computer_use_consent(
                "session-1",
                "user-1",
                machine_id="machine-1",
                features_coarse=["screenshot"],
                features=[],
            )
            == "approved"
        )

    redis.get.assert_awaited_once_with(
        "copilot:local-executor:computer-use-consent:user-1:session-1"
    )


@pytest.mark.asyncio
async def test_missing_user_and_redis_errors_fail_closed() -> None:
    assert await is_computer_use_approved("session-1", None) is False

    with patch(
        "backend.api.features.local_executor.consent.get_redis_async",
        side_effect=ConnectionError("redis unavailable"),
    ):
        assert await is_computer_use_approved("session-1", "user-1") is False


@pytest.mark.asyncio
async def test_explicit_denial_remains_denied() -> None:
    redis = AsyncMock()

    with patch(
        "backend.api.features.local_executor.consent.get_redis_async",
        return_value=redis,
    ):
        state = await set_computer_use_consent("session-1", "user-1", approved=False)

    assert state == "denied"
    redis.setex.assert_awaited_once_with(
        "copilot:local-executor:computer-use-consent:user-1:session-1",
        60 * 60 * 24 * 30,
        json.dumps(
            {
                "state": "denied",
                "machine_id": None,
                "features_coarse": [],
                "features": [],
            },
            separators=(",", ":"),
        ),
    )


@pytest.mark.asyncio
async def test_approval_is_invalidated_by_machine_or_feature_change() -> None:
    redis = AsyncMock()
    redis.get.return_value = json.dumps(
        {
            "state": "approved",
            "machine_id": "machine-a",
            "features_coarse": ["screenshot"],
            "features": [],
        }
    )

    with patch(
        "backend.api.features.local_executor.consent.get_redis_async",
        return_value=redis,
    ):
        assert await is_computer_use_approved(
            "session-1",
            "user-1",
            machine_id="machine-a",
            features_coarse=["screenshot"],
            features=[],
        )
        assert not await is_computer_use_approved(
            "session-1",
            "user-1",
            machine_id="machine-b",
            features_coarse=["screenshot"],
            features=[],
        )
        assert not await is_computer_use_approved(
            "session-1",
            "user-1",
            machine_id="machine-a",
            features_coarse=["screenshot", "input", "clipboard"],
            features=[],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored_coarse", "stored_fine", "current_coarse", "current_fine"),
    [
        (["input.click", "screenshot"], [], ["screenshot"], ["input.click"]),
        ([" input", "screenshot"], [], ["input", "screenshot"], []),
    ],
)
async def test_approval_preserves_feature_provenance_and_exact_values(
    stored_coarse: list[str],
    stored_fine: list[str],
    current_coarse: list[str],
    current_fine: list[str],
) -> None:
    redis = AsyncMock()
    redis.get.return_value = json.dumps(
        {
            "state": "approved",
            "machine_id": "machine-a",
            "features_coarse": stored_coarse,
            "features": stored_fine,
        }
    )

    with patch(
        "backend.api.features.local_executor.consent.get_redis_async",
        return_value=redis,
    ):
        assert not await is_computer_use_approved(
            "session-1",
            "user-1",
            machine_id="machine-a",
            features_coarse=current_coarse,
            features=current_fine,
        )


@pytest.mark.asyncio
async def test_approval_requires_a_connected_machine_scope() -> None:
    with pytest.raises(ValueError, match="connected machine"):
        await set_computer_use_consent(
            "session-1",
            "user-1",
            approved=True,
        )
