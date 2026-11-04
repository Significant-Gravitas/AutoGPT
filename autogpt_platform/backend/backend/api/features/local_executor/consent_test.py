import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.api.features.local_executor.consent import (
    get_computer_use_consent,
    is_computer_use_approved,
    set_computer_use_consent,
)

_DRIVER = "cua-driver 0.38.0"
_TOOLS = ["click", "get_window_state", "type_text"]


def _approved(**overrides) -> str:
    return json.dumps(
        {
            "state": "approved",
            "machine_id": "machine-a",
            "driver": _DRIVER,
            "tools": _TOOLS,
            **overrides,
        }
    )


@pytest.mark.asyncio
async def test_consent_is_scoped_to_owner_and_session() -> None:
    redis = AsyncMock()
    redis.get.return_value = _approved(machine_id="machine-1").encode()

    with patch(
        "backend.api.features.local_executor.consent.get_redis_async",
        return_value=redis,
    ):
        assert (
            await get_computer_use_consent(
                "session-1",
                "user-1",
                machine_id="machine-1",
                driver=_DRIVER,
                tools=_TOOLS,
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
            {"state": "denied", "machine_id": None, "driver": "", "tools": []},
            separators=(",", ":"),
        ),
    )


@pytest.mark.asyncio
async def test_approval_stores_the_scope_it_covers() -> None:
    redis = AsyncMock()

    with patch(
        "backend.api.features.local_executor.consent.get_redis_async",
        return_value=redis,
    ):
        state = await set_computer_use_consent(
            "session-1",
            "user-1",
            approved=True,
            machine_id="machine-a",
            driver=_DRIVER,
            tools=["type_text", "click", "get_window_state", "click"],
        )

    assert state == "approved"
    stored = json.loads(redis.setex.await_args.args[2])
    assert stored == json.loads(_approved())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "current",
    [
        {"machine_id": "machine-b"},
        {"driver": "cua-driver 0.39.0"},
        {"tools": [*_TOOLS, "launch_app"]},
        {"tools": _TOOLS[:2]},
    ],
    ids=["machine", "driver", "more-tools", "fewer-tools"],
)
async def test_approval_is_invalidated_by_a_scope_change(current: dict) -> None:
    redis = AsyncMock()
    redis.get.return_value = _approved()
    scope = {"machine_id": "machine-a", "driver": _DRIVER, "tools": _TOOLS, **current}

    with patch(
        "backend.api.features.local_executor.consent.get_redis_async",
        return_value=redis,
    ):
        assert await is_computer_use_approved(
            "session-1",
            "user-1",
            machine_id="machine-a",
            driver=_DRIVER,
            tools=_TOOLS,
        )
        assert not await is_computer_use_approved("session-1", "user-1", **scope)


@pytest.mark.asyncio
async def test_an_approval_from_the_feature_era_asks_again() -> None:
    """Before cua-driver, consent covered feature lists; those records no
    longer describe what the user approved."""
    redis = AsyncMock()
    redis.get.return_value = json.dumps(
        {
            "state": "approved",
            "machine_id": "machine-a",
            "features_coarse": ["screenshot", "input"],
            "features": [],
        }
    )

    with patch(
        "backend.api.features.local_executor.consent.get_redis_async",
        return_value=redis,
    ):
        assert (
            await get_computer_use_consent(
                "session-1",
                "user-1",
                machine_id="machine-a",
                driver=_DRIVER,
                tools=_TOOLS,
            )
            == "pending"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope",
    [
        {},
        {"machine_id": "machine-a", "driver": _DRIVER},
        {"machine_id": "machine-a", "tools": _TOOLS},
        {"driver": _DRIVER, "tools": _TOOLS},
    ],
)
async def test_approval_requires_a_connected_machine_scope(scope: dict) -> None:
    with pytest.raises(ValueError, match="connected machine"):
        await set_computer_use_consent("session-1", "user-1", approved=True, **scope)
