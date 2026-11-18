from unittest.mock import AsyncMock, patch

import pytest

from backend.notifications.dedupe import claim_once, is_claimed


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_open", [True, False])
async def test_a_redis_outage_follows_the_callers_choice(fail_open):
    """Open by default, so a one-off billing notice still goes; closed for a
    claim a scheduled pass retries, so an outage cannot send it every run."""
    with patch(
        "backend.notifications.dedupe.get_redis_async",
        AsyncMock(side_effect=RuntimeError("redis down")),
    ):
        assert await claim_once("trial:t1:last_call:1", fail_open=fail_open) is (
            fail_open
        )


@pytest.mark.asyncio
async def test_claim_lookup_answers_no_when_redis_is_down():
    """Only an early exit: claim_once still decides."""
    with patch(
        "backend.notifications.dedupe.get_redis_async",
        AsyncMock(side_effect=RuntimeError("redis down")),
    ):
        assert await is_claimed("trial:t1:last_call:1") is False
