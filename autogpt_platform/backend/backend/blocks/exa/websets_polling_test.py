"""Wait for Webset: the timed-out flag after the last poll."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.exa import websets_polling
from backend.blocks.exa._test import TEST_CREDENTIALS
from backend.blocks.exa.websets_fixtures import (
    WEBSET_ID,
    collect_outputs,
    sample_webset,
)
from backend.blocks.exa.websets_polling import ExaWaitForWebsetBlock
from backend.blocks.exa.websets_types import ApiWebset


@pytest.mark.asyncio
@pytest.mark.parametrize("final_status, timed_out", [("paused", False), ("idle", True)])
async def test_the_final_poll_is_checked_against_the_target(final_status, timed_out):
    block = ExaWaitForWebsetBlock()
    # The loop sees "running" until the timeout passes; only the final fetch
    # can reach the target.
    get = AsyncMock(
        side_effect=[
            ApiWebset.model_validate(sample_webset(status="running")),
            ApiWebset.model_validate(sample_webset(status=final_status)),
        ]
    )
    clock = iter([0, 0, 0, 10, 10, 10, 10])
    with (
        patch.object(block, "_get_webset", get),
        patch.object(websets_polling, "sleep", AsyncMock()),
        patch.object(
            websets_polling, "time", SimpleNamespace(time=lambda: next(clock))
        ),
    ):
        outputs = dict(
            await collect_outputs(
                block, webset_id=WEBSET_ID, target_status="paused", timeout=5
            )
        )

    assert outputs["final_status"] == final_status
    assert outputs["timed_out"] is timed_out


@pytest.mark.asyncio
async def test_waiting_for_idle_backs_off_and_stops_at_the_deadline():
    block = ExaWaitForWebsetBlock()
    running = ApiWebset.model_validate(sample_webset(status="running"))
    get = AsyncMock(return_value=running)
    now = [0.0]
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    with (
        patch.object(block, "_get_webset", get),
        patch("backend.blocks.exa.websets_api.sleep", fake_sleep),
        patch("backend.blocks.exa.websets_api.monotonic", lambda: now[0]),
        pytest.raises(TimeoutError),
    ):
        await block._wait_until_idle(
            TEST_CREDENTIALS, WEBSET_ID, timeout=60, poll_interval=5, max_interval=10
        )

    # 5, then 1.5x up to the 10 s cap, and the last sleep is cut to the deadline.
    assert sleeps[:3] == [5, 7.5, 10]
    assert all(s <= 10 for s in sleeps)
    assert sum(sleeps) == 60
