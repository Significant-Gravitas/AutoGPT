"""Wait for Webset: the timed-out flag after the last poll."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.exa import websets_polling
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
