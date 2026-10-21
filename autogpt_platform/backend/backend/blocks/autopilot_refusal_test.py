"""AutoPilotBlock must treat a refused turn as terminal.

``allow_queue=False`` (autopilot.py) made ``refused`` reachable for this block:
``run_copilot_turn_via_queue`` returns it when the target session already has a
turn in flight, and when the tree ledger refuses the spawn. Without a branch for
it the block falls through and renders an empty ``SessionResult`` as a
successful turn, dropping the refusal.
"""

from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.autopilot import AutoPilotBlock
from backend.copilot.sdk.session_waiter import SessionResult


@pytest.mark.asyncio
async def test_a_refused_turn_raises_instead_of_returning_an_empty_success():
    refusal = "That session already has a turn in flight."

    with patch(
        "backend.copilot.sdk.session_waiter.run_copilot_turn_via_queue",
        new=AsyncMock(return_value=("refused", SessionResult(refusal=refusal))),
    ):
        with pytest.raises(RuntimeError, match="already has a turn in flight"):
            await AutoPilotBlock().execute_copilot(
                prompt="do the thing",
                system_context="",
                session_id="sess-busy",
                max_recursion_depth=3,
                user_id="u1",
            )


@pytest.mark.asyncio
async def test_a_refusal_with_no_message_still_raises():
    """``SessionResult.refusal`` is optional, so the fallback must not turn an
    empty refusal back into a silent success."""
    with patch(
        "backend.copilot.sdk.session_waiter.run_copilot_turn_via_queue",
        new=AsyncMock(return_value=("refused", SessionResult())),
    ):
        with pytest.raises(RuntimeError, match="refused"):
            await AutoPilotBlock().execute_copilot(
                prompt="do the thing",
                system_context="",
                session_id="sess-busy",
                max_recursion_depth=3,
                user_id="u1",
            )
