"""Every Exa block's own timeout input fits inside the executor's cap on run()."""

from unittest.mock import AsyncMock, patch

import pytest
from annotated_types import Le

from backend.blocks._base import Block
from backend.blocks.exa._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.exa._timeouts import WAIT_SLACK_SECONDS
from backend.blocks.exa.agent import ExaAgentRunBlock
from backend.blocks.exa.agent_fixtures import SAMPLE_QUEUED_RUN
from backend.blocks.exa.exa_platform_credentials_test import ALL_BLOCKS
from backend.blocks.exa.research import ExaCreateResearchBlock

# Inputs for how long a block waits, in seconds. (livecrawl_timeout is a
# per-page limit in milliseconds that Exa enforces, not a wait in run().)
WAIT_INPUTS = {"timeout", "polling_timeout", "wait_timeout"}


def _timeout_ceilings(block: Block) -> dict[str, int]:
    ceilings = {}
    for name, field in block.input_schema.model_fields.items():
        if name not in WAIT_INPUTS:
            continue
        for meta in field.metadata:
            if isinstance(meta, Le):
                ceilings[name] = int(meta.le)
    return ceilings


WAITING_BLOCKS = [cls for cls in ALL_BLOCKS if _timeout_ceilings(cls())]


def test_the_waiting_blocks_are_found():
    names = {cls.__name__ for cls in WAITING_BLOCKS}
    assert {
        "ExaAgentRunBlock",
        "ExaWaitForAgentRunBlock",
        "ExaCreateResearchBlock",
        "ExaWaitForResearchBlock",
        "ExaWaitForWebsetBlock",
        "ExaCreateWebsetBlock",
    } <= names


@pytest.mark.parametrize("block_cls", WAITING_BLOCKS, ids=lambda c: c.__name__)
def test_the_longest_timeout_input_finishes_before_the_executor_cap(block_cls):
    # Otherwise asyncio.wait_for cancels run() before the block's own timeout
    # output (or its timeout error naming the run) fires.
    block = block_cls()
    cap = block.execution_timeout_seconds
    assert cap is not None
    for name, ceiling in _timeout_ceilings(block).items():
        assert ceiling + WAIT_SLACK_SECONDS <= cap, name


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "block_cls, id_output, inputs",
    [
        (ExaAgentRunBlock, "run_id", {"query": "q"}),
        (ExaCreateResearchBlock, "research_id", {"instructions": "q"}),
    ],
)
async def test_a_started_run_hands_over_its_id_before_waiting(
    block_cls, id_output, inputs
):
    block = block_cls()
    outputs = []
    with (
        patch.object(block, "create_run", AsyncMock(return_value=SAMPLE_QUEUED_RUN)),
        patch.object(block, "wait_for_run", AsyncMock(side_effect=TimeoutError)),
        pytest.raises(TimeoutError),
    ):
        async for name, value in block.run(
            block.input_schema(credentials=TEST_CREDENTIALS_INPUT, **inputs),
            credentials=TEST_CREDENTIALS,
            user_id="user-a",
        ):
            outputs.append((name, value))

    assert outputs == [(id_output, SAMPLE_QUEUED_RUN["id"])]
