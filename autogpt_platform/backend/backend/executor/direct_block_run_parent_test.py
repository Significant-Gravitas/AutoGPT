"""What a run's child answers, built in the test's own process. The run parent and
its children are tested end to end in ``direct_block_execution_test.py``."""

import asyncio
import json
import pickle
from datetime import datetime, timezone

import pydantic
import pytest

from backend.executor import direct_block_run_parent
from backend.executor.direct_block_run_parent import (
    FAILED,
    FRAME_HEADER,
    OUTPUTS,
    _answer,
)
from backend.util.exceptions import BlockUnknownError


class _Reading(pydantic.BaseModel):
    taken_at: datetime
    count: int


class _ReturnsAModel:
    async def execute(self, input_data, **kwargs):
        taken_at = datetime(2026, 11, 24, tzinfo=timezone.utc)
        yield "reading", _Reading(taken_at=taken_at, count=2**70)


class _RunsOutOfMemory:
    """Fails as ``Block.execute`` reports a MemoryError: wrapped in its own error."""

    async def execute(self, input_data, **kwargs):
        try:
            raise MemoryError()
        except MemoryError as e:
            raise BlockUnknownError(message="", block_name="b", block_id="b") from e
        yield "never", None


@pytest.fixture(autouse=True)
def blocks(monkeypatch: pytest.MonkeyPatch):
    blocks = {"model": _ReturnsAModel, "out-of-memory": _RunsOutOfMemory}
    monkeypatch.setattr(direct_block_run_parent, "get_blocks", lambda: blocks)


@pytest.mark.asyncio
async def test_outputs_are_encoded_as_the_route_encodes_its_response():
    kind, body = await _answer_to(("model", {}, {}))

    assert kind == OUTPUTS
    assert body == {"reading": [{"taken_at": "2026-11-24T00:00:00Z", "count": 2**70}]}


@pytest.mark.asyncio
async def test_a_block_that_ran_out_of_memory_gets_the_memory_refusal():
    kind, body = await _answer_to(("out-of-memory", {}, {}))

    assert kind == FAILED
    assert body == {
        "message": "The block used more than 1024 MB of memory.",
        "value_error": True,
    }


async def _answer_to(request: tuple) -> tuple[bytes, object]:
    # _answer runs its own event loop, so off the test's.
    answer = await asyncio.to_thread(_answer, pickle.dumps(request))
    kind, size = FRAME_HEADER.unpack(answer[: FRAME_HEADER.size])
    return kind, json.loads(answer[FRAME_HEADER.size :][:size])
