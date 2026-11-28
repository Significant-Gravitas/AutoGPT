"""What a run's child answers, built in the test's own process. The run parent and
its children are tested end to end in ``direct_block_execution_test.py``."""

import importlib
import json
import pickle
import sys
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
    count: int = pydantic.Field(alias="readingCount")


class _ReturnsAModel:
    async def execute(self, input_data, **kwargs):
        taken_at = datetime(2026, 11, 24, tzinfo=timezone.utc)
        yield "reading", _Reading.model_validate(
            {"taken_at": taken_at, "readingCount": 2**70}
        )


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
    assert body == {
        "reading": [{"taken_at": "2026-11-24T00:00:00Z", "readingCount": 2**70}]
    }


@pytest.mark.asyncio
async def test_a_block_that_ran_out_of_memory_gets_the_memory_refusal():
    kind, body = await _answer_to(("out-of-memory", {}, {}))

    assert kind == FAILED
    assert body == {
        "message": "The block used more than 1024 MB of memory.",
        "value_error": True,
    }


def test_the_module_imports_where_there_is_no_resource_module(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "resource", None)
    try:
        importlib.reload(direct_block_run_parent)
    finally:
        monkeypatch.undo()
        importlib.reload(direct_block_run_parent)


async def _answer_to(request: tuple) -> tuple[bytes, object]:
    answer = await _answer(pickle.dumps(request))
    kind, size = FRAME_HEADER.unpack(answer[: FRAME_HEADER.size])
    return kind, json.loads(answer[FRAME_HEADER.size :][:size])
