"""The Mem0 blocks against the mem0ai 1.x client's calling conventions."""

from typing import Any, cast
from unittest.mock import patch

import pytest

from backend.blocks.mem0 import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    AddMemoryBlock,
    GetLatestMemoryBlock,
    SearchMemoryBlock,
)

_CREDENTIALS = cast(Any, TEST_CREDENTIALS_INPUT)
_RUN = {
    "credentials": TEST_CREDENTIALS,
    "user_id": "user-1",
    "graph_id": "graph-1",
    "graph_exec_id": "exec-1",
}


class _Client:
    """Answers like mem0ai 1.x and records how it was called."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict[str, Any]]] = []

    def add(self, *args, **kwargs):
        self.calls.append(("add", args, kwargs))
        return {"results": [{"event": "ADD", "memory": "is a vegetarian"}]}

    def search(self, *args, **kwargs):
        self.calls.append(("search", args, kwargs))
        return {"results": [{"id": "m-1", "memory": "is a vegetarian"}]}

    def get_all(self, *args, **kwargs):
        self.calls.append(("get_all", args, kwargs))
        return {"results": [{"id": "m-2"}, {"id": "m-1"}]}


async def _outputs(block, input_data) -> dict[str, Any]:
    return {name: value async for name, value in block.run(input_data, **_RUN)}


@pytest.mark.asyncio
async def test_add_waits_for_the_memory_instead_of_queueing_it():
    client = _Client()
    block = AddMemoryBlock()
    input_data = AddMemoryBlock.Input(
        content={"discriminator": "content", "content": "I am a vegetarian"},
        credentials=_CREDENTIALS,
    )
    with patch.object(AddMemoryBlock, "_get_client", return_value=client):
        outputs = await _outputs(block, input_data)

    assert client.calls[0][2]["async_mode"] is False
    assert outputs["memory"] == "is a vegetarian"


@pytest.mark.asyncio
async def test_search_outputs_the_list_inside_results():
    client = _Client()
    block = SearchMemoryBlock()
    input_data = SearchMemoryBlock.Input(query="food", credentials=_CREDENTIALS)
    with patch.object(SearchMemoryBlock, "_get_client", return_value=client):
        outputs = await _outputs(block, input_data)

    assert outputs["memories"] == [{"id": "m-1", "memory": "is a vegetarian"}]
    assert "version" not in client.calls[0][2]


@pytest.mark.asyncio
async def test_latest_memory_is_the_first_of_results():
    client = _Client()
    block = GetLatestMemoryBlock()
    input_data = GetLatestMemoryBlock.Input(credentials=_CREDENTIALS)
    with patch.object(GetLatestMemoryBlock, "_get_client", return_value=client):
        outputs = await _outputs(block, input_data)

    assert outputs["memory"] == {"id": "m-2"}
    assert outputs["found"] is True
