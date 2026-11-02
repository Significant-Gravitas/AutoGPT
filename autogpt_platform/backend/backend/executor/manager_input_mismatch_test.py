"""A node that gets a linked value it can't use fails with an input error.

Before, the executor left such a node unqueued with only a log line, and the
run still finished COMPLETED without saying which block was skipped or why.
"""

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.llm import AITextGeneratorBlock
from backend.data.execution import ExecutionContext, ExecutionStatus, NodeExecutionEntry
from backend.data.graph import Link, Node
from backend.data.model import NodeExecutionStats
from backend.executor.manager import (
    ExecutionProcessor,
    _enqueue_next_nodes,
    execute_node,
)
from backend.executor.utils import (
    InputMismatchError,
    NodeExecutionProgress,
    validate_exec,
)
from backend.util.exceptions import BlockInputError

_MOD = "backend.executor.manager"
_BLOCK = AITextGeneratorBlock()
_CREDENTIALS = {"id": "cred-1", "provider": "openai", "type": "api_key", "title": None}
_BAD_MODEL = "not-a-model"


def _llm_node(*linked_pins: str, static: frozenset[str] = frozenset()) -> Node:
    return Node(
        id="llm-node",
        block_id=_BLOCK.id,
        input_default={"credentials": _CREDENTIALS},
        input_links=[
            Link(
                source_id="source-node",
                sink_id="llm-node",
                source_name=f"{pin}_out",
                sink_name=pin,
                is_static=pin in static,
            )
            for pin in linked_pins
        ],
    )


def _source_node(target: Node) -> Node:
    return Node(
        id="source-node",
        block_id=_BLOCK.id,
        output_links=list(target.input_links),
    )


# ---------------------------------------------------------------- validate_exec


def test_a_value_that_cannot_be_converted_is_an_input_error():
    data, error = validate_exec(
        _llm_node("model", "prompt"), {"model": _BAD_MODEL, "prompt": "hi"}
    )

    assert data is None
    assert "`model`" in error and _BAD_MODEL in error


def test_a_value_that_cannot_be_converted_raises_when_asked():
    with pytest.raises(InputMismatchError) as raised:
        validate_exec(
            _llm_node("model", "prompt"),
            {"model": _BAD_MODEL, "prompt": "hi"},
            raise_on_mismatch=True,
        )

    assert "`model`" in str(raised.value)
    # The resolved input the node is queued with, defaults included.
    assert raised.value.data["model"] == _BAD_MODEL
    assert raised.value.data["credentials"] == _CREDENTIALS


def test_a_value_that_does_not_match_the_schema_raises_when_asked():
    # Not converted (credentials aren't a plain type), and not an object.
    with pytest.raises(InputMismatchError, match="is not of type 'object'"):
        validate_exec(
            _llm_node("prompt", "credentials"),
            {"prompt": "hi", "credentials": "not-credentials"},
            raise_on_mismatch=True,
        )


@pytest.mark.parametrize(
    "node, data, expected",
    [
        # A linked input that hasn't arrived yet.
        (_llm_node("model", "prompt"), {"model": _BAD_MODEL}, "unpopulated links"),
        # A required input that nothing provides yet.
        (_llm_node("model"), {"model": _BAD_MODEL}, "missing input"),
    ],
    ids=["unpopulated link", "missing required input"],
)
def test_missing_input_is_reported_before_a_bad_value(
    node: Node, data: dict[str, Any], expected: str
):
    """A node still waiting for input keeps waiting, even when a value it
    already has can't be used: that is checked once the input is complete."""
    result, error = validate_exec(node, data, raise_on_mismatch=True)

    assert result is None
    assert expected in error


def test_a_list_into_a_text_input_is_still_converted():
    data, _ = validate_exec(
        _llm_node("prompt"), {"prompt": ["a", "b"]}, raise_on_mismatch=True
    )

    assert data is not None
    assert data["prompt"] == '["a", "b"]'


# ------------------------------------------------------------- queueing a node


@asynccontextmanager
async def _no_lock(_key: str):
    yield


def _db_client(target: Node, incomplete: list[Any] | None = None) -> MagicMock:
    stored: dict[str, Any] = {}

    async def upsert_execution_input(node_id, graph_exec_id, input_name, input_data):
        stored[input_name] = input_data
        return MagicMock(node_exec_id="llm-exec-1"), dict(stored)

    db = MagicMock()
    db.get_node = AsyncMock(return_value=target)
    db.upsert_execution_input = AsyncMock(side_effect=upsert_execution_input)
    db.get_latest_node_execution = AsyncMock(return_value=None)
    db.get_node_executions = AsyncMock(return_value=incomplete or [])
    return db


async def _route(
    target: Node, outputs: list[tuple[str, Any]], db: MagicMock
) -> list[NodeExecutionEntry]:
    source = _source_node(target)
    queued: list[NodeExecutionEntry] = []
    with (
        patch(f"{_MOD}.synchronized", new=_no_lock),
        patch(f"{_MOD}.send_async_execution_update", new=AsyncMock()),
        patch(f"{_MOD}.async_update_node_execution_status", new=AsyncMock()),
    ):
        for output in outputs:
            queued += await _enqueue_next_nodes(
                db_client=db,
                node=source,
                output=output,
                user_id="user-1",
                graph_exec_id="graph-exec-1",
                graph_id="graph-1",
                graph_version=1,
                log_metadata=MagicMock(),
                nodes_input_masks=None,
                execution_context=ExecutionContext(),
            )
    return queued


@pytest.mark.asyncio
async def test_a_node_given_a_value_it_cannot_use_is_queued_to_fail():
    target = _llm_node("model", "prompt")

    queued = await _route(
        target,
        [("model_out", _BAD_MODEL), ("prompt_out", "hi")],
        _db_client(target),
    )

    assert len(queued) == 1
    assert queued[0].node_id == "llm-node"
    assert queued[0].inputs["model"] == _BAD_MODEL


@pytest.mark.asyncio
async def test_a_node_still_missing_input_is_not_queued():
    target = _llm_node("model", "prompt")

    queued = await _route(target, [("model_out", _BAD_MODEL)], _db_client(target))

    assert queued == []


@pytest.mark.asyncio
async def test_a_node_waiting_on_a_static_link_is_queued_to_fail():
    """A value on a static link also completes the node's other executions that
    were waiting for it; each one that can't use it is queued to fail too."""
    target = _llm_node("model", "prompt", static=frozenset({"model"}))
    waiting = MagicMock(node_exec_id="llm-exec-0", input_data={"prompt": "earlier"})

    queued = await _route(
        target,
        [("prompt_out", "hi"), ("model_out", _BAD_MODEL)],
        _db_client(target, [waiting]),
    )

    assert [entry.node_exec_id for entry in queued] == ["llm-exec-1", "llm-exec-0"]
    assert all(entry.inputs["model"] == _BAD_MODEL for entry in queued)


# ------------------------------------------------------------- running the node


def _entry(inputs: dict[str, Any]) -> NodeExecutionEntry:
    return NodeExecutionEntry(
        user_id="user-1",
        graph_exec_id="graph-exec-1",
        graph_id="graph-1",
        graph_version=1,
        node_exec_id="llm-exec-1",
        node_id="llm-node",
        block_id=_BLOCK.id,
        inputs=inputs,
        execution_context=ExecutionContext(),
    )


_QUEUED_INPUT = {"credentials": _CREDENTIALS, "model": _BAD_MODEL, "prompt": "hi"}


@pytest.mark.asyncio
async def test_running_a_node_with_an_input_it_cannot_use_raises_the_input_error():
    with pytest.raises(BlockInputError, match="`model`") as raised:
        async for _ in execute_node(
            _llm_node("model", "prompt"),
            _entry(dict(_QUEUED_INPUT)),
            MagicMock(),
        ):
            pass

    assert _BAD_MODEL in str(raised.value)


@pytest.mark.asyncio
async def test_the_node_ends_failed_with_the_input_error():
    node = _llm_node("model", "prompt")
    db = MagicMock()
    db.upsert_execution_output = AsyncMock()
    db.get_node_execution = AsyncMock(return_value=None)
    processor = ExecutionProcessor()
    processor.creds_manager = MagicMock()

    with patch(f"{_MOD}.async_update_node_execution_status", new=AsyncMock()):
        _, status = await processor._on_node_execution(
            node=node,
            node_exec=_entry(dict(_QUEUED_INPUT)),
            node_exec_progress=NodeExecutionProgress(),
            stats=NodeExecutionStats(),
            db_client=db,
            log_metadata=MagicMock(),
        )

    assert status == ExecutionStatus.FAILED
    output = db.upsert_execution_output.await_args.kwargs
    assert output["output_name"] == "error"
    assert "`model`" in output["output_data"]
    assert _BAD_MODEL in output["output_data"]
