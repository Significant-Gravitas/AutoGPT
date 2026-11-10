"""A node that gets a linked value it can't use fails with an input error.

Before, the executor left such a node unqueued with only a log line, and the
run still finished COMPLETED without saying which block was skipped or why.
"""

import threading
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.ai_image_generator_block import AIImageGeneratorBlock, ImageGenModel
from backend.blocks.anysearch.search import AnySearchBlock
from backend.blocks.llm import AITextGeneratorBlock
from backend.blocks.orchestrator import OrchestratorBlock
from backend.data.execution import ExecutionContext, ExecutionStatus, NodeExecutionEntry
from backend.data.graph import Link, Node
from backend.data.model import GraphExecutionStats, NodeExecutionStats
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


def _image_node(*linked_pins: str, **input_default: Any) -> Node:
    """A block whose credentials are required, unlike the LLM node's defaults."""
    return Node(
        id="image-node",
        block_id=AIImageGeneratorBlock().id,
        input_default=input_default,
        input_links=[
            Link(
                source_id="source-node",
                sink_id="image-node",
                source_name=f"{pin}_out",
                sink_name=pin,
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


def test_a_long_bad_value_is_cut_short_in_the_error():
    value = "x" * 200

    _, error = validate_exec(
        _llm_node("model", "prompt"), {"model": value, "prompt": "hi"}
    )

    assert "..." in error and value not in error


@pytest.mark.parametrize(
    "other", [{}, {"model": _BAD_MODEL}], ids=["alone", "with a bad value"]
)
def test_a_required_input_that_arrives_as_none_is_skipped_as_before(
    other: dict[str, Any],
):
    """None is sometimes sent on purpose, so it doesn't fail the node, even
    when another input can't be converted."""
    data, error = validate_exec(
        _llm_node("prompt", *other), {"prompt": None, **other}, raise_on_mismatch=True
    )

    assert data is None
    assert error.startswith("Input data missing or mismatch")


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


@pytest.mark.parametrize(
    "extra",
    [{}, {"model": _BAD_MODEL}],
    ids=["usable input", "with a bad value"],
)
def test_a_node_waiting_for_a_tool_output_keeps_waiting(extra: dict[str, Any]):
    """The Orchestrator's missing-input check reads the converted conversation
    history, also when another value can't be converted: here a JSON string
    with a tool call that has no output yet."""
    block = OrchestratorBlock()
    node = Node(block_id=block.id, input_default={"prompt": "hi"})
    history = '[{"role": "assistant", "tool_calls": [{"id": "call-1"}]}]'

    data, error = validate_exec(
        node, {**extra, "conversation_history": history}, raise_on_mismatch=True
    )

    assert data is None
    assert "last_tool_output" in error


def test_a_dry_run_fails_on_a_value_it_cannot_convert():
    """A dry run checks the graph is wired right, so it reports the bad value."""
    with pytest.raises(InputMismatchError, match="`model`"):
        validate_exec(
            _llm_node("model", "prompt"),
            {"model": _BAD_MODEL, "prompt": "hi"},
            dry_run=True,
            raise_on_mismatch=True,
        )


def test_a_dry_run_fails_on_a_bad_value_for_a_block_that_needs_credentials():
    """A dry run fills in missing credentials with None; that placeholder isn't
    a None that arrived on purpose, so it doesn't turn the failure into a skip."""
    block = AIImageGeneratorBlock()
    node = Node(
        id="image-node",
        block_id=block.id,
        input_default={
            "credentials": {"id": "c", "provider": "replicate", "type": "api_key"},
            "prompt": "a cat",
        },
    )

    with pytest.raises(InputMismatchError, match="`model`"):
        validate_exec(node, {"model": _BAD_MODEL}, dry_run=True, raise_on_mismatch=True)


def test_a_dry_run_still_tolerates_a_schema_mismatch():
    data, _ = validate_exec(
        _llm_node("prompt", "credentials"),
        {"prompt": "hi", "credentials": "not-credentials"},
        dry_run=True,
        raise_on_mismatch=True,
    )

    assert data is not None


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
    target: Node,
    outputs: list[tuple[str, Any]],
    db: MagicMock,
    dry_run: bool = False,
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
                execution_context=ExecutionContext(dry_run=dry_run),
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
    # Marked, so the dispatcher doesn't bill a block that won't run.
    assert queued[0].input_error and "`model`" in queued[0].input_error


@pytest.mark.asyncio
async def test_a_node_with_usable_input_is_queued_unmarked():
    target = _llm_node("model", "prompt")

    queued = await _route(
        target,
        [("model_out", "gpt-4o"), ("prompt_out", "hi")],
        _db_client(target),
    )

    assert len(queued) == 1
    assert queued[0].input_error is None


@pytest.mark.asyncio
async def test_unused_credentials_do_not_mark_a_node_to_fail():
    """`execute_node` drops optional credentials the chosen auth doesn't use
    before validating, so queueing does the same: an anonymous search with an
    empty credentials object runs, and is billed, like any other node."""
    block = AnySearchBlock()
    target = Node(
        id="llm-node",
        block_id=block.id,
        input_default={"auth": "anonymous", "credentials": {}},
        input_links=[
            Link(
                source_id="source-node",
                sink_id="llm-node",
                source_name="query_out",
                sink_name="query",
            )
        ],
    )

    queued = await _route(target, [("query_out", "news")], _db_client(target))

    assert len(queued) == 1
    assert queued[0].input_error is None


@pytest.mark.asyncio
async def test_unused_credentials_stay_in_the_queued_input():
    """They are dropped only to validate. A static link passes the queued input
    on to the node's other executions, whose auth may need them."""
    block = AnySearchBlock()
    target = Node(
        id="llm-node",
        block_id=block.id,
        input_default={"auth": "anonymous"},
        input_links=[
            Link(
                source_id="source-node",
                sink_id="llm-node",
                source_name=f"{pin}_out",
                sink_name=pin,
                is_static=pin == "credentials",
            )
            for pin in ("query", "credentials")
        ],
    )

    queued = await _route(
        target,
        [("credentials_out", _CREDENTIALS), ("query_out", "news")],
        _db_client(target),
    )

    assert len(queued) == 1
    assert queued[0].input_error is None
    assert queued[0].inputs["credentials"] == _CREDENTIALS


@pytest.mark.asyncio
async def test_a_node_waits_for_a_linked_credentials_pin_it_does_not_use():
    """Credential clean-up doesn't fill in a linked pin that hasn't arrived, so
    the node waits for it and its later value pairs with this execution."""
    block = AnySearchBlock()
    target = Node(
        id="llm-node",
        block_id=block.id,
        input_default={"auth": "anonymous"},
        input_links=[
            Link(
                source_id="source-node",
                sink_id="llm-node",
                source_name=f"{pin}_out",
                sink_name=pin,
            )
            for pin in ("query", "credentials")
        ],
    )

    queued = await _route(target, [("query_out", "news")], _db_client(target))

    assert queued == []


@pytest.mark.asyncio
async def test_an_unhashable_auth_value_is_queued_to_fail():
    """Credential clean-up can't look a list up, so it leaves the value for
    validation to report instead of dropping the node."""
    block = AnySearchBlock()
    target = Node(
        id="llm-node",
        block_id=block.id,
        input_default={"query": "news"},
        input_links=[
            Link(
                source_id="source-node",
                sink_id="llm-node",
                source_name="auth_out",
                sink_name="auth",
            )
        ],
    )

    queued = await _route(target, [("auth_out", ["anonymous"])], _db_client(target))

    assert len(queued) == 1
    assert queued[0].input_error and "`auth`" in queued[0].input_error


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
    assert all(entry.input_error for entry in queued)


@pytest.mark.asyncio
async def test_a_static_link_leaves_an_execution_still_missing_input_waiting():
    target = _llm_node("model", "prompt", "sys_prompt", static=frozenset({"model"}))
    # Still has no `sys_prompt`, so the static `model` doesn't complete it.
    waiting = MagicMock(node_exec_id="llm-exec-0", input_data={"prompt": "earlier"})

    queued = await _route(
        target,
        [
            ("prompt_out", "hi"),
            ("sys_prompt_out", "be brief"),
            ("model_out", _BAD_MODEL),
        ],
        _db_client(target, [waiting]),
    )

    assert [entry.node_exec_id for entry in queued] == ["llm-exec-1"]


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
async def test_running_a_node_still_missing_input_yields_the_error_as_before():
    """Only a value the node can't use fails it. A node run without an input
    it needs (e.g. an Orchestrator tool call missing an argument) still
    yields the error as an output, so the caller gets the message."""
    outputs = [
        item
        async for item in execute_node(
            _llm_node("model"),
            _entry({"credentials": _CREDENTIALS, "model": "gpt-4o"}),
            MagicMock(),
        )
    ]

    assert len(outputs) == 1
    assert outputs[0][0] == "error"
    assert "missing input" in outputs[0][1]


async def _run_queued(
    node: Node, entry: NodeExecutionEntry
) -> tuple[ExecutionStatus, MagicMock]:
    db = MagicMock()
    db.upsert_execution_output = AsyncMock()
    db.get_node_execution = AsyncMock(return_value=None)
    processor = ExecutionProcessor()
    processor.creds_manager = MagicMock()

    with patch(f"{_MOD}.async_update_node_execution_status", new=AsyncMock()):
        _, status = await processor._on_node_execution(
            node=node,
            node_exec=entry,
            node_exec_progress=NodeExecutionProgress(),
            stats=NodeExecutionStats(),
            db_client=db,
            log_metadata=MagicMock(),
        )
    return status, db


@pytest.mark.asyncio
async def test_the_node_ends_failed_with_the_input_error():
    status, db = await _run_queued(
        _llm_node("model", "prompt"), _entry(dict(_QUEUED_INPUT))
    )

    assert status == ExecutionStatus.FAILED
    output = db.upsert_execution_output.await_args.kwargs
    assert output["output_name"] == "error"
    assert "`model`" in output["output_data"]
    assert _BAD_MODEL in output["output_data"]


_IMAGE_CREDENTIALS = {"id": "c", "provider": "replicate", "type": "api_key"}
_with_and_without_credentials = pytest.mark.parametrize(
    "configured",
    [{}, {"credentials": _IMAGE_CREDENTIALS}],
    ids=["no credentials set", "credentials set"],
)


@pytest.mark.asyncio
@_with_and_without_credentials
async def test_a_dry_run_of_a_block_needing_credentials_ends_failed_on_a_bad_value(
    configured: dict[str, Any],
):
    """A dry run queues required credentials as a None placeholder, and running
    the node validates that queued input again. The placeholder isn't a None
    that arrived on purpose, so the node still ends FAILED rather than skipped."""
    target = _image_node("model", prompt="a cat", **configured)

    queued = await _route(
        target, [("model_out", _BAD_MODEL)], _db_client(target), dry_run=True
    )

    assert len(queued) == 1
    assert queued[0].input_error and "`model`" in queued[0].input_error
    assert queued[0].inputs["credentials"] is None

    status, db = await _run_queued(target, queued[0])

    assert status == ExecutionStatus.FAILED
    output = db.upsert_execution_output.await_args.kwargs
    assert output["output_name"] == "error"
    assert "`model`" in output["output_data"]
    assert _BAD_MODEL in output["output_data"]


@pytest.mark.asyncio
@_with_and_without_credentials
async def test_a_dry_run_of_a_block_needing_credentials_still_simulates_good_input(
    configured: dict[str, Any],
):
    target = _image_node("model", prompt="a cat", **configured)

    queued = await _route(
        target,
        [("model_out", ImageGenModel.FLUX.value)],
        _db_client(target),
        dry_run=True,
    )

    assert len(queued) == 1
    assert queued[0].input_error is None

    async def simulated(*_args: Any, **_kwargs: Any):
        yield "image_url", "https://example.com/simulated.png"

    with patch(f"{_MOD}.simulate_block", new=simulated):
        status, db = await _run_queued(target, queued[0])

    assert status == ExecutionStatus.COMPLETED
    output = db.upsert_execution_output.await_args.kwargs
    assert output["output_name"] == "image_url"


@pytest.mark.asyncio
@pytest.mark.parametrize("pin", ["prompt", "credentials"])
async def test_a_dry_run_skips_a_required_input_that_arrives_as_none(pin: str):
    """Only the credentials placeholder is exempt: a None a linked block really
    sent still skips the node, even next to a value that can't be converted."""
    target = _image_node("model", pin, prompt="a cat")

    queued = await _route(
        target,
        [("model_out", _BAD_MODEL), (f"{pin}_out", None)],
        _db_client(target),
        dry_run=True,
    )

    assert queued == []


@pytest.mark.asyncio
@pytest.mark.parametrize("input_error", [None, "can't use it"])
async def test_a_node_queued_to_fail_on_its_input_is_not_reconciled(
    input_error: str | None,
):
    """It was never charged up front, so settling it must not bill it either."""
    reconcile = AsyncMock(return_value=(0, 0))
    entry = _entry(dict(_QUEUED_INPUT))
    entry.input_error = input_error

    with (
        patch(f"{_MOD}.log_system_credential_cost", new=AsyncMock()),
        patch(f"{_MOD}.billing.charge_reconciled_usage", new=reconcile),
        patch(f"{_MOD}.async_update_node_execution_status", new=AsyncMock()),
        patch(f"{_MOD}.async_update_graph_execution_state", new=AsyncMock()),
    ):
        await ExecutionProcessor()._settle_node(
            node=_llm_node("model", "prompt"),
            node_exec=entry,
            status=ExecutionStatus.FAILED,
            execution_stats=NodeExecutionStats(),
            graph_stats_pair=(GraphExecutionStats(), threading.Lock()),
            db_client=MagicMock(),
            log_metadata=MagicMock(),
        )

    assert reconcile.await_count == (1 if input_error is None else 0)


@pytest.mark.asyncio
async def test_an_input_error_fails_the_block_but_not_the_run():
    """Like any other block error, the run itself still ends COMPLETED; only
    the block is FAILED."""
    entry = _entry(dict(_QUEUED_INPUT))
    entry.input_error = "can't use it"
    graph_stats = GraphExecutionStats()

    with (
        patch(f"{_MOD}.log_system_credential_cost", new=AsyncMock()),
        patch(f"{_MOD}.async_update_node_execution_status", new=AsyncMock()),
        patch(f"{_MOD}.async_update_graph_execution_state", new=AsyncMock()),
    ):
        await ExecutionProcessor()._settle_node(
            node=_llm_node("model", "prompt"),
            node_exec=entry,
            status=ExecutionStatus.FAILED,
            execution_stats=NodeExecutionStats(
                error=BlockInputError(
                    message="can't use it", block_name=_BLOCK.name, block_id=_BLOCK.id
                )
            ),
            graph_stats_pair=(graph_stats, threading.Lock()),
            db_client=MagicMock(),
            log_metadata=MagicMock(),
        )

    assert graph_stats.node_error_count == 1
    assert graph_stats.failure_reason is None
    assert graph_stats.error is None
