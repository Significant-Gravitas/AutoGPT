from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from backend.blocks.basic import StoreValueBlock
from backend.blocks.time_blocks import CountdownTimerBlock
from backend.data.graph import GraphModel, Link, Node


async def _run(block: CountdownTimerBlock, **input_kwargs):
    outputs = []
    async for name, value in block.run(block.input_schema(**input_kwargs)):
        outputs.append((name, value))
    return outputs


@pytest.mark.asyncio
async def test_countdown_timer_rejects_excessive_duration():
    block = CountdownTimerBlock()
    with pytest.raises(ValueError, match="exceeds max"):
        await _run(block, days=365, repeat=1)


@pytest.mark.asyncio
async def test_countdown_timer_rejects_cumulative_duration_over_cap():
    block = CountdownTimerBlock()
    with pytest.raises(ValueError, match="exceeds max"):
        await _run(block, days=1, repeat=10)


@pytest.mark.asyncio
async def test_countdown_timer_rejects_negative_duration():
    block = CountdownTimerBlock()
    with pytest.raises(ValueError, match="non-negative"):
        await _run(block, seconds="-1")


def test_countdown_timer_rejects_repeat_zero_at_schema():
    block = CountdownTimerBlock()
    with pytest.raises(ValidationError):
        block.input_schema(seconds=1, repeat=0)


def test_countdown_timer_rejects_repeat_over_max_at_schema():
    block = CountdownTimerBlock()
    with pytest.raises(ValidationError):
        block.input_schema(seconds=1, repeat=1001)


@pytest.mark.asyncio
async def test_countdown_timer_run_rejects_repeat_zero_defense_in_depth():
    block = CountdownTimerBlock()
    bypassed = block.input_schema.model_construct(seconds=1, repeat=0)
    with pytest.raises(ValueError, match="Repeat must be between"):
        async for _ in block.run(bypassed):
            pass


@pytest.mark.asyncio
async def test_countdown_timer_run_rejects_repeat_over_max_defense_in_depth():
    block = CountdownTimerBlock()
    bypassed = block.input_schema.model_construct(seconds=1, repeat=1001)
    with pytest.raises(ValueError, match="Repeat must be between"):
        async for _ in block.run(bypassed):
            pass


@pytest.mark.asyncio
async def test_countdown_timer_allows_duration_at_cap(mocker):
    sleep_mock = mocker.patch(
        "backend.blocks.time_blocks.asyncio.sleep", new_callable=AsyncMock
    )
    block = CountdownTimerBlock()
    outputs = await _run(block, days=7, repeat=1)
    assert outputs == [("output_message", "timer finished")]
    sleep_mock.assert_awaited_once_with(7 * 86400)


def test_countdown_timer_execution_timeout_covers_max_duration():
    block = CountdownTimerBlock()
    assert block.execution_timeout_seconds is not None
    assert block.execution_timeout_seconds >= block.MAX_TOTAL_SECONDS


def test_countdown_timer_get_field_errors_reports_per_field_bound_violations():
    block = CountdownTimerBlock()
    errors = block.input_schema.get_field_errors({"repeat": 1200})
    assert "repeat" in errors
    assert "1200" in errors["repeat"]
    assert "1000" in errors["repeat"]


def test_countdown_timer_get_field_errors_clean_when_within_bounds():
    block = CountdownTimerBlock()
    assert block.input_schema.get_field_errors({"repeat": 5}) == {}


@pytest.mark.asyncio
async def test_countdown_timer_rejects_non_numeric_string_duration():
    block = CountdownTimerBlock()
    with pytest.raises(ValueError, match="seconds must be a valid integer"):
        await _run(block, seconds="abc")


@pytest.mark.asyncio
async def test_countdown_timer_emits_one_message_per_repeat(mocker):
    sleep_mock = mocker.patch(
        "backend.blocks.time_blocks.asyncio.sleep", new_callable=AsyncMock
    )
    block = CountdownTimerBlock()
    outputs = await _run(block, seconds=1, repeat=3)
    assert outputs == [("output_message", "timer finished")] * 3
    assert sleep_mock.await_count == 3


def _countdown_node(node_id: str, **input_default) -> Node:
    return Node(
        id=node_id,
        block_id=CountdownTimerBlock().id,
        input_default=input_default,
    )


def _graph(nodes: list[Node], links: list[Link] | None = None) -> GraphModel:
    # Bypass GraphModel field validators by constructing the structural fields
    # directly — these tests only exercise the per-field jsonschema bound check
    # in ``_validate_graph_get_errors`` and don't need DB-side metadata.
    return GraphModel.model_construct(
        id="g",
        version=1,
        name="t",
        description="t",
        nodes=nodes,
        links=links or [],
        sub_graphs=[],
    )


def test_validate_graph_surfaces_bound_violation_inline_on_field():
    node = _countdown_node("n1", repeat=1200)
    graph = _graph([node])

    node_errors = graph.validate_graph_get_errors(for_run=True)

    assert "n1" in node_errors
    assert "repeat" in node_errors["n1"]
    assert "1000" in node_errors["n1"]["repeat"]


def test_validate_graph_skips_bound_check_when_field_is_linked():
    # ``repeat`` is linked from an upstream block — the runtime value isn't
    # known at validation time, so the saved ``input_default`` placeholder
    # (here, an out-of-range value) should NOT raise a spurious field error.
    source = Node(id="src", block_id=StoreValueBlock().id, input_default={"input": 5})
    countdown = _countdown_node("n1", repeat=1200)
    link = Link(
        source_id="src",
        sink_id="n1",
        source_name="output",
        sink_name="repeat",
    )
    graph = _graph([source, countdown], [link])

    node_errors = graph.validate_graph_get_errors(for_run=True)

    assert "repeat" not in node_errors.get("n1", {})
