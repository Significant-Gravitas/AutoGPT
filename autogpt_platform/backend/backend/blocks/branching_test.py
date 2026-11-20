"""ConditionBlock passes value1 through when a branch value is omitted."""

from __future__ import annotations

import pytest

from backend.blocks import get_block
from backend.blocks.branching import ComparisonOperator, ConditionBlock
from backend.data.execution import ExecutionContext
from backend.data.graph import Link, Node
from backend.executor.utils import validate_exec
from backend.util.exceptions import BlockExecutionError


async def _run(**fields: object) -> dict[str, object]:
    node = Node(block_id=ConditionBlock().id, input_default=fields)
    return await _execute_node(node, {})


async def _execute_node(node: Node, data: dict[str, object]) -> dict[str, object]:
    block = get_block(node.block_id)
    assert type(block) is ConditionBlock
    validated, error = validate_exec(node, data)
    assert validated is not None, error
    return {
        name: value
        async for name, value in block.execute(
            validated, execution_context=ExecutionContext()
        )
    }


async def test_omitted_false_value_passes_through_value1():
    outputs = await _run(
        value1=3,
        operator=ComparisonOperator.GREATER_THAN.value,
        value2=5,
    )

    assert outputs["result"] is False
    assert outputs["no_output"] == 3
    assert "yes_output" not in outputs


async def test_omitted_true_value_passes_through_value1():
    outputs = await _run(
        value1=10,
        operator=ComparisonOperator.GREATER_THAN.value,
        value2=5,
    )

    assert outputs["result"] is True
    assert outputs["yes_output"] == 10


async def test_explicit_false_value_is_kept():
    outputs = await _run(
        value1=3,
        operator=ComparisonOperator.GREATER_THAN.value,
        value2=5,
        no_value="Not greater",
    )

    assert outputs["no_output"] == "Not greater"


@pytest.mark.parametrize("overrides", [{}, {"yes_value": None, "no_value": None}])
@pytest.mark.parametrize("operator,result", [("==", False), ("!=", True)])
@pytest.mark.parametrize(
    "value1,value2,expected",
    [
        (" 3 ", " 5 ", 3.0),
        (" left ", "right", "left"),
        (False, True, False),
        (0, 1, 0),
        ("", "other", ""),
        ([], [1], []),
        ({}, {"key": "value"}, {}),
    ],
)
async def test_missing_or_null_override_uses_normalized_value1(
    overrides: dict[str, object],
    operator: str,
    result: bool,
    value1: object,
    value2: object,
    expected: object,
):
    outputs = await _run(value1=value1, operator=operator, value2=value2, **overrides)

    output_name = "yes_output" if result else "no_output"
    assert outputs == {"result": result, output_name: expected}
    assert outputs["result"] is result
    assert type(outputs[output_name]) is type(expected)


@pytest.mark.parametrize("override", [False, 0, "", [], {}, " 3 ", "Not greater"])
@pytest.mark.parametrize("operator,result", [("==", False), ("!=", True)])
async def test_explicit_overrides_are_preserved(
    override: object, operator: str, result: bool
):
    outputs = await _run(
        value1=3,
        operator=operator,
        value2=5,
        yes_value=override,
        no_value=override,
    )

    output_name = "yes_output" if result else "no_output"
    assert outputs == {"result": result, output_name: override}
    assert outputs["result"] is result
    assert type(outputs[output_name]) is type(override)


@pytest.mark.parametrize("override", [None, False, 0, "", [], {}, " 3 "])
async def test_saved_workflow_link_overrides_default(override: object):
    node = Node(
        id="condition-node",
        block_id="715696a0-e1da-45c8-b209-c2fa9c3b0be6",
        input_default={"value1": 3, "operator": ">", "value2": 5, "no_value": "saved"},
        input_links=[
            Link(
                source_id="upstream-node",
                sink_id="condition-node",
                source_name="output",
                sink_name="no_value",
            )
        ],
    )
    restored = Node.model_validate_json(node.model_dump_json())
    missing, error = validate_exec(restored, {})
    assert missing is None
    assert "unpopulated links" in error

    outputs = await _execute_node(restored, {"no_value": override})

    expected = 3 if override is None else override
    assert outputs == {"result": False, "no_output": expected}
    assert type(outputs["no_output"]) is type(expected)


@pytest.mark.parametrize(
    "fields",
    [
        {"value1": 3, "operator": "invalid", "value2": 5},
        {"operator": ">", "value2": 5},
        {"value1": 3, "operator": ">"},
    ],
)
def test_executor_rejects_invalid_inputs(fields: dict[str, object]):
    data, error = validate_exec(
        Node(block_id=ConditionBlock().id, input_default=fields), {}
    )
    assert data is None
    assert "Input data missing or mismatch" in error


async def test_incomparable_values_raise_block_execution_error():
    with pytest.raises(BlockExecutionError, match="Comparison failed"):
        await _run(value1=[1], operator=">", value2=3)
