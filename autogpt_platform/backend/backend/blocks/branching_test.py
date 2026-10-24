"""ConditionBlock passes value1 through when a branch value is omitted."""

from __future__ import annotations

from backend.blocks.branching import ComparisonOperator, ConditionBlock


async def _run(**fields: object) -> dict[str, object]:
    block = ConditionBlock()
    data = ConditionBlock.Input(
        value1=fields["value1"],
        operator=fields["operator"],
        value2=fields["value2"],
        yes_value=fields.get("yes_value"),
        no_value=fields.get("no_value"),
    )
    outputs: dict[str, object] = {}
    async for name, value in block.run(data):
        outputs[name] = value
    return outputs


async def test_omitted_false_value_passes_through_value1():
    outputs = await _run(
        value1=3,
        operator=ComparisonOperator.GREATER_THAN,
        value2=5,
    )

    assert outputs["result"] is False
    assert outputs["no_output"] == 3
    assert "yes_output" not in outputs


async def test_omitted_true_value_passes_through_value1():
    outputs = await _run(
        value1=10,
        operator=ComparisonOperator.GREATER_THAN,
        value2=5,
    )

    assert outputs["result"] is True
    assert outputs["yes_output"] == 10


async def test_explicit_false_value_is_kept():
    outputs = await _run(
        value1=3,
        operator=ComparisonOperator.GREATER_THAN,
        value2=5,
        no_value="Not greater",
    )

    assert outputs["no_output"] == "Not greater"
