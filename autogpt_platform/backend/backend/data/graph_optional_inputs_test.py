"""An input with Advanced on is optional: a run can leave it blank (#10614)."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.blocks.io import (
    AgentFileInputBlock,
    AgentGoogleDriveFileInputBlock,
    AgentInputBlock,
    AgentOutputBlock,
    AgentTableInputBlock,
)
from backend.data.graph import BaseGraph, missing_inputs


@pytest.mark.parametrize(
    "input_default,required,advanced",
    [
        ({"name": "x"}, True, False),
        ({"name": "x", "value": "a"}, False, False),
        ({"name": "x", "advanced": True}, False, True),
        ({"name": "x", "value": "a", "advanced": True}, False, True),
    ],
)
def test_an_input_is_optional_when_it_has_a_default_or_is_advanced(
    input_default: dict[str, Any], required: bool, advanced: bool
) -> None:
    schema = BaseGraph._generate_schema((AgentInputBlock.Input, input_default))

    assert ("x" in schema["required"]) is required
    assert schema["properties"]["x"]["advanced"] is advanced
    assert missing_inputs(schema, {}) == (["x"] if required else [])


def test_an_advanced_output_keeps_its_old_schema() -> None:
    schema = BaseGraph._generate_schema(
        (AgentOutputBlock.Input, {"name": "out", "advanced": True})
    )

    assert schema["required"] == ["out"]
    assert schema["properties"]["out"]["advanced"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_block", [AgentInputBlock, AgentGoogleDriveFileInputBlock]
)
@pytest.mark.parametrize("advanced,outputs", [(True, [("result", None)]), (False, [])])
async def test_a_blank_input_sends_none_only_when_optional(
    input_block: type[AgentInputBlock], advanced: bool, outputs: list
) -> None:
    input_data = input_block.Input(name="x", advanced=advanced)

    assert [o async for o in input_block().run(input_data)] == outputs


@pytest.mark.asyncio
@pytest.mark.parametrize("advanced,outputs", [(True, [("result", None)]), (False, [])])
async def test_a_blank_file_input_sends_none_only_when_optional(
    advanced: bool, outputs: list
) -> None:
    input_data = AgentFileInputBlock.Input(name="x", advanced=advanced)

    result = AgentFileInputBlock().run(input_data, execution_context=MagicMock())

    assert [o async for o in result] == outputs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "advanced,value,expected",
    [
        (True, None, None),
        (True, [], []),
        (False, None, []),
        (True, [{"x": "a"}], [{"x": "a"}]),
    ],
)
async def test_a_blank_optional_table_preserves_downstream_defaults(
    advanced: bool,
    value: list[dict[str, Any]] | None,
    expected: list[dict[str, Any]] | None,
) -> None:
    input_data = AgentTableInputBlock.Input(name="x", advanced=advanced, value=value)

    assert [o async for o in AgentTableInputBlock().run(input_data)] == [
        ("result", expected)
    ]
