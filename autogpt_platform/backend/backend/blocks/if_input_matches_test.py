import json
from typing import Any

import pytest

from backend.blocks.branching import ConditionBlock, IfInputMatchesBlock
from backend.data.execution import ExecutionContext
from backend.data.graph import Link, Node
from backend.executor.utils import validate_exec
from backend.util.exceptions import BlockInputError
from backend.util.type import coerce_inputs_to_schema


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    """These pure block tests do not create persisted graphs or need services."""
    yield


async def _matches(input_value: Any, value: Any) -> bool:
    block = IfInputMatchesBlock()
    outputs = [
        output
        async for output in block.run(
            IfInputMatchesBlock.Input(
                input=input_value, value=value, yes_value="yes", no_value="no"
            )
        )
    ]
    return dict(outputs)["result"]


@pytest.mark.parametrize(
    "input_value, value",
    [
        (0, "hello"),  # non-numeric text used to coerce to 0
        (0, "None"),
        (0.0, "abc"),
        (10, "10.9"),  # used to truncate to 10
        (3, [1, 2, 3]),  # used to compare against len(value)
        (2, {"a": 1, "b": 2}),
        (False, "no"),  # any string other than "true"/"1" used to become False
        (False, ""),
        (9007199254740992, "9007199254740993"),  # float() rounded these equal
        (0.1, "0.10000000000000001"),
        ([0.1], "[0.10000000000000001]"),  # nested numbers are exact too
        ({"a": 0.1}, '{"a": 0.10000000000000001}'),
        (1.0, "nan"),
        (["hello"], "hello"),  # convert() wrapped scalars into containers
        ([3], 3),
        ({"value": "x"}, "x"),
        ({"value": 5}, "5"),
        (("x",), "x"),
        ({"x"}, "x"),
        ({1}, "[1, 1]"),  # set() would drop the duplicate
        ([1, 2], '{"a": 1}'),
        (True, "yes"),  # only the spellings convert() treats as booleans
        (False, "off"),
        (True, 2),  # used to compare against bool(value)
        ([], {}),
    ],
)
async def test_lossy_conversion_does_not_report_a_match(input_value, value):
    assert await _matches(input_value, value) is False


@pytest.mark.parametrize(
    "input_value, value",
    [
        (10, 10),
        (10, "10"),
        (10, " 10 "),
        (10, "10.0"),
        (2.5, "2.5"),
        (0.1, "0.1"),
        (1000, "1e3"),
        (1.5, "1.50"),
        (0, "0"),
        (True, "true"),
        (True, "True"),
        (False, "false"),
        ("hello", "hello"),
        ("5", 5),
        ([1, 2], "[1, 2]"),
        ([1, 2], (1, 2)),
        ((1, 2), [1, 2]),
        ({"a": 1}, '{"a": 1}'),
        (True, "1"),  # saved graphs compare toggles against "1"/"0"
        (True, " 1 "),
        (False, "0"),
        (10, "1_0"),
        (1000, "1_000"),
        (9007199254740993, "9007199254740993"),  # float() used to miss this
        (True, 1),  # plain equality between bool and number is unchanged
        ([True], "[1]"),
        ([0.1], "[0.10]"),
        ({"a": [1.5]}, '{"a": [1.50]}'),
        ([100.0], "[1e2]"),
        ({1, 2}, "[2, 1]"),
        ([1, 1], "[1, 1]"),
    ],
)
async def test_equivalent_values_still_match(input_value, value):
    assert await _matches(input_value, value) is True


@pytest.mark.parametrize(
    "input_value, value",
    [
        (10, 20),
        (10, "None"),
        (True, "false"),
        ("hello", "world"),
    ],
)
async def test_different_values_do_not_match(input_value, value):
    assert await _matches(input_value, value) is False


@pytest.mark.parametrize(
    "input_value, value, expected",
    [
        (True, " 1 ", True),
        (False, " 0 ", True),
        (True, " TRUE ", True),
        (False, " FALSE ", True),
        (True, 1, True),
        (False, 0, True),
        (10, "1_0", True),
        (1000.0, "1_000", True),
        (9007199254740993, "9007199254740993", True),
        (12345678901234567890, "12345678901234567890", True),
        ([True], "[1]", True),
        ({"a": 1}, '{"a": 1}', True),
        ("example.com", "example.com", True),
        ("example.com", "other.example", False),
        (0, "hello", False),
        (False, "off", False),
        (True, 2, False),
        (10, "10.9", False),
        (10, 10.9, False),
        (9007199254740992, "9007199254740993", False),
        (0.1, "0.10000000000000001", False),
        (0, "1e999999999999999999999", False),
        (0, "nan", False),
        (0, "Infinity", False),
        (3, [1, 2, 3], False),
        (2, {"a": 1, "b": 2}, False),
        (["x"], "x", False),
        ({"value": "x"}, "x", False),
        ([], {}, False),
    ],
)
async def test_saved_node_inputs_keep_their_types_and_route_correctly(
    input_value, value, expected
):
    block = IfInputMatchesBlock()
    fields = json.loads(json.dumps({"input": input_value, "value": value}))
    coerce_inputs_to_schema(fields, block.input_schema)
    assert fields["input"] == input_value
    assert type(fields["input"]) is type(input_value)
    assert fields["value"] == value
    assert type(fields["value"]) is type(value)

    node = Node(
        id="match-node",
        block_id=block.id,
        input_default={"value": fields["value"], "yes_value": "yes", "no_value": "no"},
        input_links=[
            Link(
                id="input-link",
                source_id="upstream-node",
                sink_id="match-node",
                source_name="output",
                sink_name="input",
            )
        ],
    )
    validated, message = validate_exec(node, {"input": fields["input"]})
    assert validated is not None, message
    assert validated["input"] == input_value
    assert type(validated["input"]) is type(input_value)
    assert validated["value"] == value
    assert type(validated["value"]) is type(value)
    outputs = [
        output
        async for output in block.execute(
            validated, execution_context=ExecutionContext()
        )
    ]
    branch = ("yes_output", "yes") if expected else ("no_output", "no")
    assert outputs == [("result", expected), branch]


@pytest.mark.parametrize("missing", ["input", "value"])
async def test_null_required_inputs_remain_invalid_at_execution(missing):
    block = IfInputMatchesBlock()
    fields = {"input": 0, "value": 0, missing: None}
    coerce_inputs_to_schema(fields, block.input_schema)
    assert fields[missing] is None
    with pytest.raises(BlockInputError):
        async for _ in block.execute(fields, execution_context=ExecutionContext()):
            pass


@pytest.mark.parametrize("block_class", [ConditionBlock, IfInputMatchesBlock])
async def test_builtin_branching_vectors(block_class):
    block = block_class()
    inputs = (
        block.test_input if isinstance(block.test_input, list) else [block.test_input]
    )
    outputs = []
    for fields in inputs:
        assert fields is not None
        outputs.extend(
            [
                output
                async for output in block.execute(
                    fields, execution_context=ExecutionContext()
                )
            ]
        )
    assert outputs == block.test_output
