from typing import Any

import pytest

from backend.blocks.branching import IfInputMatchesBlock


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
        (10, "1_0"),
        (1.0, "nan"),
        (["hello"], "hello"),  # convert() wrapped scalars into containers
        ([3], 3),
        ({"value": "x"}, "x"),
        ({"value": 5}, "5"),
        (("x",), "x"),
        ({"x"}, "x"),
        ([1, 2], '{"a": 1}'),
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
