import json
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.type import convert


class ComparisonOperator(Enum):
    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN_OR_EQUAL = "<="


class ConditionBlock(Block):
    class Input(BlockSchemaInput):
        value1: Any = SchemaField(
            description="Enter the first value for comparison",
            placeholder="For example: 10 or 'hello' or True",
        )
        operator: ComparisonOperator = SchemaField(
            description="Choose the comparison operator",
            placeholder="Select an operator",
        )
        value2: Any = SchemaField(
            description="Enter the second value for comparison",
            placeholder="For example: 20 or 'world' or False",
        )
        yes_value: Any = SchemaField(
            description="(Optional) Value to output if the condition is true. If not provided, value1 will be used.",
            placeholder="Leave empty to use value1, or enter a specific value",
            default=None,
        )
        no_value: Any = SchemaField(
            description="(Optional) Value to output if the condition is false. If not provided, value1 will be used.",
            placeholder="Leave empty to use value1, or enter a specific value",
            default=None,
        )

    class Output(BlockSchemaOutput):
        result: bool = SchemaField(
            description="The result of the condition evaluation (True or False)"
        )
        yes_output: Any = SchemaField(
            description="The output value if the condition is true"
        )
        no_output: Any = SchemaField(
            description="The output value if the condition is false"
        )

    def __init__(self):
        super().__init__(
            id="715696a0-e1da-45c8-b209-c2fa9c3b0be6",
            input_schema=ConditionBlock.Input,
            output_schema=ConditionBlock.Output,
            description="Handles conditional logic based on comparison operators",
            categories={BlockCategory.LOGIC},
            test_input={
                "value1": 10,
                "operator": ComparisonOperator.GREATER_THAN.value,
                "value2": 5,
                "yes_value": "Greater",
                "no_value": "Not greater",
            },
            test_output=[
                ("result", True),
                ("yes_output", "Greater"),
            ],
            effect=BlockEffect.NONE,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        operator = input_data.operator

        value1 = input_data.value1
        if isinstance(value1, str):
            try:
                value1 = float(value1.strip())
            except ValueError:
                value1 = value1.strip()

        value2 = input_data.value2
        if isinstance(value2, str):
            try:
                value2 = float(value2.strip())
            except ValueError:
                value2 = value2.strip()

        yes_value = input_data.yes_value if input_data.yes_value is not None else value1
        no_value = input_data.no_value if input_data.no_value is not None else value2

        comparison_funcs = {
            ComparisonOperator.EQUAL: lambda a, b: a == b,
            ComparisonOperator.NOT_EQUAL: lambda a, b: a != b,
            ComparisonOperator.GREATER_THAN: lambda a, b: a > b,
            ComparisonOperator.LESS_THAN: lambda a, b: a < b,
            ComparisonOperator.GREATER_THAN_OR_EQUAL: lambda a, b: a >= b,
            ComparisonOperator.LESS_THAN_OR_EQUAL: lambda a, b: a <= b,
        }

        try:
            result = comparison_funcs[operator](value1, value2)
        except Exception as e:
            raise ValueError(f"Comparison failed: {e}") from e

        yield "result", result

        if result:
            yield "yes_output", yes_value
        else:
            yield "no_output", no_value


def _convert_for_match(value: Any, target: Any) -> Any:
    """Convert *value* to the type of *target* only when the conversion is lossless.

    ``convert`` is deliberately lenient (non-numeric text becomes 0, a list
    becomes its length, any text other than "true"/"1" becomes False), which
    is fine for filling block inputs but makes unrelated values compare equal.
    Values that cannot be converted faithfully are returned unchanged.
    """
    input_type = type(target)
    if input_type in (Any, object):
        return value
    if issubclass(input_type, bool):
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        return value
    if issubclass(input_type, (int, float)):
        if isinstance(value, str):
            # Compare exact decimal values: float() would round, so that
            # "9007199254740993" matched 9007199254740992.
            text = value.strip()
            if "_" in text:
                return value
            try:
                number = Decimal(text)
            except InvalidOperation:
                return value
            if number.is_finite() and number == Decimal(str(target)):
                return target
        return value
    if issubclass(input_type, (list, tuple, set, dict)):
        # convert() wraps a scalar into a container (["x"], {"value": "x"}),
        # which made "x" match ["x"]. Accept only text that parses to the
        # same shape, or a list/tuple holding the same items.
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except ValueError:
                return value
            if issubclass(input_type, dict):
                return parsed if isinstance(parsed, dict) else value
            return input_type(parsed) if isinstance(parsed, list) else value
        if isinstance(value, (list, tuple)) and issubclass(input_type, (list, tuple)):
            return input_type(value)
        return value
    return convert(value, input_type)


class IfInputMatchesBlock(Block):
    class Input(BlockSchemaInput):
        input: Any = SchemaField(
            description="The input to match against",
            placeholder="For example: 10 or 'hello' or True",
        )
        value: Any = SchemaField(
            description="The value to output if the input matches",
            placeholder="For example: 'Greater' or 20 or False",
        )
        yes_value: Any = SchemaField(
            description="The value to output if the input matches",
            placeholder="For example: 'Greater' or 20 or False",
            default=None,
        )
        no_value: Any = SchemaField(
            description="The value to output if the input does not match",
            placeholder="For example: 'Greater' or 20 or False",
            default=None,
        )

    class Output(BlockSchemaOutput):
        result: bool = SchemaField(
            description="The result of the condition evaluation (True or False)"
        )
        yes_output: Any = SchemaField(
            description="The output value if the condition is true"
        )
        no_output: Any = SchemaField(
            description="The output value if the condition is false"
        )

    def __init__(self):
        super().__init__(
            id="6dbbc4b3-ca6c-42b6-b508-da52d23e13f2",
            input_schema=IfInputMatchesBlock.Input,
            output_schema=IfInputMatchesBlock.Output,
            description="Handles conditional logic based on comparison operators",
            categories={BlockCategory.LOGIC},
            test_input=[
                {
                    "input": 10,
                    "value": 10,
                    "yes_value": "Greater",
                    "no_value": "Not greater",
                },
                {
                    "input": 10,
                    "value": 20,
                    "yes_value": "Greater",
                    "no_value": "Not greater",
                },
                {
                    "input": 10,
                    "value": "None",
                    "yes_value": "Yes",
                    "no_value": "No",
                },
            ],
            test_output=[
                ("result", True),
                ("yes_output", "Greater"),
                ("result", False),
                ("no_output", "Not greater"),
                ("result", False),
                ("no_output", "No"),
                # ("result", True),
                # ("yes_output", "Yes"),
            ],
            effect=BlockEffect.NONE,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:

        # If input_data.value is not matching input_data.input, convert value to type of input
        if (
            input_data.input != input_data.value
            and input_data.input is not input_data.value
        ):
            try:
                # Only attempt conversion if input is not None and value is not None
                if input_data.input is not None and input_data.value is not None:
                    input_data.value = _convert_for_match(
                        input_data.value, input_data.input
                    )
            except Exception:
                pass  # If conversion fails, just leave value as is

        if input_data.input == input_data.value:
            yield "result", True
            yield "yes_output", input_data.yes_value
        else:
            yield "result", False
            yield "no_output", input_data.no_value
