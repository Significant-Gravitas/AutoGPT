from enum import Enum
from typing import Any

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class ComparisonOperator(Enum):
    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN_OR_EQUAL = "<="


class ConditionBlock(Block):
    class Input(BlockSchema):
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

    class Output(BlockSchema):
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
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
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

        result = comparison_funcs[operator](value1, value2)

        yield "result", result

        if result:
            yield "yes_output", yes_value
        else:
            yield "no_output", no_value


class IfInputMatchesBlock(Block):
    class Input(BlockSchema):
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

    class Output(BlockSchema):
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
                    "value": None,
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
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        if input_data.input == input_data.value or input_data.input is input_data.value:
            yield "result", True
            yield "yes_output", input_data.yes_value
        else:
            yield "result", False
            yield "no_output", input_data.no_value
