from enum import Enum
from typing import Any

from autogpt_server.data.block import Block, BlockOutput, BlockSchema
from autogpt_server.data.model import SchemaField


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
        true_value: Any = SchemaField(
            description="(Optional) Value to output if the condition is true. If not provided, value1 will be used.",
            placeholder="Leave empty to use value1, or enter a specific value",
            default=None,
        )
        false_value: Any = SchemaField(
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
            test_input={
                "value1": 10,
                "operator": ComparisonOperator.GREATER_THAN.value,
                "value2": 5,
                "true_value": "Greater",
                "false_value": "Not greater",
            },
            test_output=[
                ("result", True),
                ("true_output", "Greater"),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        value1 = input_data.value1
        operator = input_data.operator
        value2 = input_data.value2
        true_value = (
            input_data.true_value if input_data.true_value is not None else value1
        )
        false_value = (
            input_data.false_value if input_data.false_value is not None else value1
        )

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

            yield "result", result

            if result:
                yield "true_output", true_value
            else:
                yield "false_output", false_value

        except Exception:
            yield "result", None
            yield "true_output", None
            yield "false_output", None
