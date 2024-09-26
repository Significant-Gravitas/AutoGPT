import operator
from enum import Enum
from typing import Any

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class Operation(Enum):
    ADD = "Add"
    SUBTRACT = "Subtract"
    MULTIPLY = "Multiply"
    DIVIDE = "Divide"
    POWER = "Power"


class CalculatorBlock(Block):
    class Input(BlockSchema):
        operation: Operation = SchemaField(
            description="Choose the math operation you want to perform",
            placeholder="Select an operation",
        )
        a: float = SchemaField(
            description="Enter the first number (A)", placeholder="For example: 10"
        )
        b: float = SchemaField(
            description="Enter the second number (B)", placeholder="For example: 5"
        )
        round_result: bool = SchemaField(
            description="Do you want to round the result to a whole number?",
            default=False,
        )

    class Output(BlockSchema):
        result: float = SchemaField(description="The result of your calculation")

    def __init__(self):
        super().__init__(
            id="b1ab9b19-67a6-406d-abf5-2dba76d00c79",
            input_schema=CalculatorBlock.Input,
            output_schema=CalculatorBlock.Output,
            description="Performs a mathematical operation on two numbers.",
            categories={BlockCategory.LOGIC},
            test_input={
                "operation": Operation.ADD.value,
                "a": 10.0,
                "b": 5.0,
                "round_result": False,
            },
            test_output=[
                ("result", 15.0),
            ],
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        operation = input_data.operation
        a = input_data.a
        b = input_data.b

        operations = {
            Operation.ADD: operator.add,
            Operation.SUBTRACT: operator.sub,
            Operation.MULTIPLY: operator.mul,
            Operation.DIVIDE: operator.truediv,
            Operation.POWER: operator.pow,
        }

        op_func = operations[operation]

        try:
            if operation == Operation.DIVIDE and b == 0:
                raise ZeroDivisionError("Cannot divide by zero")

            result = op_func(a, b)

            if input_data.round_result:
                result = round(result)

            yield "result", result

        except ZeroDivisionError:
            yield "result", float("inf")  # Return infinity for division by zero
        except Exception:
            yield "result", float("nan")  # Return NaN for other errors


class CountItemsBlock(Block):
    class Input(BlockSchema):
        collection: Any = SchemaField(
            description="Enter the collection you want to count. This can be a list, dictionary, string, or any other iterable.",
            placeholder="For example: [1, 2, 3] or {'a': 1, 'b': 2} or 'hello'",
        )

    class Output(BlockSchema):
        count: int = SchemaField(description="The number of items in the collection")

    def __init__(self):
        super().__init__(
            id="3c9c2f42-b0c3-435f-ba35-05f7a25c772a",
            input_schema=CountItemsBlock.Input,
            output_schema=CountItemsBlock.Output,
            description="Counts the number of items in a collection.",
            categories={BlockCategory.LOGIC},
            test_input={"collection": [1, 2, 3, 4, 5]},
            test_output=[
                ("count", 5),
            ],
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        collection = input_data.collection

        try:
            if isinstance(collection, (str, list, tuple, set, dict)):
                count = len(collection)
            elif hasattr(collection, "__iter__"):
                count = sum(1 for _ in collection)
            else:
                raise ValueError("Input is not a countable collection")

            yield "count", count

        except Exception:
            yield "count", -1  # Return -1 to indicate an error
