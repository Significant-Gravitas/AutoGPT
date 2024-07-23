import operator
from enum import Enum
from typing import Any, Union

import pydantic

from autogpt_server.data.block import Block, BlockOutput, BlockSchema
from autogpt_server.data.model import SchemaField


class Operation(Enum):
    ADD = "Add"
    SUBTRACT = "Subtract"
    MULTIPLY = "Multiply"
    DIVIDE = "Divide"
    POWER = "Power"


class MathsResult(pydantic.BaseModel):
    result: Union[float, int] | None = None
    explanation: str


class CounterResult(pydantic.BaseModel):
    count: int | None = None
    type: str
    explanation: str


class MathsBlock(Block):
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
        result: MathsResult = SchemaField(
            description="The result of your calculation with an explanation"
        )

    def __init__(self):
        super().__init__(
            id="b1ab9b19-67a6-406d-abf5-2dba76d00c79",
            input_schema=MathsBlock.Input,
            output_schema=MathsBlock.Output,
            test_input={
                "operation": Operation.ADD.value,
                "a": 10.0,
                "b": 5.0,
                "round_result": False,
            },
            test_output=[
                (
                    "result",
                    MathsResult(
                        result=15.0, explanation="Added 10.0 and 5.0 to get 15.0"
                    ),
                ),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        operation = input_data.operation
        a = input_data.a
        b = input_data.b

        operations = {
            Operation.ADD: (operator.add, "Added"),
            Operation.SUBTRACT: (operator.sub, "Subtracted"),
            Operation.MULTIPLY: (operator.mul, "Multiplied"),
            Operation.DIVIDE: (operator.truediv, "Divided"),
            Operation.POWER: (operator.pow, "Raised"),
        }

        op_func, op_word = operations[operation]

        try:
            if operation == Operation.DIVIDE and b == 0:
                raise ZeroDivisionError("Cannot divide by zero")

            result = op_func(a, b)

            if operation == Operation.POWER:
                explanation = f"{op_word} {a} to the power of {b} to get {result}"
            elif operation == Operation.DIVIDE:
                explanation = f"{op_word} {a} by {b} to get {result}"
            else:
                explanation = f"{op_word} {a} and {b} to get {result}"

            if input_data.round_result:
                result = round(result)
                explanation += " (rounded to the nearest whole number)"

            yield "result", MathsResult(result=result, explanation=explanation)

        except ZeroDivisionError:
            yield "result", MathsResult(
                result=None, explanation="Cannot divide by zero"
            )
        except Exception as e:
            yield "result", MathsResult(
                result=None, explanation=f"An error occurred: {str(e)}"
            )


class CounterBlock(Block):
    class Input(BlockSchema):
        collection: Any = SchemaField(
            description="Enter the collection you want to count. This can be a list, dictionary, string, or any other iterable.",
            placeholder="For example: [1, 2, 3] or {'a': 1, 'b': 2} or 'hello'",
        )

    class Output(BlockSchema):
        result: CounterResult = SchemaField(description="The result of the count")

    def __init__(self):
        super().__init__(
            id="3c9c2f42-b0c3-435f-ba35-05f7a25c772a",
            input_schema=CounterBlock.Input,
            output_schema=CounterBlock.Output,
            test_input={"collection": [1, 2, 3, 4, 5]},
            test_output=[
                (
                    "result",
                    CounterResult(
                        count=5, type="list", explanation="Counted 5 items in a list"
                    ),
                ),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        collection = input_data.collection

        try:
            if isinstance(collection, (str, list, tuple, set, dict)):
                count = len(collection)
                collection_type = type(collection).__name__
            elif hasattr(collection, "__iter__"):
                count = sum(1 for _ in collection)
                collection_type = "iterable"
            else:
                raise ValueError("Input is not a countable collection")

            if isinstance(collection, str):
                item_word = "character" if count == 1 else "characters"
            elif isinstance(collection, dict):
                item_word = "key-value pair" if count == 1 else "key-value pairs"
            else:
                item_word = "item" if count == 1 else "items"

            explanation = f"Counted {count} {item_word} in a {collection_type}"

            yield "result", CounterResult(
                count=count, type=collection_type, explanation=explanation
            )

        except Exception as e:
            yield "result", CounterResult(
                count=None, type="error", explanation=f"An error occurred: {str(e)}"
            )
