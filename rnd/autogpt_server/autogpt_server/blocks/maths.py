from autogpt_server.data.block import Block, BlockSchema, BlockOutput
from autogpt_server.data.model import SchemaField
from typing import Union
from enum import Enum
import operator
from typing import Any

class Operation(Enum):
    ADD = "Add"
    SUBTRACT = "Subtract"
    MULTIPLY = "Multiply"
    DIVIDE = "Divide"
    POWER = "Power"

class MathsBlock(Block):
    class Input(BlockSchema):
        operation: Operation = SchemaField(
            description="Choose the math operation you want to perform",
            placeholder="Select an operation"
        )
        a: float = SchemaField(
            description="Enter the first number (A)",
            placeholder="For example: 10"
        )
        b: float = SchemaField(
            description="Enter the second number (B)",
            placeholder="For example: 5"
        )
        round_result: bool = SchemaField(
            description="Do you want to round the result to a whole number?",
            default=False
        )

    class Output(BlockSchema):
        result: Union[float, int] = SchemaField(
            description="The result of your calculation"
        )
        explanation: str = SchemaField(
            description="A simple explanation of the calculation performed"
        )

    def __init__(self):
        super().__init__(
            id="simple-arithmetic-block",
            input_schema=MathsBlock.Input,
            output_schema=MathsBlock.Output,
            test_input={
                "operation": Operation.ADD,
                "a": 10,
                "b": 5,
                "round_result": False
            },
            test_output=[
                ("result", 15),
                ("explanation", "Added 10 and 5 to get 15")
            ]
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
            Operation.POWER: (operator.pow, "Raised")
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

            yield "result", result
            yield "explanation", explanation

        except ZeroDivisionError as e:
            yield "result", None
            yield "explanation", str(e)
        except Exception as e:
            yield "result", None
            yield "explanation", f"An error occurred: {str(e)}"



class CounterBlock(Block):
    class Input(BlockSchema):
        collection: Any = SchemaField(
            description="Enter the collection you want to count. This can be a list, dictionary, string, or any other iterable.",
            placeholder="For example: [1, 2, 3] or {'a': 1, 'b': 2} or 'hello'",
        )

    class Output(BlockSchema):
        count: int = SchemaField(
            description="The number of items in the collection"
        )
        type: str = SchemaField(
            description="The type of the input collection"
        )
        explanation: str = SchemaField(
            description="A simple explanation of what was counted"
        )

    def __init__(self):
        super().__init__(
            id="count-block",
            input_schema=CounterBlock.Input,
            output_schema=CounterBlock.Output,
            test_input={
                "collection": [1, 2, 3, 4, 5]
            },
            test_output=[
                ("count", 5),
                ("type", "list"),
                ("explanation", "Counted 5 items in a list")
            ]
        )

    def run(self, input_data: Input) -> BlockOutput:
        collection = input_data.collection
        
        try:
            if isinstance(collection, (str, list, tuple, set, dict)):
                count = len(collection)
                collection_type = type(collection).__name__
            elif hasattr(collection, '__iter__'):
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

            yield "count", count
            yield "type", collection_type
            yield "explanation", explanation

        except Exception as e:
            yield "count", None
            yield "type", "error"
            yield "explanation", f"An error occurred: {str(e)}"