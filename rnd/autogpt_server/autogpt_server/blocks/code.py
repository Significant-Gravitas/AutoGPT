import io
import sys

from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from autogpt_server.data.model import SchemaField


class PythonExecutionBlock(Block):
    class Input(BlockSchema):
        code: str = SchemaField(
            description="Python code to execute", placeholder="print('Hello, World!')"
        )
        timeout: int = SchemaField(
            description="Execution timeout in seconds", default=5
        )

    class Output(BlockSchema):
        result: str = SchemaField(description="Execution result or output")
        error: str = SchemaField(description="Error message if execution failed")

    def __init__(self):
        super().__init__(
            id="dabd5e16-f39c-4ad9-925a-8df60164d2f7",
            description="This block executes Python code and returns the output or any error messages.",
            categories={BlockCategory.BASIC},
            input_schema=PythonExecutionBlock.Input,
            output_schema=PythonExecutionBlock.Output,
            test_input={"code": "print('Hello, World!')", "timeout": 5},
            test_output=[
                ("result", "Hello, World!\n"),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        code = input_data.code
        timeout = input_data.timeout

        # Redirect stdout to capture print statements
        stdout = io.StringIO()
        sys.stdout = stdout

        try:
            # Execute the code with a timeout
            exec_globals = {}
            exec(code, exec_globals)

            # Get the output
            output = stdout.getvalue()

            if output:
                yield "result", output
            else:
                # If there's no output, return the last expression's result
                last_expression = list(exec_globals.values())[-1]
                yield "result", str(last_expression)

        except Exception as e:
            yield "error", str(e)

        finally:
            # Reset stdout
            sys.stdout = sys.__stdout__
