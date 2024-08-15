import io
import sys
import multiprocessing
from typing import Any

from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from autogpt_server.data.model import SchemaField


class PythonExecutionBlock(Block):
    class Input(BlockSchema):
        code: str = SchemaField(
            description="Python code to execute", placeholder="print(f'Hello, {name}!')"
        )
        args: dict[str, Any] = SchemaField(
            description="Arguments to pass to the code",
            default={},
            placeholder='{"name": "World", "number": 42}',
        )
        timeout: float = SchemaField(
            description="Execution timeout in seconds", default=5.0
        )

    class Output(BlockSchema):
        result: str = SchemaField(description="Execution result or output")
        error: str = SchemaField(description="Error message if execution failed")

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6",
            description="This block executes Python code with provided arguments, enforces a timeout, and returns the output or any error messages.",
            categories={BlockCategory.BASIC},
            input_schema=PythonExecutionBlock.Input,
            output_schema=PythonExecutionBlock.Output,
            test_input=[
                {
                    "code": "print(f'Hello, {name}! Your number is {number}.')",
                    "args": {"name": "Alice", "number": 42},
                    "timeout": 5.0,
                },
                {
                    "code": "import time\ntime.sleep(10)\nprint('This should timeout')",
                    "timeout": 2.0,
                },
            ],
            test_output=[
                ("result", "Hello, Alice! Your number is 42.\n"),
                ("error", "Execution timed out after 2.0 seconds."),
            ],
        )

    @staticmethod
    def _execute_code(code: str, args: dict, result_queue: multiprocessing.Queue):
        try:
            # Redirect stdout to capture print statements
            stdout = io.StringIO()
            sys.stdout = stdout

            # Prepare the execution environment with the provided args
            exec_globals = args.copy()

            # Execute the code
            exec(code, exec_globals)

            # Get the output
            output = stdout.getvalue()

            if output:
                result_queue.put(("result", output))
            else:
                # If there's no output, return the last expression's result
                last_expression = list(exec_globals.values())[-1]
                result_queue.put(("result", str(last_expression)))

        except Exception as e:
            result_queue.put(("error", str(e)))

        finally:
            # Reset stdout
            sys.stdout = sys.__stdout__

    def run(self, input_data: Input) -> BlockOutput:
        code = input_data.code
        args = input_data.args
        timeout = input_data.timeout

        # Create a multiprocessing Queue to get the result
        result_queue = multiprocessing.Queue()

        # Start the process
        process = multiprocessing.Process(
            target=self._execute_code, args=(code, args, result_queue)
        )
        process.start()

        # Wait for the process to complete or timeout
        process.join(timeout)

        if process.is_alive():
            # If the process is still running after the timeout, terminate it
            process.terminate()
            process.join()
            yield "error", f"Execution timed out after {timeout} seconds."
        elif not result_queue.empty():
            # If the process completed and we have a result, yield it
            result_type, result_value = result_queue.get()
            yield result_type, result_value
        else:
            # If the process completed but we don't have a result, it's an error
            yield "error", "Execution completed but no result was returned."
