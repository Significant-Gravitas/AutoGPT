import io
import sys
import json
import multiprocessing
from typing import Any, Union, Dict, List

from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from autogpt_server.data.model import SchemaField


import subprocess
import venv
import os
import tempfile
import shutil


class PythonExecutionBlock(Block):
    class Input(BlockSchema):
        code: str = SchemaField(
            description="Python code to execute", placeholder="print(f'Hello, {name}!')"
        )
        args: Union[Dict[str, Any], List[Dict[str, Any]]] = SchemaField(
            description="Arguments to pass to the code. Can be a dictionary or a list of dictionaries.",
            default={},
            placeholder='{"name": "World", "number": 42}',
        )
        timeout: float = SchemaField(
            description="Execution timeout in seconds", default=5.0
        )

    class Output(BlockSchema):
        result: str = SchemaField(description="Execution result or output")
        error: str = SchemaField(description="Error message if execution failed")

    def __init__(self) -> None:
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
                {
                    "code": "print(json.dumps(data, indent=2))",
                    "args": [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}],
                    "timeout": 5.0,
                },
            ],
            test_output=[
                ("result", "Hello, Alice! Your number is 42.\n"),
                ("error", "Execution timed out after 2.0 seconds."),
                (
                    "result",
                    '[\n  {\n    "name": "John",\n    "age": 30\n  },\n  {\n    "name": "Jane",\n    "age": 25\n  }\n]\n',
                ),
            ],
        )

    @staticmethod
    def _execute_code(
        code: str,
        args: Union[Dict[str, Any], List[Dict[str, Any]]],
        result_queue: multiprocessing.Queue,
    ) -> None:
        try:
            # Redirect stdout to capture print statements
            stdout = io.StringIO()
            sys.stdout = stdout

            # Prepare the execution environment with the provided args
            exec_globals: Dict[str, Any] = {
                "json": json,
                "print": lambda *args, **kwargs: print(*args, **kwargs, file=stdout),
            }
            if isinstance(args, dict):
                exec_globals.update(args)
            elif isinstance(args, list):
                exec_globals["data"] = args

            # Execute the code
            exec(code, exec_globals)

            # Get the output
            output: str = stdout.getvalue()

            if output:
                result_queue.put(("result", output))
            else:
                # If there's no output, return the last expression's result
                last_expression: Any = list(exec_globals.values())[-1]
                result_queue.put(("result", str(last_expression)))

        except Exception as e:
            result_queue.put(("error", f"Execution error: {str(e)}"))

        finally:
            # Reset stdout
            sys.stdout = sys.__stdout__

    def run(self, input_data: Input) -> BlockOutput:
        code: str = input_data.code
        args: Union[Dict[str, Any], List[Dict[str, Any]]] = input_data.args
        timeout: float = input_data.timeout

        # Create a multiprocessing Queue to get the result
        result_queue: multiprocessing.Queue = multiprocessing.Queue()

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


class AdvancedPythonExecutionBlock(Block):
    class Input(BlockSchema):
        code: str = SchemaField(
            description="Python code to execute",
            placeholder="import pandas as pd\nprint(pd.DataFrame({'A': [1, 2, 3]}))",
        )
        dependencies: List[str] = SchemaField(
            description="List of Python packages to install",
            default=[],
            placeholder="['pandas', 'numpy']",
        )
        args: Union[Dict[str, Any], List[Dict[str, Any]], str] = SchemaField(
            description="Arguments to pass to the code. Can be a dictionary, list of dictionaries, or a JSON string.",
            default={},
            placeholder='{"data": [{"A": 1, "B": 2}, {"A": 3, "B": 4}]}',
        )
        timeout: float = SchemaField(
            description="Execution timeout in seconds", default=60.0
        )

    class Output(BlockSchema):
        result: str = SchemaField(description="Execution result or output")
        error: str = SchemaField(description="Error message if execution failed")

    def __init__(self) -> None:
        super().__init__(
            id="b2c3d4e5-f6g7-h8i9-j0k1-l2m3n4o5p6q7",
            description="This block executes Python code with dynamic dependencies, enforces a timeout, and returns the output or any error messages.",
            categories={BlockCategory.BASIC},
            input_schema=AdvancedPythonExecutionBlock.Input,
            output_schema=AdvancedPythonExecutionBlock.Output,
        )
        self.allowed_packages = set(
            [
                "pandas",
                "numpy",
                "matplotlib",
                "scipy",
                "sklearn",
                "requests",
                "beautifulsoup4",
                "lxml",
                "pyyaml",
                "jinja2",
            ]
        )
        self.venv_path = tempfile.mkdtemp()
        self.create_venv()

    def create_venv(self):
        venv.create(self.venv_path, with_pip=True)

    def install_dependencies(self, dependencies: List[str]) -> str:
        pip_path = os.path.join(self.venv_path, "bin", "pip")
        for pkg in dependencies:
            if pkg not in self.allowed_packages:
                return f"Error: Package '{pkg}' is not in the allowed list."

        try:
            subprocess.run(
                [pip_path, "install"] + dependencies, check=True, capture_output=True
            )
            return "Dependencies installed successfully."
        except subprocess.CalledProcessError as e:
            return f"Error installing dependencies: {e.stderr.decode()}"

    def execute_code(
        self, code: str, args: Union[Dict[str, Any], List[Dict[str, Any]], str]
    ) -> str:
        python_path = os.path.join(self.venv_path, "bin", "python")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as temp_file:
            temp_file.write(f"import sys\nimport json\n\n")
            temp_file.write(f"args = json.loads('''{json.dumps(args)}''')\n\n")
            temp_file.write(code)

        try:
            result = subprocess.run(
                [python_path, temp_file.name],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            os.unlink(temp_file.name)
            return (
                result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
            )
        except subprocess.TimeoutExpired:
            os.unlink(temp_file.name)
            return f"Error: Execution timed out after {self.timeout} seconds."

    def run(self, input_data: Input) -> BlockOutput:
        self.timeout = input_data.timeout

        # Install dependencies
        install_result = self.install_dependencies(input_data.dependencies)
        if install_result.startswith("Error"):
            yield "error", install_result
            return

        # Execute code
        execution_result = self.execute_code(input_data.code, input_data.args)
        if execution_result.startswith("Error"):
            yield "error", execution_result
        else:
            yield "result", execution_result

    def __del__(self):
        # Clean up the virtual environment
        shutil.rmtree(self.venv_path, ignore_errors=True)
