import io
import logging
import sys
import json
import multiprocessing
import subprocess
import time
import venv
import os
import tempfile
import shutil
from typing import Any, Union, Dict, List
from abc import ABC, abstractmethod

from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from autogpt_server.data.model import SchemaField


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasePythonExecutionBlock(Block, ABC):
    class Input(BlockSchema):
        code: str = SchemaField(
            description="Python code to execute", placeholder="print(f'Hello, {name}!')"
        )
        args: Union[Dict[str, Any], List[Dict[str, Any]], str] = SchemaField(
            description="Arguments to pass to the code. Can be a dictionary, list of dictionaries, or a JSON string.",
            default={},
            placeholder='{"name": "World", "number": 42}',
        )
        timeout: float = SchemaField(
            description="Execution timeout in seconds", default=30.0
        )

    class Output(BlockSchema):
        result: str = SchemaField(description="Execution result or output")
        error: str = SchemaField(description="Error message if execution failed")

    def __init__(self, block_id: str, description: str) -> None:
        super().__init__(
            id=block_id,
            description=description,
            categories={BlockCategory.BASIC},
            input_schema=self.Input,
            output_schema=self.Output,
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

    @abstractmethod
    def execute_code(self, code: str, args: Any, timeout: float) -> str:
        pass

    def run(self, input_data: Input) -> BlockOutput:
        try:
            result = self.execute_code(
                input_data.code, input_data.args, input_data.timeout
            )
            yield "result", result
        except Exception as e:
            yield "error", str(e)


class FastPythonExecutionBlock(BasePythonExecutionBlock):
    def __init__(self) -> None:
        super().__init__(
            block_id="ffb7dd8e-8a9e-42cd-a620-dc58a6c78d8c",
            description="This block executes Python code quickly using multiprocessing.",
        )

    def execute_code(self, code: str, args: Any, timeout: float) -> str:
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._execute_in_process, args=(code, args, result_queue)
        )
        process.start()
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            raise TimeoutError(f"Execution timed out after {timeout} seconds.")

        if not result_queue.empty():
            return result_queue.get()
        else:
            raise RuntimeError("Execution completed but no result was returned.")

    @staticmethod
    def _execute_in_process(
        code: str, args: Any, result_queue: multiprocessing.Queue
    ) -> None:
        try:
            stdout = io.StringIO()
            sys.stdout = stdout
            exec_globals = {
                "args": args,
                "print": lambda *a, **kw: print(*a, **kw, file=stdout),
            }
            exec(code, exec_globals)
            result_queue.put(stdout.getvalue())
        except Exception as e:
            result_queue.put(f"Error: {str(e)}")
        finally:
            sys.stdout = sys.__stdout__


class FlexiblePythonExecutionBlock(BasePythonExecutionBlock):
    def __init__(self) -> None:
        super().__init__(
            block_id="96e5a653-6b3b-46c3-87ed-56a7ff098f28",
            description="This block executes Python code with dynamic dependencies using virtual environments.",
        )
        self.venv_path = tempfile.mkdtemp()
        self.create_venv()

    def create_venv(self):
        venv.create(self.venv_path, with_pip=True)

    def install_dependencies(self, dependencies: List[str]) -> None:
        pip_path = os.path.join(self.venv_path, "bin", "pip")
        for pkg in dependencies:
            if pkg not in self.allowed_packages:
                raise ValueError(f"Package '{pkg}' is not in the allowed list.")
        subprocess.run(
            [pip_path, "install"] + dependencies, check=True, capture_output=True
        )

    def execute_code(self, code: str, args: Any, timeout: float) -> str:
        python_path = os.path.join(self.venv_path, "bin", "python")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as temp_file:
            temp_file.write(
                f"import json\nargs = json.loads('''{json.dumps(args)}''')\n"
            )
            temp_file.write(code)
        try:
            result = subprocess.run(
                [python_path, temp_file.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return (
                result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
            )
        finally:
            os.unlink(temp_file.name)

    def __del__(self):
        shutil.rmtree(self.venv_path, ignore_errors=True)


class UnifiedPythonExecutionBlock(BasePythonExecutionBlock):

    class Input(BasePythonExecutionBlock.Input):
        mode: str = SchemaField(
            description="Execution mode: 'fast' or 'flexible'", default="fast"
        )
        dependencies: List[str] = SchemaField(
            description="List of Python packages to install (only for 'flexible' mode)",
            default=[],
        )

    class Output(BasePythonExecutionBlock.Output):
        execution_time: float = SchemaField(description="Execution time in seconds")
        memory_usage: float = SchemaField(description="Peak memory usage in MB")

    def __init__(self) -> None:
        super().__init__(
            block_id="d72ba52d-45ee-4c3b-baa0-75bafcae0183",
            description="This block executes Python code in either 'fast' or 'flexible' mode, with optional dependencies.",
        )
        self.fast_executor = FastPythonExecutionBlock()
        self.flexible_executor = FlexiblePythonExecutionBlock()

    def execute_code(self, code: str, args: Any, timeout: float) -> str:
        self.start_time = time.time()
        peak_memory = 0

        try:
            if self.input_data.mode == "fast":
                result = self.fast_executor.execute_code(code, args, timeout)
            elif self.input_data.mode == "flexible":
                self.flexible_executor.install_dependencies(
                    self.input_data.dependencies
                )
                result = self.flexible_executor.execute_code(code, args, timeout)
            else:
                raise ValueError(f"Invalid mode: {self.input_data.mode}")

            peak_memory = self.get_peak_memory_usage()
            return result
        finally:
            execution_time = time.time() - self.start_time
            logger.info(f"Execution completed in {execution_time:.2f} seconds")
            logger.info(f"Peak memory usage: {peak_memory:.2f} MB")

    def run(self, input_data: Input) -> BlockOutput:
        self.input_data = input_data
        self.validate_input()

        try:
            result = self.execute_code(
                input_data.code, input_data.args, input_data.timeout
            )
            yield "result", result
            yield "execution_time", time.time() - self.start_time
            yield "memory_usage", self.get_peak_memory_usage()
        except Exception as e:
            logger.error(f"Error during execution: {str(e)}")
            yield "error", str(e)

    def validate_input(self):
        if self.input_data.mode not in ["fast", "flexible"]:
            raise ValueError(f"Invalid mode: {self.input_data.mode}")

        if self.input_data.mode == "flexible":
            for pkg in self.input_data.dependencies:
                if pkg not in self.allowed_packages:
                    raise ValueError(f"Package '{pkg}' is not in the allowed list.")

    @staticmethod
    def get_peak_memory_usage():
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # Convert to MB
