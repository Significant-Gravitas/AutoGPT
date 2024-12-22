import os
import subprocess
import hashlib
import tempfile
from pathlib import Path
import shutil
import ast
import sysconfig
import importlib

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField

class PythonExecutorBlock(Block):
    class Input(BlockSchema):
        script_content: str = SchemaField(description="Python code to be executed")

    class Output(BlockSchema):
        stdout: str = SchemaField(description="Execution output")
        stderr: str = SchemaField(description="Execution errors")

    def __init__(self):
        super().__init__(
            id="5e489acd-a955-4dd4-9d46-b975c2162dc2",
            description="Executes arbitrary Python code",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=PythonExecutorBlock.Input,
            output_schema=PythonExecutorBlock.Output,
            test_input={
                "script_content": """print("Hello, world!")""",
            },
            test_output=[("stdout", "Hello, world!"), ("stderr", "")],
            test_mock={"execute_code": lambda script_content: ""},
        )
        self.prepare()

    @staticmethod
    def prepare():
        current_directory = os.path.dirname(os.path.abspath(__file__))
        source_file = os.path.join(current_directory, '../../credentials/python_executor_credentials.json')

        destination_dir = os.path.expanduser('~/.config/gspread')  # Expands '~' to the full home directory path
        os.makedirs(destination_dir, exist_ok=True)

        destination_file = os.path.join(destination_dir, 'service_account.json')
        shutil.copy(source_file, destination_file)

    @staticmethod
    def execute_code(script_content: str) -> tuple[str, str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_script_path = Path(temp_dir) / (hashlib.md5(script_content.encode()).hexdigest() + ".py")
            with open(temp_script_path, "w") as script_file:
                script_file.write(script_content)
            return PythonExecutorBlock.execute_script_in_isolated_env(temp_script_path)
    
    def execute_script_in_isolated_env(script_path):
        script_path = Path(script_path).resolve()

        if not script_path.exists() or not script_path.is_file():
            raise FileNotFoundError(f"Script file '{script_path}' does not exist.")
    
        requirements = PythonExecutorBlock.get_dependencies(script_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a virtual environment
            venv_dir = Path(temp_dir)  
            runner_file_path = Path(temp_dir) / f"runner.sh"
            temp_script_content = f"""#!/bin/bash
set -e
python3 -m venv "{venv_dir}"
source "{venv_dir}/bin/activate"
"""

            if requirements:
                temp_script_content += f"python -m pip install --quiet --disable-pip-version-check --ignore-installed {' '.join(requirements)}\n"

            temp_script_content += f"python {script_path}\n"

            with open(runner_file_path, "w") as script_file:
                script_file.write(temp_script_content)
                os.chmod(runner_file_path, 0o770)

            result = subprocess.run(
                f"bash -c {runner_file_path}",
                shell=True,
                capture_output=True,
                text=True,
            )

            # Returning stdout and stderr after subprocess ends
            return result.stdout, result.stderr

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        stdout, stderr = self.execute_code(input_data.script_content)
        yield "stdout", stdout
        yield "stderr", stderr

    @staticmethod
    def get_dependencies(filename):
        # Parse the Python file
        with open(filename, 'r') as f:
            tree = ast.parse(f.read())
        
        # Extract all import statements
        dependencies = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                dependencies.add(node.module)
        
        # Remove standard library imports
        external_dependencies = [
            dep for dep in dependencies if not PythonExecutorBlock.is_standard_lib(dep)
        ]

        return list(set(external_dependencies))

    @staticmethod
    def is_standard_lib(dep):
        try:
            # Try to import and check if it's part of the standard library
            module = importlib.import_module(dep)
            return sysconfig.get_paths()['stdlib'] in module.__file__
        except ModuleNotFoundError:
            return False

