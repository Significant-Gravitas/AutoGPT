"""
Docker Sandbox Block — executes code in an isolated Docker container.

Provides a secure, reproducible execution environment for agent-generated code.
Supports Python, Node.js, Bash, and custom Docker images.
Automatically captures stdout, stderr, and exit codes.
"""

import logging
import os
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60  # seconds
DEFAULT_MEMORY_LIMIT = "512m"
DEFAULT_CPU_LIMIT = "1.0"


class SandboxLanguage(str, Enum):
    PYTHON = "python"
    NODEJS = "nodejs"
    BASH = "bash"
    CUSTOM = "custom"


LANGUAGE_IMAGES = {
    SandboxLanguage.PYTHON: "python:3.11-slim",
    SandboxLanguage.NODEJS: "node:20-slim",
    SandboxLanguage.BASH: "ubuntu:22.04",
    SandboxLanguage.CUSTOM: "",
}

LANGUAGE_COMMANDS = {
    SandboxLanguage.PYTHON: ["python3", "/sandbox/code.py"],
    SandboxLanguage.NODEJS: ["node", "/sandbox/code.js"],
    SandboxLanguage.BASH: ["bash", "/sandbox/code.sh"],
}

LANGUAGE_EXTENSIONS = {
    SandboxLanguage.PYTHON: "py",
    SandboxLanguage.NODEJS: "js",
    SandboxLanguage.BASH: "sh",
    SandboxLanguage.CUSTOM: "txt",
}


class DockerSandboxInput(BlockSchemaInput):
    code: str = SchemaField(
        description="Code to execute in the sandbox.",
    )
    language: SandboxLanguage = SchemaField(
        default=SandboxLanguage.PYTHON,
        description="Programming language / runtime.",
    )
    docker_image: str = SchemaField(
        default="",
        description="Custom Docker image (overrides language default if set).",
    )
    timeout_seconds: int = SchemaField(
        default=DEFAULT_TIMEOUT,
        description="Execution timeout in seconds.",
    )
    memory_limit: str = SchemaField(
        default=DEFAULT_MEMORY_LIMIT,
        description="Docker memory limit (e.g., '512m', '1g').",
    )
    cpu_limit: str = SchemaField(
        default=DEFAULT_CPU_LIMIT,
        description="Docker CPU limit (e.g., '1.0' for 1 CPU core).",
    )
    environment: dict = SchemaField(
        default_factory=dict,
        description="Environment variables to pass to the container.",
    )
    working_files: dict = SchemaField(
        default_factory=dict,
        description="Additional files to mount: {filename: content} dict.",
    )
    network_disabled: bool = SchemaField(
        default=True,
        description="Disable network access in the sandbox for security.",
    )


class DockerSandboxOutput(BlockSchemaOutput):
    stdout: str = SchemaField(description="Standard output from the executed code.")
    stderr: str = SchemaField(description="Standard error from the executed code.")
    exit_code: int = SchemaField(description="Exit code of the process.")
    success: bool = SchemaField(description="True if exit code is 0.")
    execution_time_ms: int = SchemaField(description="Execution time in milliseconds.")
    status: str = SchemaField(description="Status message.")


class DockerSandboxBlock(Block):
    """
    Executes agent-generated code in an isolated Docker container.

    Provides a secure sandbox with configurable resource limits, timeout,
    and optional network isolation. Supports Python, Node.js, Bash, and
    custom Docker images. Captures all output for the agent to analyze.
    """

    class Input(DockerSandboxInput):
        pass

    class Output(DockerSandboxOutput):
        pass

    def __init__(self):
        super().__init__(
            id="b8c9d0e1-f2a3-4567-bcde-890123456789",
            description=(
                "Executes code in a Docker-isolated sandbox with resource limits. "
                "Supports Python, Node.js, Bash, and custom images."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=DockerSandboxBlock.Input,
            output_schema=DockerSandboxBlock.Output,
            test_input={
                "code": "print('Hello from sandbox!')",
                "language": SandboxLanguage.PYTHON.value,
                "timeout_seconds": 10,
            },
            test_output=[
                ("success", True),
                ("stdout", "Hello from sandbox!\n"),
                ("exit_code", 0),
            ],
        )

    def run(self, input_data: Input, *, execution_stats=None, **kwargs) -> BlockOutput:
        import time

        # Determine Docker image
        image = input_data.docker_image
        if not image:
            image = LANGUAGE_IMAGES.get(input_data.language, "python:3.11-slim")

        # Determine file extension
        ext = LANGUAGE_EXTENSIONS.get(input_data.language, "txt")

        # Create temp directory for code files
        with tempfile.TemporaryDirectory(prefix="autogpt_sandbox_") as tmpdir:
            # Write main code file
            code_file = Path(tmpdir) / f"code.{ext}"
            code_file.write_text(input_data.code)

            # Write additional files
            for filename, content in input_data.working_files.items():
                safe_name = Path(filename).name  # Prevent path traversal
                (Path(tmpdir) / safe_name).write_text(content)

            # Build Docker command
            cmd = ["docker", "run", "--rm"]

            # Resource limits
            cmd += ["--memory", input_data.memory_limit]
            cmd += ["--cpus", input_data.cpu_limit]

            # Network
            if input_data.network_disabled:
                cmd += ["--network", "none"]

            # Security options
            cmd += ["--security-opt", "no-new-privileges"]
            cmd += ["--read-only"]
            cmd += ["--tmpfs", "/tmp:rw,noexec,nosuid,size=64m"]

            # Mount code directory
            cmd += ["-v", f"{tmpdir}:/sandbox:ro"]

            # Environment variables
            for key, value in input_data.environment.items():
                cmd += ["-e", f"{key}={value}"]

            # Image and command
            cmd += [image]
            lang_cmd = LANGUAGE_COMMANDS.get(input_data.language)
            if lang_cmd:
                cmd += lang_cmd
            else:
                cmd += ["cat", "/sandbox/code.txt"]

            start_time = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=input_data.timeout_seconds,
                )
                elapsed_ms = int((time.time() - start_time) * 1000)

                yield "stdout", result.stdout
                yield "stderr", result.stderr
                yield "exit_code", result.returncode
                yield "success", result.returncode == 0
                yield "execution_time_ms", elapsed_ms
                yield "status", (
                    f"Execution completed in {elapsed_ms}ms "
                    f"(exit code {result.returncode})."
                )

            except subprocess.TimeoutExpired:
                elapsed_ms = int((time.time() - start_time) * 1000)
                yield "stdout", ""
                yield "stderr", f"Execution timed out after {input_data.timeout_seconds}s."
                yield "exit_code", 124
                yield "success", False
                yield "execution_time_ms", elapsed_ms
                yield "status", f"Timeout after {input_data.timeout_seconds}s."

            except FileNotFoundError:
                yield "stdout", ""
                yield "stderr", "Docker not found. Ensure Docker is installed and running."
                yield "exit_code", 1
                yield "success", False
                yield "execution_time_ms", 0
                yield "status", "Docker not available."

            except Exception as e:
                yield "stdout", ""
                yield "stderr", str(e)
                yield "exit_code", 1
                yield "success", False
                yield "execution_time_ms", 0
                yield "status", f"Sandbox error: {e}"
