import logging
import os
import random
import shlex
import string
import subprocess
from pathlib import Path
from typing import Iterator

import docker
from docker.errors import DockerException, ImageNotFound, NotFound
from docker.models.containers import Container as DockerContainer

from autogpt.agents.base import BaseAgentSettings
from autogpt.agents.protocols import CommandProvider
from autogpt.command_decorator import command
from autogpt.config import Config
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.file_storage.base import FileStorage
from autogpt.models.command import Command
from autogpt.utils.exceptions import (
    CodeExecutionError,
    CommandExecutionError,
    InvalidArgumentError,
    OperationNotAllowedError,
)

logger = logging.getLogger(__name__)

ALLOWLIST_CONTROL = "allowlist"
DENYLIST_CONTROL = "denylist"


def we_are_running_in_a_docker_container() -> bool:
    """Check if we are running in a Docker container

    Returns:
        bool: True if we are running in a Docker container, False otherwise
    """
    return os.path.exists("/.dockerenv")


def is_docker_available() -> bool:
    """Check if Docker is available and supports Linux containers

    Returns:
        bool: True if Docker is available and supports Linux containers, False otherwise
    """
    try:
        client = docker.from_env()
        docker_info = client.info()
        return docker_info["OSType"] == "linux"
    except Exception:
        return False


class CodeExecutorComponent(CommandProvider):
    """Provides commands to execute Python code and shell commands."""

    def __init__(
        self, workspace: FileStorage, state: BaseAgentSettings, config: Config
    ):
        self.workspace = workspace
        self.state = state
        self.legacy_config = config

        if not we_are_running_in_a_docker_container() and not is_docker_available():
            logger.info(
                "Docker is not available or does not support Linux containers. "
                "The code execution commands will not be available."
            )

        if not self.legacy_config.execute_local_commands:
            logger.info(
                "Local shell commands are disabled. To enable them,"
                " set EXECUTE_LOCAL_COMMANDS to 'True' in your config file."
            )

    def get_commands(self) -> Iterator[Command]:
        if we_are_running_in_a_docker_container() or is_docker_available():
            yield self.execute_python_code
            yield self.execute_python_file

        if self.legacy_config.execute_local_commands:
            yield self.execute_shell
            yield self.execute_shell_popen

    @command(
        ["execute_python_code"],
        "Executes the given Python code inside a single-use Docker container"
        " with access to your workspace folder",
        {
            "code": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The Python code to run",
                required=True,
            ),
        },
    )
    async def execute_python_code(self, code: str) -> str:
        """
        Create and execute a Python file in a Docker container
        and return the STDOUT of the executed code.

        If the code generates any data that needs to be captured,
        use a print statement.

        Args:
            code (str): The Python code to run.
            agent (Agent): The Agent executing the command.

        Returns:
            str: The STDOUT captured from the code when it ran.
        """

        temp_path = ""
        while True:
            temp_path = f"temp{self._generate_random_string()}.py"
            if not self.workspace.exists(temp_path):
                break
        await self.workspace.write_file(temp_path, code)

        try:
            return self.execute_python_file(temp_path)
        except Exception as e:
            raise CommandExecutionError(*e.args)
        finally:
            self.workspace.delete_file(temp_path)

    @command(
        ["execute_python_file"],
        "Execute an existing Python file inside a single-use Docker container"
        " with access to your workspace folder",
        {
            "filename": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The name of the file to execute",
                required=True,
            ),
            "args": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                description="The (command line) arguments to pass to the script",
                required=False,
                items=JSONSchema(type=JSONSchema.Type.STRING),
            ),
        },
    )
    def execute_python_file(self, filename: str, args: list[str] = []) -> str:
        """Execute a Python file in a Docker container and return the output

        Args:
            filename (Path): The name of the file to execute
            args (list, optional): The arguments with which to run the python script

        Returns:
            str: The output of the file
        """
        logger.info(f"Executing python file '{filename}'")

        if not str(filename).endswith(".py"):
            raise InvalidArgumentError("Invalid file type. Only .py files are allowed.")

        file_path = self.workspace.get_path(filename)
        if not self.workspace.exists(file_path):
            # Mimic the response that you get from the command line to make it
            # intuitively understandable for the LLM
            raise FileNotFoundError(
                f"python: can't open file '{filename}': "
                f"[Errno 2] No such file or directory"
            )

        if we_are_running_in_a_docker_container():
            logger.debug(
                "AutoGPT is running in a Docker container; "
                f"executing {file_path} directly..."
            )
            with self.workspace.mount() as local_path:
                result = subprocess.run(
                    ["python", "-B", str(file_path)] + args,
                    capture_output=True,
                    encoding="utf8",
                    cwd=str(local_path),
                )
                if result.returncode == 0:
                    return result.stdout
                else:
                    raise CodeExecutionError(result.stderr)

        logger.debug("AutoGPT is not running in a Docker container")
        return self._run_python_code_in_docker(filename, args)

    def validate_command(self, command_line: str, config: Config) -> tuple[bool, bool]:
        """Check whether a command is allowed and whether it may be executed in a shell.

        If shell command control is enabled, we disallow executing in a shell, because
        otherwise the model could circumvent the command filter using shell features.

        Args:
            command_line (str): The command line to validate
            config (Config): The app config including shell command control settings

        Returns:
            bool: True if the command is allowed, False otherwise
            bool: True if the command may be executed in a shell, False otherwise
        """
        if not command_line:
            return False, False

        command_name = shlex.split(command_line)[0]

        if config.shell_command_control == ALLOWLIST_CONTROL:
            return command_name in config.shell_allowlist, False
        elif config.shell_command_control == DENYLIST_CONTROL:
            return command_name not in config.shell_denylist, False
        else:
            return True, True

    @command(
        ["execute_shell"],
        "Execute a Shell Command, non-interactive commands only",
        {
            "command_line": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The command line to execute",
                required=True,
            )
        },
    )
    def execute_shell(self, command_line: str) -> str:
        """Execute a shell command and return the output

        Args:
            command_line (str): The command line to execute

        Returns:
            str: The output of the command
        """
        allow_execute, allow_shell = self.validate_command(
            command_line, self.legacy_config
        )
        if not allow_execute:
            logger.info(f"Command '{command_line}' not allowed")
            raise OperationNotAllowedError("This shell command is not allowed.")

        current_dir = Path.cwd()
        # Change dir into workspace if necessary
        if not current_dir.is_relative_to(self.workspace.root):
            os.chdir(self.workspace.root)

        logger.info(
            f"Executing command '{command_line}' in working directory '{os.getcwd()}'"
        )

        result = subprocess.run(
            command_line if allow_shell else shlex.split(command_line),
            capture_output=True,
            shell=allow_shell,
        )
        output = f"STDOUT:\n{result.stdout.decode()}\nSTDERR:\n{result.stderr.decode()}"

        # Change back to whatever the prior working dir was
        os.chdir(current_dir)

        return output

    @command(
        ["execute_shell_popen"],
        "Execute a Shell Command, non-interactive commands only",
        {
            "command_line": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The command line to execute",
                required=True,
            )
        },
    )
    def execute_shell_popen(self, command_line: str) -> str:
        """Execute a shell command with Popen and returns an english description
        of the event and the process id

        Args:
            command_line (str): The command line to execute

        Returns:
            str: Description of the fact that the process started and its id
        """
        allow_execute, allow_shell = self.validate_command(
            command_line, self.legacy_config
        )
        if not allow_execute:
            logger.info(f"Command '{command_line}' not allowed")
            raise OperationNotAllowedError("This shell command is not allowed.")

        current_dir = Path.cwd()
        # Change dir into workspace if necessary
        if not current_dir.is_relative_to(self.workspace.root):
            os.chdir(self.workspace.root)

        logger.info(
            f"Executing command '{command_line}' in working directory '{os.getcwd()}'"
        )

        do_not_show_output = subprocess.DEVNULL
        process = subprocess.Popen(
            command_line if allow_shell else shlex.split(command_line),
            shell=allow_shell,
            stdout=do_not_show_output,
            stderr=do_not_show_output,
        )

        # Change back to whatever the prior working dir was
        os.chdir(current_dir)

        return f"Subprocess started with PID:'{str(process.pid)}'"

    def _run_python_code_in_docker(self, filename: str, args: list[str]) -> str:
        """Run a Python script in a Docker container"""
        file_path = self.workspace.get_path(filename)
        try:
            assert self.state.agent_id, "Need Agent ID to attach Docker container"

            client = docker.from_env()
            image_name = "python:3-alpine"
            container_is_fresh = False
            container_name = f"{self.state.agent_id}_sandbox"
            with self.workspace.mount() as local_path:
                try:
                    container: DockerContainer = client.containers.get(
                        container_name
                    )  # type: ignore
                except NotFound:
                    try:
                        client.images.get(image_name)
                        logger.debug(f"Image '{image_name}' found locally")
                    except ImageNotFound:
                        logger.info(
                            f"Image '{image_name}' not found locally,"
                            " pulling from Docker Hub..."
                        )
                        # Use the low-level API to stream the pull response
                        low_level_client = docker.APIClient()
                        for line in low_level_client.pull(
                            image_name, stream=True, decode=True
                        ):
                            # Print the status and progress, if available
                            status = line.get("status")
                            progress = line.get("progress")
                            if status and progress:
                                logger.info(f"{status}: {progress}")
                            elif status:
                                logger.info(status)

                    logger.debug(f"Creating new {image_name} container...")
                    container: DockerContainer = client.containers.run(
                        image_name,
                        ["sleep", "60"],  # Max 60 seconds to prevent permanent hangs
                        volumes={
                            str(Path(local_path).absolute()): {
                                "bind": "/workspace",
                                "mode": "rw",
                            }
                        },
                        working_dir="/workspace",
                        stderr=True,
                        stdout=True,
                        detach=True,
                        name=container_name,
                    )  # type: ignore
                    container_is_fresh = True

                if not container.status == "running":
                    container.start()
                elif not container_is_fresh:
                    container.restart()

                logger.debug(f"Running {file_path} in container {container.name}...")

                exec_result = container.exec_run(
                    [
                        "python",
                        "-B",
                        filename,
                    ]
                    + args,
                    stderr=True,
                    stdout=True,
                )

                if exec_result.exit_code != 0:
                    raise CodeExecutionError(exec_result.output.decode("utf-8"))

                return exec_result.output.decode("utf-8")

        except DockerException as e:
            logger.warning(
                "Could not run the script in a container. "
                "If you haven't already, please install Docker: "
                "https://docs.docker.com/get-docker/"
            )
            raise CommandExecutionError(f"Could not run the script in a container: {e}")

    def _generate_random_string(self, length: int = 8):
        # Create a string of all letters and digits
        characters = string.ascii_letters + string.digits
        # Use random.choices to generate a random string
        random_string = "".join(random.choices(characters, k=length))
        return random_string
