"""Commands to execute code"""

COMMAND_CATEGORY = "execute_code"
COMMAND_CATEGORY_TITLE = "Execute Code"

import logging
import os
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

import docker
from docker.errors import DockerException, ImageNotFound
from docker.models.containers import Container as DockerContainer

from autogpt.agents.agent import Agent
from autogpt.agents.utils.exceptions import (
    CodeExecutionError,
    CommandExecutionError,
    InvalidArgumentError,
    OperationNotAllowedError,
)
from autogpt.command_decorator import command
from autogpt.config import Config

from .decorators import sanitize_path_arg

logger = logging.getLogger(__name__)

ALLOWLIST_CONTROL = "allowlist"
DENYLIST_CONTROL = "denylist"


@command(
    "execute_python_code",
    "Executes the given Python code inside a single-use Docker container"
    " with access to your workspace folder",
    {
        "code": {
            "type": "string",
            "description": "The Python code to run",
            "required": True,
        },
    },
)
def execute_python_code(code: str, agent: Agent) -> str:
    """Create and execute a Python file in a Docker container and return the STDOUT of the
    executed code. If there is any data that needs to be captured use a print statement

    Args:
        code (str): The Python code to run
        name (str): A name to be given to the Python file

    Returns:
        str: The STDOUT captured from the code when it ran
    """

    tmp_code_file = NamedTemporaryFile(
        "w", dir=agent.workspace.root, suffix=".py", encoding="utf-8"
    )
    tmp_code_file.write(code)
    tmp_code_file.flush()

    try:
        return execute_python_file(tmp_code_file.name, agent)
    except Exception as e:
        raise CommandExecutionError(*e.args)
    finally:
        tmp_code_file.close()


@command(
    "execute_python_file",
    "Execute an existing Python file inside a single-use Docker container"
    " with access to your workspace folder",
    {
        "filename": {
            "type": "string",
            "description": "The name of te file to execute",
            "required": True,
        },
        "args": {
            "type": "list[str]",
            "description": "The (command line) arguments to pass to the script",
            "required": False,
        },
    },
)
@sanitize_path_arg("filename")
def execute_python_file(
    filename: Path, agent: Agent, args: list[str] | str = []
) -> str:
    """Execute a Python file in a Docker container and return the output

    Args:
        filename (Path): The name of the file to execute
        args (list, optional): The arguments with which to run the python script

    Returns:
        str: The output of the file
    """
    logger.info(
        f"Executing python file '{filename}' in working directory '{agent.config.workspace_path}'"
    )

    if isinstance(args, str):
        args = args.split()  # Convert space-separated string to a list

    if not str(filename).endswith(".py"):
        raise InvalidArgumentError("Invalid file type. Only .py files are allowed.")

    file_path = filename
    if not file_path.is_file():
        # Mimic the response that you get from the command line so that it's easier to identify
        raise FileNotFoundError(
            f"python: can't open file '{filename}': [Errno 2] No such file or directory"
        )

    if we_are_running_in_a_docker_container():
        logger.debug(
            f"Auto-GPT is running in a Docker container; executing {file_path} directly..."
        )
        result = subprocess.run(
            ["python", "-B", str(file_path)] + args,
            capture_output=True,
            encoding="utf8",
            cwd=str(agent.workspace.root),
        )
        if result.returncode == 0:
            return result.stdout
        else:
            raise CodeExecutionError(result.stderr)

    logger.debug("Auto-GPT is not running in a Docker container")
    try:
        client = docker.from_env()
        # You can replace this with the desired Python image/version
        # You can find available Python images on Docker Hub:
        # https://hub.docker.com/_/python
        image_name = "python:3-alpine"
        try:
            client.images.get(image_name)
            logger.debug(f"Image '{image_name}' found locally")
        except ImageNotFound:
            logger.info(
                f"Image '{image_name}' not found locally, pulling from Docker Hub..."
            )
            # Use the low-level API to stream the pull response
            low_level_client = docker.APIClient()
            for line in low_level_client.pull(image_name, stream=True, decode=True):
                # Print the status and progress, if available
                status = line.get("status")
                progress = line.get("progress")
                if status and progress:
                    logger.info(f"{status}: {progress}")
                elif status:
                    logger.info(status)

        logger.debug(f"Running {file_path} in a {image_name} container...")
        container: DockerContainer = client.containers.run(
            image_name,
            [
                "python",
                "-B",
                file_path.relative_to(agent.workspace.root).as_posix(),
            ] + args,
            volumes={
                str(agent.workspace.root): {
                    "bind": "/workspace",
                    "mode": "rw",
                }
            },
            working_dir="/workspace",
            stderr=True,
            stdout=True,
            detach=True,
        )  # type: ignore

        container.wait()
        logs = container.logs().decode("utf-8")
        container.remove()

        # print(f"Execution complete. Output: {output}")
        # print(f"Logs: {logs}")

        return logs

    except DockerException as e:
        logger.warn(
            "Could not run the script in a container. If you haven't already, please install Docker https://docs.docker.com/get-docker/"
        )
        raise CommandExecutionError(f"Could not run the script in a container: {e}")


def validate_command(command: str, config: Config) -> bool:
    """Validate a command to ensure it is allowed

    Args:
        command (str): The command to validate
        config (Config): The config to use to validate the command

    Returns:
        bool: True if the command is allowed, False otherwise
    """
    if not command:
        return False

    command_name = command.split()[0]

    if config.shell_command_control == ALLOWLIST_CONTROL:
        return command_name in config.shell_allowlist
    else:
        return command_name not in config.shell_denylist


@command(
    "execute_shell",
    "Execute a Shell Command, non-interactive commands only",
    {
        "command_line": {
            "type": "string",
            "description": "The command line to execute",
            "required": True,
        }
    },
    enabled=lambda config: config.execute_local_commands,
    disabled_reason="You are not allowed to run local shell commands. To execute"
    " shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' "
    "in your config file: .env - do not attempt to bypass the restriction.",
)
def execute_shell(command_line: str, agent: Agent) -> str:
    """Execute a shell command and return the output

    Args:
        command_line (str): The command line to execute

    Returns:
        str: The output of the command
    """
    if not validate_command(command_line, agent.config):
        logger.info(f"Command '{command_line}' not allowed")
        raise OperationNotAllowedError("This shell command is not allowed.")

    current_dir = Path.cwd()
    # Change dir into workspace if necessary
    if not current_dir.is_relative_to(agent.workspace.root):
        os.chdir(agent.workspace.root)

    logger.info(
        f"Executing command '{command_line}' in working directory '{os.getcwd()}'"
    )

    result = subprocess.run(command_line, capture_output=True, shell=True)
    output = f"STDOUT:\n{result.stdout.decode()}\nSTDERR:\n{result.stderr.decode()}"

    # Change back to whatever the prior working dir was
    os.chdir(current_dir)

    return output


@command(
    "execute_shell_popen",
    "Execute a Shell Command, non-interactive commands only",
    {
        "command_line": {
            "type": "string",
            "description": "The command line to execute",
            "required": True,
        }
    },
    lambda config: config.execute_local_commands,
    "You are not allowed to run local shell commands. To execute"
    " shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' "
    "in your config. Do not attempt to bypass the restriction.",
)
def execute_shell_popen(command_line: str, agent: Agent) -> str:
    """Execute a shell command with Popen and returns an english description
    of the event and the process id

    Args:
        command_line (str): The command line to execute

    Returns:
        str: Description of the fact that the process started and its id
    """
    if not validate_command(command_line, agent.config):
        logger.info(f"Command '{command_line}' not allowed")
        raise OperationNotAllowedError("This shell command is not allowed.")

    current_dir = Path.cwd()
    # Change dir into workspace if necessary
    if not current_dir.is_relative_to(agent.workspace.root):
        os.chdir(agent.workspace.root)

    logger.info(
        f"Executing command '{command_line}' in working directory '{os.getcwd()}'"
    )

    do_not_show_output = subprocess.DEVNULL
    process = subprocess.Popen(
        command_line, shell=True, stdout=do_not_show_output, stderr=do_not_show_output
    )

    # Change back to whatever the prior working dir was
    os.chdir(current_dir)

    return f"Subprocess started with PID:'{str(process.pid)}'"


def we_are_running_in_a_docker_container() -> bool:
    """Check if we are running in a Docker container

    Returns:
        bool: True if we are running in a Docker container, False otherwise
    """
    return os.path.exists("/.dockerenv")
