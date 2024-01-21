"""Tools to execute code"""


import os
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

import docker
from docker.errors import DockerException, ImageNotFound, NotFound
from docker.models.containers import Container as DockerContainer

from AFAAS.core.tools.tool_decorator import SAFE_MODE, tool
from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.interfaces.tools.base import AbstractTool
from AFAAS.lib.sdk.errors import (
    CodeExecutionError,
    InvalidArgumentError,
    ToolExecutionError,
)
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.task.task import Task
from AFAAS.lib.utils.json_schema import JSONSchema

from .file_operations_utils import sanitize_path_arg

LOG = AFAASLogger(name=__name__)
TOOL_CATEGORY = "execute_code"


ALLOWLIST_CONTROL = "allowlist"
DENYLIST_CONTROL = "denylist"


@tool(
    name="execute_python_code",
    description="Executes the given Python code inside a single-use Docker container"
    " with access to your workspace folder",
    parameters={
        "code": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The Python code to run",
            required=True,
        ),
    },
    categories=[AbstractTool.FRAMEWORK_CATEGORY, TOOL_CATEGORY],
)
def execute_python_code(code: str, task: Task, agent: BaseAgent) -> str:
    """
    Create and execute a Python file in a Docker container and return the STDOUT of the
    executed code.

    If the code generates any data that needs to be captured, use a print statement.

    Args:
        code (str): The Python code to run.
        agent (Agent): The Agent executing the command.

    Returns:
        str: The STDOUT captured from the code when it ran.
    """

    tmp_code_file = NamedTemporaryFile(
        "w", dir=agent.workspace.root, suffix=".py", encoding="utf-8"
    )
    tmp_code_file.write(code)
    tmp_code_file.flush()

    try:
        return execute_python_file(tmp_code_file.name, agent)  # type: ignore
    except Exception as e:
        raise ToolExecutionError(*e.args)
    finally:
        tmp_code_file.close()


@tool(
    "execute_python_file",
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
@sanitize_path_arg("filename")
def execute_python_file(
    filename: Path, agent: BaseAgent, args: list[str] | str = []
) -> str:
    """Execute a Python file in a Docker container and return the output

    Args:
        filename (Path): The name of the file to execute
        args (list, optional): The arguments with which to run the python script

    Returns:
        str: The output of the file
    """
    LOG.info(
        f"Executing python file '{filename}' "
        f"in working directory '{agent.workspace.root}'"
    )

    if isinstance(args, str):
        args = args.split()  # Convert space-separated string to a list

    if not str(filename).endswith(".py"):
        raise InvalidArgumentError("Invalid file type. Only .py files are allowed.")

    file_path = filename
    if not file_path.is_file():
        # Mimic the response that you get from the command line to make it
        # intuitively understandable for the LLM
        raise FileNotFoundError(
            f"python: can't open file '{filename}': [Errno 2] No such file or directory"
        )

    if we_are_running_in_a_docker_container():
        LOG.trace(
            "AutoGPT is running in a Docker container; "
            f"executing {file_path} directly..."
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

    LOG.trace("AutoGPT is not running in a Docker container")
    try:
        assert agent.state.agent_id, "Need Agent ID to attach Docker container"

        client = docker.from_env()
        # You can replace this with the desired Python image/version
        # You can find available Python images on Docker Hub:
        # https://hub.docker.com/_/python
        image_name = "python:3-alpine"
        container_is_fresh = False
        container_name = f"{agent.state.agent_id}_sandbox"
        try:
            container: DockerContainer = client.containers.get(
                container_name
            )  # type: ignore
        except NotFound:
            try:
                client.images.get(image_name)
                LOG.trace(f"Image '{image_name}' found locally")
            except ImageNotFound:
                LOG.info(
                    f"Image '{image_name}' not found locally,"
                    " pulling from Docker Hub..."
                )
                # Use the low-level API to stream the pull response
                low_level_client = docker.APIClient()
                for line in low_level_client.pull(image_name, stream=True, decode=True):
                    # Print the status and progress, if available
                    status = line.get("status")
                    progress = line.get("progress")
                    if status and progress:
                        LOG.info(f"{status}: {progress}")
                    elif status:
                        LOG.info(status)

            LOG.trace(f"Creating new {image_name} container...")
            container: DockerContainer = client.containers.run(
                image_name,
                ["sleep", "60"],  # Max 60 seconds to prevent permanent hangs
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
                name=container_name,
            )  # type: ignore
            container_is_fresh = True

        if not container.status == "running":
            container.start()
        elif not container_is_fresh:
            container.restart()

        LOG.trace(f"Running {file_path} in container {container.name}...")
        exec_result = container.exec_run(
            [
                "python",
                "-B",
                file_path.relative_to(agent.workspace.root).as_posix(),
            ]
            + args,
            stderr=True,
            stdout=True,
        )

        if exec_result.exit_code != 0:
            raise CodeExecutionError(exec_result.output.decode("utf-8"))

        return exec_result.output.decode("utf-8")

    except DockerException as e:
        LOG.warning(
            "Could not run the script in a container. "
            "If you haven't already, please install Docker: "
            "https://docs.docker.com/get-docker/"
        )
        raise ToolExecutionError(f"Could not run the script in a container: {e}")


# TODO Implement security
# def validate_command(command: str) -> bool:
#     """Validate a command to ensure it is allowed

#     Args:
#         command (str): The command to validate
#         config (Config): The config to use to validate the command

#     Returns:
#         bool: True if the command is allowed, False otherwise
#     """
#     if not command:
#         return False

#     command_name = command.split()[0]

#     if config.shell_command_control == ALLOWLIST_CONTROL:
#         return command_name in config.shell_allowlist
#     else:
#         return command_name not in config.shell_denylist


@tool(
    "execute_shell",
    "Execute a Shell Tool, non-interactive commands only",
    {
        "command_line": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The command line to execute",
            required=True,
        )
    },
    enabled=lambda config: config.execute_local_commands,
    disabled_reason="You are not allowed to run local shell commands. To execute"
    " shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' "
    "in your config file: .env - do not attempt to bypass the restriction.",
)
def execute_shell(command_line: str, task: Task, agent: BaseAgent) -> str:
    """Execute a shell command and return the output

    Args:
        command_line (str): The command line to execute

    Returns:
        str: The output of the command
    """
    # TODO IMPLEMENT OR NO SECURITY MECHANISM
    # if not validate_command(command_line, agent.legacy_config):
    #     LOG.info(f"Tool '{command_line}' not allowed")
    #     raise OperationNotAllowedError("This shell command is not allowed.")

    current_dir = Path.cwd()
    # Change dir into workspace if necessary
    if not current_dir.is_relative_to(agent.workspace.root):
        os.chdir(agent.workspace.root)

    LOG.info(f"Executing command '{command_line}' in working directory '{os.getcwd()}'")

    result = subprocess.run(command_line, capture_output=True, shell=True)
    output = f"STDOUT:\n{result.stdout.decode()}\nSTDERR:\n{result.stderr.decode()}"

    # Change back to whatever the prior working dir was
    os.chdir(current_dir)

    return output


@tool(
    "execute_shell_popen",
    "Execute a Shell Tool, non-interactive commands only",
    {
        "command_line": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The command line to execute",
            required=True,
        )
    },
    lambda config: config.execute_local_commands,
    "You are not allowed to run local shell commands. To execute"
    " shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' "
    "in your config. Do not attempt to bypass the restriction.",
)
def execute_shell_popen(command_line: str, task: Task, agent: BaseAgent) -> str:
    """Execute a shell command with Popen and returns an english description
    of the event and the process id

    Args:
        command_line (str): The command line to execute

    Returns:
        str: Description of the fact that the process started and its id
    """
    # if not validate_command(command_line, agent.legacy_config):
    #     LOG.info(f"Tool '{command_line}' not allowed")
    #     raise OperationNotAllowedError("This shell command is not allowed.")

    current_dir = Path.cwd()
    # Change dir into workspace if necessary
    if not current_dir.is_relative_to(agent.workspace.root):
        os.chdir(agent.workspace.root)

    LOG.info(f"Executing command '{command_line}' in working directory '{os.getcwd()}'")

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
