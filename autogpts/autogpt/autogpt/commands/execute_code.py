"""Commands to execute code"""

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import docker
from docker.errors import DockerException, ImageNotFound, NotFound
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
from autogpt.core.utils.json_schema import JSONSchema

from .decorators import run_in_workspace, sanitize_path_arg

COMMAND_CATEGORY = "execute_code"
COMMAND_CATEGORY_TITLE = "Execute Code"


logger = logging.getLogger(__name__)

ALLOWLIST_CONTROL = "allowlist"
DENYLIST_CONTROL = "denylist"
TIMEOUT_SECONDS: int = 900


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


@command(
    "execute_python_code",
    "Executes the given Python code inside a single-use Docker container"
    " with access to your workspace folder",
    {
        "code": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The Python code to run",
            required=True,
        ),
    },
    disabled_reason="To execute python code agent "
    "must be running in a Docker container or "
    "Docker must be available on the system.",
    available=we_are_running_in_a_docker_container() or is_docker_available(),
)
def execute_python_code(code: str, agent: Agent) -> str:
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
        raise CommandExecutionError(*e.args)
    finally:
        tmp_code_file.close()


@command(
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
    disabled_reason="To execute python code agent "
    "must be running in a Docker container or "
    "Docker must be available on the system.",
    available=we_are_running_in_a_docker_container() or is_docker_available(),
)
@sanitize_path_arg("filename")
def execute_python_file(
    filename: Path, agent: Agent, args: list[str] | str = []
) -> str:
    """Execute a Python file in a Docker container and return the output

    Args:
        filename (Path): The path of the file to execute
        agent (Agent): The agent that is executing the command
        args (list, optional): The arguments with which to run the python script

    Returns:
        str: The output of the file
    """
    logger.info(
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
        logger.debug(
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

    logger.debug("AutoGPT is not running in a Docker container")
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
                logger.debug(f"Image '{image_name}' found locally")
            except ImageNotFound:
                logger.info(
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
                        logger.info(f"{status}: {progress}")
                    elif status:
                        logger.info(status)

            logger.debug(f"Creating new {image_name} container...")
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

        logger.debug(f"Running {file_path} in container {container.name}...")
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
        logger.warning(
            "Could not run the script in a container. "
            "If you haven't already, please install Docker: "
            "https://docs.docker.com/get-docker/"
        )
        raise CommandExecutionError(f"Could not run the script in a container: {e}")


def validate_command(command_line: str, config: Config) -> tuple[bool, bool]:
    """Check whether a command is allowed and whether it may be executed in a shell.

    If shell command control is enabled, we disallow executing in a shell, because
    otherwise the model could easily circumvent the command filter using shell features.

    Args:
        command_line (str): The command line to validate
        config (Config): The application config including shell command control settings

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
    "execute_shell",
    "Execute a Shell Command, non-interactive commands only",
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
@run_in_workspace
def execute_shell(command_line: str, agent: Agent) -> str:
    """Execute a shell command and return the output

    Args:
        command_line (str): The command line to execute
        agent (Agent): The agent that is executing the command

    Returns:
        str: The output of the command
    """
    allow_execute, allow_shell = validate_command(command_line, agent.legacy_config)
    if not allow_execute:
        logger.info(f"Command '{command_line}' not allowed")
        raise OperationNotAllowedError("This shell command is not allowed.")

    logger.info(
        f"Executing command '{command_line}' in working directory '{os.getcwd()}'"
    )

    result = subprocess.run(
        command_line if allow_shell else shlex.split(command_line),
        capture_output=True,
        shell=allow_shell,
    )
    output = f"STDOUT:\n{result.stdout.decode()}\nSTDERR:\n{result.stderr.decode()}"
    return output


@command(
    "execute_shell_popen",
    "Execute a Shell Command, non-interactive commands only",
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
@run_in_workspace
def execute_shell_popen(command_line: str, agent: Agent) -> str:
    """Execute a shell command with Popen and returns an english description
    of the event and the process id

    Args:
        command_line (str): The command line to execute
        agent (Agent): The agent that is executing the command

    Returns:
        str: Description of the fact that the process started and its id
    """
    allow_execute, allow_shell = validate_command(command_line, agent.legacy_config)
    if not allow_execute:
        logger.info(f"Command '{command_line}' not allowed")
        raise OperationNotAllowedError("This shell command is not allowed.")

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

    return f"Subprocess started with PID:'{str(process.pid)}'"


@command(
    "execute_interactive_shell",
    "Executes a Shell Command that needs interactivity and return the output.",
    {
        "command_line": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The command line to execute",
            required=True,
        ),
    },
    lambda config: config.execute_local_commands and not config.continuous_mode,
    "Either the agent is running in continuous mode, or "
    "you are not allowed to run local shell commands. To execute"
    " shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' "
    "in your config file: .env - do not attempt to bypass the restriction.",
)
@run_in_workspace
def execute_interactive_shell(command_line: str, agent: Agent) -> list[dict]:
    """Execute a shell command that requires interactivity and return the output.

    Args:
        command_line (str): The command line to execute
        agent (Agent): The agent that is executing the command

    Returns:
        list[dict]: The interaction between the user and the process,
        as a list of dictionaries:
        [
            {
                role: "user"|"system"|"error",
                content: "The content of the interaction."
            },
            ...
        ]
    """
    if not validate_command(command_line, agent.legacy_config):
        logger.info(f"Command '{command_line}' not allowed")
        return [{"role": "error", "content": "This Shell Command isn't allowed."}]

    if sys.platform == "win32":
        conversation = _exec_cross_platform(command_line)
    else:
        conversation = _exec_linux(command_line)

    return conversation


def _exec_linux(command_line: str) -> list[dict]:
    """
    Execute a linux shell command and return the output.

    Args:
        command_line (str): The command line to execute

    Returns:
        list[dict]: The interaction between the user and the process,
        as a list of dictionaries:
        [
            {
                role: "user"|"system"|"error",
                content: "The content of the interaction."
            },
            ...
        ]
    """
    import select

    process = subprocess.Popen(
        command_line,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # To capture the conversation, we'll read from one set of descriptors, save the output and write it to the other set descriptors.
    fd_map = {
        process.stdout.fileno(): ("system", sys.stdout.buffer),
        process.stderr.fileno(): ("error", sys.stderr.buffer),
        sys.stdin.fileno(): ("user", process.stdin),  # Already buffered
    }

    conversation = []

    while True:
        read_fds, _, _ = select.select(list(fd_map.keys()), [], [])
        input_fd = next(fd for fd in read_fds if fd in fd_map)
        role, output_buffer = fd_map[input_fd]

        input_buffer = os.read(input_fd, 1024)
        if input_buffer == b"":
            break
        output_buffer.write(input_buffer)
        output_buffer.flush()
        content = input_buffer.decode("utf-8")
        content = (
            content.replace("\r", "").replace("\n", " ").strip() if content else ""
        )
        conversation.append({"role": role, "content": content})

    try:
        process.wait(timeout=TIMEOUT_SECONDS)
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()
    except subprocess.TimeoutExpired:
        conversation.append(
            {"role": "error", "content": f"Timed out after {TIMEOUT_SECONDS} seconds."}
        )

    return conversation


def _exec_cross_platform(command_line: str) -> list[dict]:
    """
    Execute a shell command that requires interactivity and return the output.
    This can also work on linux, but is less native than the other function.

    Args:
        command_line (str): The command line to execute
        agent (Agent): The agent that is executing the command

    Returns:
        list[dict]: The interaction between the user and the process,
        as a list of dictionaries:
        [
            {
                role: "user"|"system"|"error",
                content: "The content of the interaction."
            },
            ...
        ]
    """
    from sarge import Capture, Command

    command = Command(
        command_line,
        stdout=Capture(buffer_size=1),
        stderr=Capture(buffer_size=1),
    )
    command.run(input=subprocess.PIPE, async_=True)

    # To capture the conversation, we'll read from one set of descriptors,
    # save the output and write it to the other set descriptors.
    fd_map = {
        command.stdout: ("system", sys.stdout.buffer),
        command.stderr: ("error", sys.stderr.buffer),
    }

    conversation = []

    while True:
        output = {fd: fd.read(timeout=0.1) for fd in fd_map.keys()}
        if not any(output.values()):
            break

        content = ""
        for fd, output_content in output.items():
            if output_content:
                output_content = (
                    output_content + b"\n"
                    if not output_content.endswith(b"\n")
                    else output_content
                )
                fd_map[fd][1].write(output_content)
                fd_map[fd][1].flush()

                content = output_content.decode("utf-8")
                content = (
                    content.replace("\r", "").replace("\n", " ").strip()
                    if content
                    else ""
                )
                conversation.append({"role": fd_map[fd][0], "content": content})

        if any(output.values()):
            prompt = "Response [None]: "
            os.write(sys.stdout.fileno(), prompt.encode("utf-8"))
            stdin = os.read(sys.stdin.fileno(), 1024)
            if stdin != b"":
                try:
                    command.stdin.write(stdin)
                    command.stdin.flush()
                    content = stdin.decode("utf-8")
                    content = (
                        content.replace("\r", "").replace("\n", " ").strip()
                        if content
                        else ""
                    )
                    conversation.append({"role": "user", "content": content})
                except (BrokenPipeError, OSError):
                    # Child process already exited
                    print("Command exited... returning.")

    try:
        command.wait(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        conversation.append(
            {"role": "error", "content": f"Timed out after {TIMEOUT_SECONDS} seconds."}
        )

    return conversation


@command(
    "ask_user",
    "Ask the user a series of questions and return the responses.",
    {
        "prompts": JSONSchema(
            type=JSONSchema.Type.ARRAY,
            items=JSONSchema(type=JSONSchema.Type.STRING),
            description="The questions to ask the user.",
            required=True,
        ),
    },
    lambda config: not config.continuous_mode,
    "The agent is running in continuous mode.",
)
def ask_user(prompts: list[str], agent: Agent) -> list[str]:
    """Ask the user a series of prompts and return the responses

    Args:
        prompts (list[str]): The prompts to ask the user
        agent (Agent): The agent that is executing the command

    Returns:
        list[str]: The responses from the user
    """

    from inputimeout import TimeoutOccurred, inputimeout

    results = []
    try:
        for prompt in prompts:
            response = inputimeout(prompt, timeout=TIMEOUT_SECONDS)
            results.append(response)
    except TimeoutOccurred:
        results.append(f"Timed out after {TIMEOUT_SECONDS} seconds.")

    return results
