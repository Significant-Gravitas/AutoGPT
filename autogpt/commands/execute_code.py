"""Execute code in a Docker container"""
import ast
import importlib
import os
import pkgutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import docker
from docker.errors import ImageNotFound

from autogpt.commands.command import command
from autogpt.config import Config
from autogpt.logs import logger

# Exclusion list for imports built in python:3-alpine image
BUILT_IN_MODULES = set(
    [
        "sys",
        "os",
        "math",
        "random",
        "datetime",
        "json",
        "re",
        "subprocess",
        "time",
        "threading",
        "logging",
        "collections",
        "itertools",
        "functools",
        "operator",
        "pathlib",
        "shutil",
        "tempfile",
        "pickle",
        "io",
        "argparse",
        "typing",
        "unittest",
        "contextlib",
        "abc",
        "heapq",
        "bisect",
        "copy",
        "decimal",
        "fractions",
        "hashlib",
        "secrets",
        "statistics",
        "difflib",
        "doctest",
        "enum",
        "inspect",
        "traceback",
        "weakref",
        "gc",
        "mmap",
        "msvcrt",
        "winreg",
        "array",
        "audioop",
        "binascii",
        "cProfile",
        "concurrent.futures",
        "configparser",
        "csv",
        "ctypes",
        "dateutil",
        "dis",
        "fnmatch",
        "getopt",
        "glob",
        "gzip",
        "pdb",
        "pprint",
        "profile",
        "pstats",
        "queue",
        "socket",
        "sqlite3",
        "ssl",
        "struct",
        "tarfile",
        "telnetlib",
        "timeit",
        "tokenize",
        "uuid",
        "xml",
        "zipfile",
        "zlib",
    ]
)


def get_imports(path: str) -> List[str]:
    """
    Returns a list of libraries that need to be imported by the file at the given path.
    """
    with open(path, "r") as file:
        root = ast.parse(file.read(), path)

    imports = []
    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.Import):
            module_names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module_names = [node.module]
        else:
            continue

        for name in module_names:
            # Exclude built-in modules
            if name not in BUILT_IN_MODULES:
                imports.append(name)

    return imports


def write_requirements(filename: str, requirements_filepath: str):
    imports = get_imports(filename)

    with open(requirements_filepath, "w") as file:
        for module in imports:
            file.write(module + "\n")


def pull_docker_image(client: docker.DockerClient, image_name: str):
    try:
        client.images.get(image_name)
        logger.warn(f"Image '{image_name}' found locally")
    except ImageNotFound:
        logger.info(f"Image '{image_name}' not found locally, pulling from Docker Hub")
        low_level_client = docker.APIClient()
        for line in low_level_client.pull(image_name, stream=True, decode=True):
            status = line.get("status")
            progress = line.get("progress")
            if status and progress:
                logger.info(f"{status}: {progress}")
            elif status:
                logger.info(status)


def exec_command(container, command):
    exit_code, output = container.exec_run(command)
    if exit_code != 0:
        return exit_code, output.decode("utf-8")
    return exit_code, output.decode("utf-8")


@command("execute_python_file", "Execute Python File", '"filename": "<filename>"')
def execute_python_file(filename: str, config: Config) -> str:
    """Execute a Python file in a Docker container and return the output

    Args:
        filename (str): The name of the file to execute

    Returns:
        str: The output of the file
    """
    logger.info(f"Executing file '{filename}'")

    if not filename.endswith(".py"):
        return "Error: Invalid file type. Only .py files are allowed."

    if not os.path.isfile(filename):
        return f"Error: File '{filename}' does not exist."

    if we_are_running_in_a_docker_container():
        result = subprocess.run(
            ["python", filename], capture_output=True, encoding="utf8"
        )
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}"

    try:
        client = docker.from_env()

        # Here another python docker image can be chosen
        image_name = "python:3.9"
        pull_docker_image(client, image_name)

        # Create the list of commands to be run
        commands = [
            ["apt-get", "update"],
            ["apt-get", "-qq", "install", "-y", "gcc", "python3-dev", "apt-utils"],
        ]

        # Parse the system_requirements.txt file and add the apt-get install commands
        system_requirements_filepath = Path(filename).parent / "system_requirements.txt"
        if system_requirements_filepath.exists():
            with open(system_requirements_filepath) as f:
                for line in f:
                    package = line.strip()
                    commands.append(["apt-get", "-qq", "install", "-y", package])

        # Add the Python package installation command
        requirements_filepath = Path(filename).parent / "requirements.txt"
        write_requirements(filename, requirements_filepath)
        commands.append(
            [
                "python",
                "-m",
                "pip",
                "install",
                "--quiet",
                str(requirements_filepath.relative_to(config.workspace_path)),
            ]
        )

        # Add the Python script execution command
        commands.append(
            ["python", str(Path(filename).relative_to(config.workspace_path))]
        )

        # Create and start a container with an interactive bash session
        container = client.containers.run(
            image_name,
            command="/bin/bash",
            volumes={
                config.workspace_path: {
                    "bind": "/workspace",
                    "mode": "ro",
                }
            },
            working_dir="/workspace",
            tty=True,  # allocate a pseudo-TTY
            detach=True,
        )

        output = ""
        # Run the commands
        for cmd in commands:
            exit_code, output = container.exec_run(cmd)
            if exit_code != 0:
                container.stop()
                container.remove()
                return f"Error: Command '{' '.join(cmd)}' failed with exit code {exit_code}. Output was: {output.decode('utf-8')}"

        container.stop()
        container.remove()

        # If everything is successful, return the output of the Python script
        return output.decode("utf-8")

    except docker.errors.BuildError as build_err:
        # This exception will be raised if there's an error installing dependencies
        logger.error(f"Error installing dependencies: {str(build_err)}")
        raise build_err

    except docker.errors.APIError as api_err:
        # This exception will be raised if there's an error executing the Python file
        logger.error(f"Error executing Python file: {str(api_err)}")
        raise api_err

    except docker.errors.DockerException as docker_err:
        # This exception covers other Docker-related errors
        logger.warn(
            "Could not run the script in a container. If you haven't already, please install Docker https://docs.docker.com/get-docker/"
        )
        raise docker_err

    except Exception as e:
        # This exception covers any other unexpected errors
        logger.error(f"Unexpected error occurred: {str(e)}")
        raise e


def validate_command(command: str, config: Config) -> bool:
    """Validate a command to ensure it is allowed

    Args:
        command (str): The command to validate

    Returns:
        bool: True if the command is allowed, False otherwise
    """
    tokens = command.split()

    if not tokens:
        return False

    if config.deny_commands and tokens[0] not in config.deny_commands:
        return False

    for keyword in config.allow_commands:
        if keyword in tokens:
            return True
    if config.allow_commands:
        return False

    return True


@command(
    "execute_shell",
    "Execute Shell Command, non-interactive commands only",
    '"command_line": "<command_line>"',
    lambda cfg: cfg.execute_local_commands,
    "You are not allowed to run local shell commands. To execute"
    " shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' "
    "in your config file: .env - do not attempt to bypass the restriction.",
)
def execute_shell(command_line: str, config: Config) -> str:
    """Execute a shell command and return the output

    Args:
        command_line (str): The command line to execute

    Returns:
        str: The output of the command
    """
    if not validate_command(command_line, config):
        logger.info(f"Command '{command_line}' not allowed")
        return "Error: This Shell Command is not allowed."

    current_dir = Path.cwd()
    # Change dir into workspace if necessary
    if not current_dir.is_relative_to(config.workspace_path):
        os.chdir(config.workspace_path)

    logger.info(
        f"Executing command '{command_line}' in working directory '{os.getcwd()}'"
    )

    result = subprocess.run(command_line, capture_output=True, shell=True)
    output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Change back to whatever the prior working dir was

    os.chdir(current_dir)
    return output


@command(
    "execute_shell_popen",
    "Execute Shell Command, non-interactive commands only",
    '"command_line": "<command_line>"',
    lambda config: config.execute_local_commands,
    "You are not allowed to run local shell commands. To execute"
    " shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' "
    "in your config. Do not attempt to bypass the restriction.",
)
def execute_shell_popen(command_line, config: Config) -> str:
    """Execute a shell command with Popen and returns an english description
    of the event and the process id

    Args:
        command_line (str): The command line to execute

    Returns:
        str: Description of the fact that the process started and its id
    """
    if not validate_command(command_line, config):
        logger.info(f"Command '{command_line}' not allowed")
        return "Error: This Shell Command is not allowed."

    current_dir = os.getcwd()
    # Change dir into workspace if necessary
    if config.workspace_path not in current_dir:
        os.chdir(config.workspace_path)

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
