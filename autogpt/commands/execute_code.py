"""Execute code in a Docker container"""
import os
import subprocess

import docker
from docker.errors import ImageNotFound

from autogpt.commands.command import command
from autogpt.config import Config
from autogpt.workspace import WORKSPACE_PATH, path_in_workspace

CFG = Config()


@command("execute_python_file", "Execute Python File", '"filename": "<filename>"')
def execute_python_file(filename: str) -> str:
    """Execute a Python file in a Docker container and return the output

    Args:
        filename (str): The name of the file to execute

    Returns:
        str: The output of the file
    """
    file = filename
    print(f"Executing file '{file}' in workspace '{WORKSPACE_PATH}'")

    if not file.endswith(".py"):
        return "Error: Invalid file type. Only .py files are allowed."

    file_path = path_in_workspace(file)

    if not os.path.isfile(file_path):
        return f"Error: File '{file}' does not exist."

    if we_are_running_in_a_docker_container():
        result = subprocess.run(
            f"python {file_path}", capture_output=True, encoding="utf8", shell=True
        )
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}"

    try:
        client = docker.from_env()

        # You can replace this with the desired Python image/version
        # You can find available Python images on Docker Hub:
        # https://hub.docker.com/_/python
        image_name = "python:3-alpine"
        try:
            client.images.get(image_name)
            print(f"Image '{image_name}' found locally")
        except ImageNotFound:
            print(f"Image '{image_name}' not found locally, pulling from Docker Hub")
            # Use the low-level API to stream the pull response
            low_level_client = docker.APIClient()
            for line in low_level_client.pull(image_name, stream=True, decode=True):
                # Print the status and progress, if available
                status = line.get("status")
                progress = line.get("progress")
                if status and progress:
                    print(f"{status}: {progress}")
                elif status:
                    print(status)

        container = client.containers.run(
            image_name,
            f"python {file}",
            volumes={
                os.path.abspath(WORKSPACE_PATH): {
                    "bind": "/workspace",
                    "mode": "ro",
                }
            },
            working_dir="/workspace",
            stderr=True,
            stdout=True,
            detach=True,
        )

        container.wait()
        logs = container.logs().decode("utf-8")
        container.remove()

        # print(f"Execution complete. Output: {output}")
        # print(f"Logs: {logs}")

        return logs

    except docker.errors.DockerException as e:
        print(
            "Could not run the script in a container. If you haven't already, please install Docker https://docs.docker.com/get-docker/"
        )
        return f"Error: {str(e)}"

    except Exception as e:
        return f"Error: {str(e)}"


@command(
    "execute_shell",
    "Execute Shell Command, non-interactive commands only",
    '"command_line": "<command_line>"',
    CFG.execute_local_commands,
    "You are not allowed to run local shell commands. To execute"
    " shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' "
    "in your config. Do not attempt to bypass the restriction.",
)
def execute_shell(command_line: str) -> str:
    """Execute a shell command and return the output

    Args:
        command_line (str): The command line to execute

    Returns:
        str: The output of the command
    """

    if not CFG.execute_local_commands:
        return (
            "You are not allowed to run local shell commands. To execute"
            " shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' "
            "in your config. Do not attempt to bypass the restriction."
        )
    current_dir = os.getcwd()
    # Change dir into workspace if necessary
    if str(WORKSPACE_PATH) not in current_dir:
        os.chdir(WORKSPACE_PATH)

    print(f"Executing command '{command_line}' in working directory '{os.getcwd()}'")

    result = subprocess.run(command_line, capture_output=True, shell=True)
    output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Change back to whatever the prior working dir was

    os.chdir(current_dir)


@command(
    "execute_shell_popen",
    "Execute Shell Command, non-interactive commands only",
    '"command_line": "<command_line>"',
    CFG.execute_local_commands,
    "You are not allowed to run local shell commands. To execute"
    " shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' "
    "in your config. Do not attempt to bypass the restriction.",
)
def execute_shell_popen(command_line) -> str:
    """Execute a shell command with Popen and returns an english description
    of the event and the process id

    Args:
        command_line (str): The command line to execute

    Returns:
        str: Description of the fact that the process started and its id
    """
    current_dir = os.getcwd()
    # Change dir into workspace if necessary
    if str(WORKSPACE_PATH) not in current_dir:
        os.chdir(WORKSPACE_PATH)

    print(f"Executing command '{command_line}' in working directory '{os.getcwd()}'")

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
