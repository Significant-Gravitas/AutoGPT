"""Execute code in a Docker container"""
import os
import subprocess

import docker
from docker.errors import ImageNotFound

from autogpt.workspace import path_in_workspace, WORKSPACE_PATH


def execute_python_file(file: str):
    """Execute a Python file in a Docker container and return the output

    Args:
        file (str): The name of the file to execute

    Returns:
        str: The output of the file
    """

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

        image_name = "python:3.10"
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

        # You can replace 'python:3.8' with the desired Python image/version
        # You can find available Python images on Docker Hub:
        # https://hub.docker.com/_/python
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

    except Exception as e:
        return f"Error: {str(e)}"


def execute_shell(command_line: str) -> str:
    """Execute a shell command and return the output

    Args:
        command_line (str): The command line to execute

    Returns:
        str: The output of the command
    """
    current_dir = os.getcwd()
    # Change dir into workspace if necessary
    if str(WORKSPACE_PATH) not in current_dir:
        os.chdir(WORKSPACE_PATH)

    print(f"Executing command '{command_line}' in working directory '{os.getcwd()}'")

    result = subprocess.run(command_line, capture_output=True, shell=True)
    output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Change back to whatever the prior working dir was

    os.chdir(current_dir)

    return output


def we_are_running_in_a_docker_container() -> bool:
    """Check if we are running in a Docker container

    Returns:
        bool: True if we are running in a Docker container, False otherwise
    """
    return os.path.exists("/.dockerenv")
