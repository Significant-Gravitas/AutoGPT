import subprocess
import docker
import os

from config import Config

working_directory = "auto_gpt_workspace"
cfg = Config()

def execute_python_file(file):
    """Execute a Python file in a Docker container and return the output"""

    print (f"Executing file '{file}' in workspace '{working_directory}'")

    if not file.endswith(".py"):
        return "Error: Invalid file type. Only .py files are allowed."

    file_path = os.path.join(working_directory, file)

    if not os.path.isfile(file_path):
        return f"Error: File '{file}' does not exist."

    try:
        client = docker.from_env()

        # You can replace 'python:3.8' with the desired Python image/version
        # You can find available Python images on Docker Hub:
        # https://hub.docker.com/_/python
        container = client.containers.run(
            'python:3.10',
            f'python {file}',
            volumes={
                os.path.abspath(working_directory): {
                    'bind': '/workspace',
                    'mode': 'ro'}},
            working_dir='/workspace',
            stderr=True,
            stdout=True,
            detach=True,
        )

        output = container.wait()
        logs = container.logs().decode('utf-8')
        container.remove()

        # print(f"Execution complete. Output: {output}")
        # print(f"Logs: {logs}")

        return logs

    except Exception as e:
        return f"Error: {str(e)}"


def execute_command_on_console(command):
    """ Execute a command on console """

    if not cfg.command_line_access:
        print("Set COMMAND_LINE_ACCESS environment variable to True to use this feature.")
        return "execute_command_on_console is disabled for this session. Do not try again"

    if not command or not isinstance(command, str):
        return "Invalid command. Please provide a non-empty string."

    try:
        result = subprocess.run(command.split(), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, cwd=working_directory)
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error executing command: {result.stderr}"
    except Exception as e:
        return f"An error occurred while executing the command: {e}"
