import docker
from docker import errors
from pathlib import Path
from constants import WORKSPACE_DIR


def execute_python_file(file):
    workspace_folder = WORKSPACE_DIR

    print (f"Executing file '{file}' in workspace '{workspace_folder}'")

    if not file.endswith(".py"):
        return "Error: Invalid file type. Only .py files are allowed."

    file_path = Path(workspace_folder) / file

    if not file_path.is_file():
        return f"Error: File '{file}' does not exist."

    try:
        client = docker.from_env()

        # You can replace 'python:3.8' with the desired Python image/version
        # You can find available Python images on Docker Hub:
        # https://hub.docker.com/_/python
        container = client.containers.run(
            'python:3.10',
            f'python {file_path}',
            volumes={
                Path(workspace_folder).resolve(): {
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

        return logs

    except docker.errors.APIError as e:
        return f"Error while running container: {e}"
    except Exception as e:
        return f"Error: {str(e)}"
