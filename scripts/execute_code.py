import docker
import os
import uuid
from file_operations import write_to_file, safe_join

def execute_python_file(file):
    """Execute a Python file in a Docker container and return the output"""
    workspace_folder = "auto_gpt_workspace"

    print (f"Executing file '{file}' in workspace '{workspace_folder}'")

    if not file.endswith(".py"):
        return "Error: Invalid file type. Only .py files are allowed."

    file_path = safe_join(workspace_folder, file)

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
                os.path.abspath(workspace_folder): {
                    'bind': '/workspace',
                    'mode': 'rw'}},
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
    
def execute_python_code(code):
    """Execute Python code in a Docker container and return the output"""

    # Generate a UUID
    unique_id = str(uuid.uuid4())

    # Create a filename with the UUID as the filename
    filename = unique_id + ".py"

    workspace_folder = "auto_gpt_workspace"
    
    write_to_file(filename, code)

    print (f"Saving code to file for history. Filename:'{filename}' in Workspace Folder:'{workspace_folder}'.\nCode:\n'{code}'")
    
    return execute_python_file(filename);
