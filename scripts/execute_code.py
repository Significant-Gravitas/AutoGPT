import docker
import os
import subprocess


WORKSPACE_FOLDER = "auto_gpt_workspace"


def execute_python_file(file):

    print (f"Executing file '{file}' in workspace '{WORKSPACE_FOLDER}'")

    if not file.endswith(".py"):
        return "Error: Invalid file type. Only .py files are allowed."

    file_path = os.path.join(WORKSPACE_FOLDER, file)

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
                os.path.abspath(WORKSPACE_FOLDER): {
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
    


def exec_shell(command_line):

    print (f"Executing command '{command_line}' in workspace '{WORKSPACE_FOLDER}'")

    args = command_line.split()
    base_path = os.getcwd()

    os.chdir(f"{base_path}/{WORKSPACE_FOLDER}")

    result = subprocess.run(args, capture_output=True)
    output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}";

    os.chdir(base_path)

    # print(f"Shell execution complete. Output: {output}")

    return output
