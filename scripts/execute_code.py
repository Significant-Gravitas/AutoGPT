import os

import docker


def execute_python_file(file):
    workspace_folder = "auto_gpt_workspace"

    print(f"Executing file '{file}' in workspace '{workspace_folder}'")

    if not file.endswith(".py"):
        return "Error: Invalid file type. Only .py files are allowed."

    file_path = os.path.join(workspace_folder, file)

    if not os.path.isfile(file_path):
        return f"Error: File '{file}' does not exist."

    try:
        client = docker.from_env()

        env_vars = {key: value for key, value in os.environ.items()}
        print(f"Environment variables: {env_vars}")

        container = client.containers.run(
            'fsamir/python:3.11',
            f'python {file}',
            volumes={os.path.abspath(workspace_folder): {
                'bind': '/workspace',
                'mode': 'rw'}
            },
            environment=env_vars,
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
