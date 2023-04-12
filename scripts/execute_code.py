import docker
import os


def execute_python_file(file):
    """Execute a Python file in a Docker container and return the output"""
    workspace_folder = "auto_gpt_workspace"

    print (f"Executing file '{file}' in workspace '{workspace_folder}'")

    if not file.endswith(".py"):
        return "Error: Invalid file type. Only .py files are allowed."

    file_path = os.path.join(workspace_folder, file)

    if not os.path.isfile(file_path):
        return f"Error: File '{file}' does not exist."

    try:
        client = docker.from_env()

        image_name = 'python:3.10'
        try:
            client.images.get(image_name)
            print(f"Image '{image_name}' found locally")
        except docker.errors.ImageNotFound:
            print(f"Image '{image_name}' not found locally, pulling from Docker Hub")
            # Use the low-level API to stream the pull response
            low_level_client = docker.APIClient()
            for line in low_level_client.pull(image_name, stream=True, decode=True):
                # Print the status and progress, if available
                status = line.get('status')
                progress = line.get('progress')
                if status and progress:
                    print(f"{status}: {progress}")
                elif status:
                    print(status)

        # You can replace 'python:3.8' with the desired Python image/version
        # You can find available Python images on Docker Hub:
        # https://hub.docker.com/_/python
        container = client.containers.run(
            image_name,
            f'python {file}',
            volumes={
                os.path.abspath(workspace_folder): {
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
