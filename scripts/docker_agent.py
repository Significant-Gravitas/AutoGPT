import docker
import os
from config import Config
import io

cfg = Config()

client = docker.from_env()

MAX_COMMAND_LENGTH = 1000

def create_docker_image(image_name, image_tag, dockerfile: str):
    try:
        file = io.BytesIO(bytes(dockerfile, 'utf-8'))
        image = client.images.build(fileobj=file, tag=f"{image_name}:{image_tag}")
        return image
    except Exception as e:
        return f"Error creating image: {str(e)}"

def create_docker_volume(volume_name):
    try:
        volume = client.volumes.create(name=volume_name)
        return volume
    except Exception as e:
        return f"Error creating volume: {str(e)}"

def create_docker_container(image, command, volume=None):
    try:
        if volume:
            container = client.containers.create(
                image,
                command=command,
                volumes={volume.name: {'bind': '/data', 'mode': 'rw'}},
                tty=True,
                stdin_open=True,
                detach=True,
            )
        else:
            container = client.containers.create(image, command=command, tty=True, stdin_open=True, detach=True)
        return container
    except Exception as e:
        return f"Error creating container: {str(e)}"

def run_docker_container(container):
    try:
        container.start()
        return f"Container {container.id} started"
    except Exception as e:
        return f"Error running container: {str(e)}"

def stop_and_cleanup_docker(container, volume=None):
    try:
        container.stop()
        container.remove()
        if volume:
            volume.remove()
        return "Stopped and cleaned up successfully"
    except Exception as e:
        return f"Error during stopping and cleanup: {str(e)}"

def setup_and_run_docker(image, command, volume_name=None):
    # pull image if not exists
    pull_docker_image(image)

    if volume_name:
        volume = create_docker_volume(volume_name)
    else:
        volume = None

    container = create_docker_container(image, command, volume)
    if isinstance(container, str):
        return container, None, None

    start_result = run_docker_container(container)

    return start_result, container, volume

def interact_with_container(container, command):
    try:
        exit_code, output_stream = container.exec_run(cmd=command, stdout=True, stderr=True, stream=True)

        output = ""
        for chunk in output_stream:
            output += chunk.decode('utf-8')
        if len(output) > MAX_COMMAND_LENGTH:
            # output only the last MAX_COMMAND_LENGTH characters
            output = output[-MAX_COMMAND_LENGTH:]
        if exit_code != None:
            return f"Error executing command. Exit code: {exit_code}. Output: {output}"
        else:
            return output
    except Exception as e:
        return f"Error interacting with container: {str(e)}"

def pull_docker_image(image_name):
    try:
        print(f"Pulling Docker image: {image_name}")
        client.images.pull(image_name)
    except Exception as e:
        return f"Error pulling Docker image: {str(e)}"

# wrapper for LLM

def start_docker_container(image, command, volume_name=None):
    start_result, container, volume = setup_and_run_docker(image, command, volume_name)
    if isinstance(container, docker.models.containers.Container):
        return f"{container.name}"
    else:
        return start_result

def run_command_in_docker_container(container_name, command):
    container = client.containers.get(container_name)
    logs = interact_with_container(container, command)
    return f"Command output:\n{logs}"

def run_multiple_commands_in_docker_container(container_name, commands):
    container = client.containers.get(container_name)
    output = ""
    for command in commands:
        output += "command: " + command
        output += interact_with_container(container, command)
    return f"Command output:\n{output}"

def stop_docker_container(container_name, volume_name=None):
    container = client.containers.get(container_name)
    if volume_name:
        volume = client.volumes.get(volume_name)
    else:
        volume = None
    cleanup_result = stop_and_cleanup_docker(container, volume)
    return f"Stopped and cleaned up container: {container_name}\n{cleanup_result}"

def build_docker_image(image_name, tag_name, dockerfile):
    image = create_docker_image(image_name, tag_name, dockerfile)
    if isinstance(image, docker.models.images.Image):
        return f"Image {image_name}:{tag_name} built successfully"
    else:
        return image

