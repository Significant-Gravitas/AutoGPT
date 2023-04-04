import docker
import os
from config import Config

cfg = Config()

client = docker.from_env()

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
    if volume_name:
        volume = create_docker_volume(volume_name)
    else:
        volume = None

    container = create_docker_container(image, command, volume)
    if isinstance(container, str):
        return container, None

    start_result = run_docker_container(container)

    return start_result, container, volume

def interact_with_container(container, command):
    try:
        exec_instance = container.exec_create(cmd=command, stdout=True, stderr=True)
        output = container.exec_start(exec_id=exec_instance['Id'])
        logs = output.decode("utf-8")
        return logs
    except Exception as e:
        return f"Error interacting with container: {str(e)}"

if __name__ == "__main__":
    image = cfg.docker_image
    command = "/bin/sh"  # Use /bin/sh to keep the container running
    volume_name = "test-volume"

    start_result, container, volume = setup_and_run_docker(image, command, volume_name)

    print("Start result:")
    print(start_result)

    # Interact with the running container
    # ...

    # Stop and clean up the container and volume when done
    cleanup_result = stop_and_cleanup_docker(container, volume)

    print("Cleanup result:")
    print(cleanup_result)

