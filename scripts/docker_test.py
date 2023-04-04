# test_docker.py
import os
import sys


from docker_agent import start_docker_container, run_command_in_docker_container, stop_docker_container
from config import Config

cfg = Config()

def test_docker_implementation():
    # Test start_docker_container
    image = "alpine:latest"
    command = "/bin/sh"
    volume_name = "test-volume"
    print("Starting Docker container...")
    container_name = start_docker_container(image, command, volume_name)
    print(container_name)

    # Test run_command_in_docker_container
    test_command = "echo 'Hello from Docker!'"
    print("Running command in Docker container...")
    command_output = run_command_in_docker_container(container_name, test_command)
    print(command_output)

    # Test stop_docker_container
    print("Stopping Docker container and cleaning up resources...")
    stop_result = stop_docker_container(container_name, volume_name)
    print(stop_result)

test_docker_implementation()
