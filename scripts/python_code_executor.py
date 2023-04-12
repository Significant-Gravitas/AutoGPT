import os

import docker

from docker_executor import DockerExecutor

class PythonCodeExecutor:

    def __init__(self):
        self.docker_executor = DockerExecutor('fsamir/python:3.11', 'python', ['.py'])

    def execute(self, file_name):
        """Execute a Python file in a Docker container and return the output"""

        return self.docker_executor.execute(file_name)