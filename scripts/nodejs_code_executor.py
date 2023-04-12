from docker_executor import DockerExecutor


class NodeJsCodeExecutor:

    def __init__(self):
        self.docker_executor = DockerExecutor('node:18-buster-slim', 'node', ['.js'])

    def execute(self, file_name):
        """Execute a JavaScript file in a Docker container and return the output"""

        return self.docker_executor.execute(file_name)
