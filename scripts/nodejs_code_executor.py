from docker_executor import DockerExecutor


class NodeJsCodeExecutor:

    def __init__(self):
        # image = 'node:18-buster-slim'
        image = 'fsamir/nodejs-puppeteer:dev'
        self.docker_executor = DockerExecutor(image, 'bash -i -c node', ['.js'])

    def execute(self, file_name):
        """Execute a JavaScript file in a Docker container and return the output"""

        return self.docker_executor.execute(file_name)
